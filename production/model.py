from collections import defaultdict
from math import floor, log2
from typing import BinaryIO, Dict, List, Literal, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from miditoolkit import Instrument, MidiFile, Note, TempoChange

AnyField = Union[int, str]
AttentionKind = Union[Literal["causal"], Literal["prefix"]]
note_map_record_dtype = np.dtype([("start", np.int16), ("end", np.int16)])


class MIDITokenizer:
    class Token(NamedTuple):
        """NamedTuple for MIDI compound token.
        One compound token represents one note. Each field is a feature of the note."""

        bar: AnyField
        position: AnyField
        duration: AnyField
        pitch: AnyField
        tempo: AnyField

    def __init__(self, granularity: int = 64, max_bar: int = 128, pitch_range: Tuple[int, int] = (0, 128)) -> None:
        """Initialize a MIDITokenizer instance.
        Args:
            granularity: The number of units per bar. Defaults to 64 (64-th note).
            max_bar: The maximum number of bar token to use. Exceeded ones will be mod by the number. Defaults to 128.
            pitch_range: The range of pitch token to use. Defaults to (0, 128)."""
        self.granularity = granularity
        self.max_bar = max_bar

        # define bins for each field
        self.pitch_range = range(pitch_range[0], pitch_range[1])
        self.ticks_per_bar = 1920
        self.units_per_bar = granularity
        self.ticks_per_beat = self.ticks_per_bar // 4
        self.ticks_per_unit = self.ticks_per_bar // self.units_per_bar
        self.ticks_per_triplet_unit = self.ticks_per_unit // 3 * 4

        double_positions = set(range(0, self.ticks_per_bar, self.ticks_per_unit))
        triplet_positions = set(range(0, self.ticks_per_bar, self.ticks_per_triplet_unit))
        self.position_bins = sorted(double_positions | triplet_positions)

        double_duration = set(range(self.ticks_per_unit, self.ticks_per_bar + 1, self.ticks_per_unit))
        triplet_ratio = floor(log2(self.granularity / 3))
        triplet_duration = set([self.ticks_per_bar // (3 * 2**r) for r in range(triplet_ratio + 1)])
        self.duration_bins = sorted(double_duration | triplet_duration)

        self.tempo_bins = list(range(60, 180 + 1, 10))
        self.default_tempo = 120

        self.vocabularies: Dict[str, List[int]] = {}
        self.define_vocabularies()
        self.field_names = self.Token._fields
        self.field_indices = {name: index for index, name in enumerate(self.field_names)}
        self.vocab_sizes = [len(self.vocabularies[field_name]) for field_name in self.field_names]
        self.field_sizes = list(self.vocab_sizes)  # will be modified when adding special tokens

        self.define_special_tokens()
        self.build_encoder_decoder()
        self.set_special_token_ids()

    def define_vocabularies(self) -> None:
        self.vocabularies["bar"] = list(range(self.max_bar))
        self.vocabularies["position"] = self.position_bins
        self.vocabularies["duration"] = self.duration_bins
        self.vocabularies["pitch"] = list(self.pitch_range)
        self.vocabularies["tempo"] = self.tempo_bins

    def define_special_tokens(self) -> None:
        self.bos_token_str = "<BOS>"
        self.eos_token_str = "<EOS>"
        self.pad_token_str = "<PAD>"
        self.sep_token_str = "<SEP>"
        self.cls_token_str = "<CLS>"
        self.mask_token_str = "[MASK]"
        self.long_mask_token_str = "[lMASK]"
        self.seg_token_str = "<SEG>"

        self.bos_token = self.Token(*[self.bos_token_str] * len(self.field_names))
        self.eos_token = self.Token(*[self.eos_token_str] * len(self.field_names))
        self.pad_token = self.Token(*[self.pad_token_str] * len(self.field_names))
        self.sep_token = self.Token(*[self.sep_token_str] * len(self.field_names))
        self.cls_token = self.Token(*[self.cls_token_str] * len(self.field_names))
        self.mask_token = self.Token(*[self.mask_token_str] * len(self.field_names))
        self.long_mask_token = self.Token(*[self.long_mask_token_str] * len(self.field_names))
        self.seg_token = self.Token(*[self.seg_token_str] * len(self.field_names))

        self.special_token_str = [
            self.bos_token_str,
            self.eos_token_str,
            self.pad_token_str,
            self.sep_token_str,
            self.cls_token_str,
            self.mask_token_str,
            self.long_mask_token_str,
            self.seg_token_str,
        ]

    def build_encoder_decoder(self) -> None:
        self.encoder: Dict[str, Dict[AnyField, int]] = {
            field_name: {field: index for index, field in enumerate(self.vocabularies[field_name])}
            for field_name in self.field_names
        }
        self.decoder: Dict[str, Dict[int, AnyField]] = {
            field_name: {index: field for index, field in enumerate(self.vocabularies[field_name])}
            for field_name in self.field_names
        }

        # add special tokens to the encoder and decoder
        for field_index, field_name in enumerate(self.field_names):
            for i, token_str in enumerate(self.special_token_str):
                token_id = len(self.vocabularies[field_name]) + i
                self.encoder[field_name][token_str] = token_id
                self.decoder[field_name][token_id] = token_str
            self.field_sizes[field_index] += len(self.special_token_str)

    def set_special_token_ids(self) -> None:
        self.bos_token_ids = self.convert_token_to_id(self.bos_token)
        self.eos_token_ids = self.convert_token_to_id(self.eos_token)
        self.pad_token_ids = self.convert_token_to_id(self.pad_token)
        self.sep_token_ids = self.convert_token_to_id(self.sep_token)
        self.cls_token_ids = self.convert_token_to_id(self.cls_token)
        self.mask_token_ids = self.convert_token_to_id(self.mask_token)
        self.long_mask_token_ids = self.convert_token_to_id(self.long_mask_token)
        self.seg_token_ids = self.convert_token_to_id(self.seg_token)

    def tokenize(self, midi: MidiFile, add_segment_token: bool = True) -> Tuple[List[Token], np.ndarray]:
        assert len(midi.instruments) == 1, "Only support single instrument midi file."

        notes = midi.instruments[0].notes
        segment_note_indices = self.get_segment_note_indices(midi) if add_segment_token else []

        current_tempo_index = 0
        tempo_changes = self.get_tempo_changes(midi)
        current_token_index = 0
        tokens: List[self.Token] = []
        note_map = np.zeros(len(notes), dtype=note_map_record_dtype)
        for note_index, note in enumerate(notes):
            # change current tempo if current note is after the next tempo change
            if (
                current_tempo_index < len(tempo_changes) - 1
                and note.start >= tempo_changes[current_tempo_index + 1].time
            ):
                current_tempo_index += 1

            bar = (note.start // self.ticks_per_bar) % self.max_bar
            position = self._find_nearest(self.position_bins, note.start % self.ticks_per_bar)
            duration = self._find_nearest(self.duration_bins, note.end - note.start)
            tempo = self._find_nearest(self.tempo_bins, tempo_changes[current_tempo_index].tempo)
            tokens.append(self.Token(bar, position, duration, note.pitch, tempo))

            # if note is a segment note, add a segment token
            if note_index in segment_note_indices:
                tokens.append(self.seg_token)
                note_map[note_index] = (current_token_index, current_token_index + 2)
                current_token_index += 2
            else:
                note_map[note_index] = (current_token_index, current_token_index + 1)
                current_token_index += 1

        return tokens, note_map

    def detokenize(self, tokens: List[Token], velocity: int = 100) -> MidiFile:
        midi = MidiFile()
        notes = []
        current_tempo = self.default_tempo
        midi.tempo_changes = [TempoChange(tempo=current_tempo, time=0)]
        for token in tokens:
            if any([field in self.special_token_str for field in token]):
                continue
            start = token.bar * self.ticks_per_bar + token.position
            end = token.bar * self.ticks_per_bar + token.position + token.duration
            note = Note(velocity=velocity, pitch=token.pitch, start=start, end=end)
            notes.append(note)

            # add tempo change if tempo changes
            if token.tempo != current_tempo:
                current_tempo = token.tempo
                midi.tempo_changes.append(TempoChange(tempo=current_tempo, time=start))

        instrument = Instrument(program=0)
        instrument.notes.extend(notes)
        midi.instruments.append(instrument)
        return midi

    def convert_tokens_to_ids(self, tokens: List[Token]) -> np.ndarray:
        token_ids = np.zeros((len(tokens), len(self.field_names)), dtype=np.int16)
        for index, token in enumerate(tokens):
            for field_index, field_name in enumerate(self.field_names):
                field = token[field_index]
                token_ids[index, field_index] = (
                    self.encoder[field_name][field] if field is not None else self.pad_token_ids[field_index]
                )
        return token_ids

    def convert_ids_to_tokens(self, tokens: np.ndarray) -> List[Token]:
        assert tokens.ndim == 2, "tokens should be 2D array."
        length, field_count = tokens.shape
        assert field_count == len(self.field_names), "field count should be equal to field names."

        result: List[self.Token] = []
        for index in range(length):
            fields = []
            for field_index, field_name in enumerate(self.field_names):
                token = tokens[index, field_index]
                field = self.decoder[field_name].get(token, None)
                fields.append(field)
            result.append(self.Token(*fields))
        return result

    def convert_token_to_id(self, token: Token) -> np.ndarray:
        return self.convert_tokens_to_ids([token])[0]

    def convert_id_to_token(self, token: np.ndarray) -> Token:
        return self.convert_ids_to_tokens(np.expand_dims(token, axis=0))[0]

    def encode(self, midi: MidiFile, add_segment_token: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        tokens, note_map = self.tokenize(midi, add_segment_token)
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids, note_map

    def decode(self, token_ids: np.ndarray) -> MidiFile:
        tokens = self.convert_ids_to_tokens(token_ids)
        return self.detokenize(tokens)

    def get_segment_note_indices(self, midi: MidiFile) -> List[int]:
        """Get indices of the segment notes of the midi file.
        The segment notes are:
        (1) the longest notes in each bar (longer than quarter note),
        (2) notes following at least eighth rest."""

        def filter_consecutive_notes(note_indices: List[int], notes: List[Note]):
            if len(note_indices) <= 1:
                return note_indices

            note_indices = np.array(note_indices)
            consecutive_mask = (note_indices[1:] - note_indices[:-1]) == 1
            result = []
            index = 0
            while index < len(note_indices) - 1:
                note_index = note_indices[index]
                if not consecutive_mask[index]:
                    result.append(note_index)
                    if index == len(note_indices) - 2:
                        # add the last note
                        result.append(note_indices[-1])
                    index += 1
                else:
                    # two consecutive notes
                    cur_note = notes[note_index]
                    next_note = notes[note_indices[index + 1]]
                    distance = next_note.start - cur_note.end
                    # if the distance between two consecutive notes is greater than eighth rest,
                    # or the first note is longer than the second note, keep the first note;
                    # otherwise, keep the second note.
                    if distance >= self.ticks_per_beat // 2:
                        result.append(note_index)
                    elif cur_note.end - cur_note.start > next_note.end - next_note.start:
                        result.append(note_index)
                    else:
                        result.append(note_indices[index + 1])
                    index += 2
            return result

        notes = midi.instruments[0].notes

        # get the longest notes in each bar (longer than quarter note)
        bar_dict = defaultdict(list)
        for note in notes:
            bar_dict[note.start // self.ticks_per_bar].append(note)
        long_notes = []
        for bar_notes in bar_dict.values():
            max_duration = max(x.end - x.start for x in bar_notes)
            if max_duration > self.ticks_per_beat:
                long_notes.extend(note for note in bar_notes if note.end - note.start == max_duration)
        long_note_indices = sorted(notes.index(note) for note in long_notes)
        long_note_indices = filter_consecutive_notes(long_note_indices, notes)

        # get notes following at least eighth rest
        rest_note_indices = [
            i
            for i, note in enumerate(notes)
            if i == len(notes) - 1 or notes[i + 1].start - note.end >= self.ticks_per_beat // 2
        ]

        # combine long notes and rest notes, then filter out consecutive notes
        segment_note_indices = sorted(set(long_note_indices) | set(rest_note_indices))
        segment_note_indices = filter_consecutive_notes(segment_note_indices, notes)

        # if the first segment note is the first note, and is too close to the second note, remove the first note
        if (
            len(segment_note_indices) >= 2
            and segment_note_indices[0] == 0
            and notes[1].start - notes[0].end < self.ticks_per_beat // 2
        ):
            segment_note_indices = segment_note_indices[1:]

        return segment_note_indices

    def get_tempo_changes(self, midi: MidiFile) -> List[TempoChange]:
        # sort and deduplicate tempo changes
        tempo_changes = midi.tempo_changes
        if len(tempo_changes) == 0:
            tempo_changes = [TempoChange(tempo=self.default_tempo, time=0)]
        elif len(tempo_changes) > 1:
            tempo_changes = sorted(midi.tempo_changes, key=lambda x: x.time)
            tempo_changes = [
                tempo_changes[i]
                for i in range(len(tempo_changes))
                if i == len(tempo_changes) - 1 or tempo_changes[i].time != tempo_changes[i + 1].time
            ]
        return tempo_changes

    def _find_nearest(self, bins: List[int], value: int) -> int:
        """Find the nearest bin to the value."""
        return min(bins, key=lambda x: abs(x - value))


class CompoundTokenFuser(nn.Module):
    """Fuses multiple token embeddings into a single embedding."""

    def __init__(self, tokenizer: MIDITokenizer, embedding_dim: Dict[str, int], model_dim: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.num_features = len(tokenizer.field_names)
        self.field_sizes = tokenizer.field_sizes
        self.total_field_size = sum(self.field_sizes)

        self.model_dim = model_dim
        self.total_embedding_dim = sum(embedding_dim[field_name] for field_name in tokenizer.field_names)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=field_size, embedding_dim=embedding_dim[field_name], padding_idx=pad_token_id
                )
                for field_name, field_size, pad_token_id in zip(
                    tokenizer.field_names, tokenizer.field_sizes, tokenizer.pad_token_ids
                )
            ]
        )
        self.encoder = nn.Linear(self.total_embedding_dim, model_dim)
        self.decoder = nn.Linear(model_dim, self.total_field_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            input_ids: (batch_size, seq_len, num_features)
        Returns:
            fused: (batch_size, seq_len, model_dim)
        """
        _, _, num_features = x.shape
        assert num_features == self.num_features, f"num_features must be {self.num_features}"

        # embeddings: (batch_size, seq_len, total_embedding_dim)
        x = torch.concat([embedding(x[:, :, i]) for i, embedding in enumerate(self.embeddings)], dim=2)
        # fused: (batch_size, seq_len, model_dim)
        x = self.encoder(x)
        return x

    def decode(self, fused: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            fused: (batch_size, seq_len, model_dim)
        Returns:
            logits: List[torch.Tensor] of length num_features,
            each of shape (batch_size, seq_len, vocab_size of the feature)
        """
        # embeddings: (batch_size, seq_len, total_field_size)
        embeddings = self.decoder(fused)
        return torch.split(embeddings, self.field_sizes, dim=2)


class MelodyGLM(nn.Module):
    def __init__(
        self,
        # Tokenizer config
        granularity: int,
        max_bar: int,
        pitch_range: Tuple[int, int],
        # Model hyperparameters
        num_layers: int,
        num_heads: int,
        model_dim: int,
        feedforward_dim: int,
        embedding_dim: Dict[str, int],
        dropout: float,
    ) -> None:
        super().__init__()

        # tokenizer
        self.tokenizer = MIDITokenizer(granularity=granularity, max_bar=max_bar, pitch_range=pitch_range)

        # components
        self.fuser = CompoundTokenFuser(self.tokenizer, embedding_dim, model_dim)

        self.num_heads = num_heads
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # special token tensors
        self.eos_token_tensor = torch.tensor(self.tokenizer.eos_token_ids, dtype=torch.long)
        self.sep_token_tensor = torch.tensor(self.tokenizer.sep_token_ids, dtype=torch.long)

        # for attention mask buffering
        self.default_seq_len = 512
        self.prefix_source_length = self.default_seq_len

        # for determining bar length
        self.bar_field_index = self.tokenizer.field_indices["bar"]
        self.bar_vocab_size = self.tokenizer.vocab_sizes[self.bar_field_index]

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str, device: Optional[torch.device] = None) -> "MelodyGLM":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = MelodyGLM(**checkpoint["hyperparameters"])
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def create_causal_attention_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def get_causal_attention_mask(self, length: int, device: torch.device) -> torch.Tensor:
        if not hasattr(self, "causal_attention_mask") or self.causal_attention_mask.shape[0] < length:
            self.causal_attention_mask = self.create_causal_attention_mask(max(length, self.default_seq_len), device)
        return self.causal_attention_mask[:length, :length]

    def create_prefix_attention_mask(self, source_length: int, length: int, device: torch.device) -> torch.Tensor:
        # bidirectional attention mask for prefix sequence
        left_prefix_part = torch.zeros((length, source_length), dtype=torch.bool, device=device)

        target_length = length - source_length
        top_right_target_part = torch.ones((source_length, target_length), dtype=torch.bool, device=device)

        # causal attention mask for infilling sequence
        bottom_right_target_part = torch.triu(
            torch.ones((target_length, target_length), dtype=torch.bool, device=device), diagonal=1
        )

        right_target_part = torch.cat([top_right_target_part, bottom_right_target_part], dim=0)
        return torch.cat([left_prefix_part, right_target_part], dim=1)

    def get_prefix_attention_mask(self, source_length: int, length: int, device: torch.device) -> torch.Tensor:
        if (
            not hasattr(self, "prefix_attention_mask")
            or self.prefix_source_length < source_length
            or self.prefix_attention_mask.shape[0] < length
        ):
            new_source_length = max(source_length, self.prefix_source_length)
            new_length = max(length, 3 * self.default_seq_len)
            self.prefix_attention_mask = self.create_prefix_attention_mask(
                source_length=new_source_length, length=new_length, device=device
            )
            self.prefix_source_length = new_source_length
        start = self.prefix_source_length - source_length
        end = start + length
        return self.prefix_attention_mask[start:end, start:end]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_kind: AttentionKind = "causal",
        source_length: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Args:
            input_ids: (batch_size, seq_len, num_features)
            attention_kind: "causal" or "prefix"
            source_length: length of prefix sequence
        Returns:
            list of num_features * (batch_size, seq_len, vocab_size of the feature)
        """
        batch_size, seq_len, _ = input_ids.shape
        device = input_ids.device

        if attention_kind == "causal":
            attention_mask = self.get_causal_attention_mask(seq_len, device)
        elif attention_kind == "prefix":
            assert source_length is not None, "source_length must be provided for prefix attention"
            attention_mask = self.get_prefix_attention_mask(source_length, seq_len, device)

        x = self.fuser(input_ids)
        x = self.transformer_encoder(x, mask=attention_mask)
        decoded = self.fuser.decode(x)
        return decoded

    def predict_for_completion(
        self,
        input_ids: torch.Tensor,
        max_length: int = 1024,
        max_bar_length: Optional[int] = None,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size, _, _ = input_ids.shape
        device = input_ids.device
        assert batch_size == 1, "Only support batch size of 1 for prediction for now"

        if self.eos_token_tensor.device != device:
            self.eos_token_tensor = self.eos_token_tensor.to(device)

        # Inference on a single sequence
        while input_ids.shape[1] < max_length:
            logits = self(input_ids, attention_kind="causal")
            sampled_tokens = []
            for logit in logits:
                # Decode according to the sampling strategy
                sampled_token = self.top_k_top_p_sample(
                    logit[:, -1, :], top_k=top_k, top_p=top_p, temperature=temperature
                )[0]
                sampled_tokens.append(sampled_token)
            sampled_tokens = torch.cat(sampled_tokens, dim=-1)

            # token = self.tokenizer.convert_id_to_token(sampled_tokens.cpu().numpy())
            # print(token)

            # until <EOS> token is generated or the predicted bar length is reached
            if torch.all(sampled_tokens == self.eos_token_tensor):
                break
            if (
                max_bar_length is not None
                and max_bar_length <= sampled_tokens[self.bar_field_index] < self.bar_vocab_size
            ):
                break

            # Append the sampled token to the input
            input_ids = torch.cat([input_ids, sampled_tokens.unsqueeze(0).unsqueeze(0)], dim=1)

        return input_ids

    def predict_for_inpainting(
        self,
        input_ids: torch.Tensor,
        source_length: int,
        mask_token_position: int,
        future_start_bar: Optional[int],
        max_length: int = 1024,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size, _, _ = input_ids.shape
        device = input_ids.device
        assert batch_size == 1, "Only support batch size of 1 for prediction for now"

        if self.sep_token_tensor.device != device:
            self.sep_token_tensor = self.sep_token_tensor.to(device)

        while input_ids.shape[1] < max_length:
            seq_len = input_ids.shape[1]
            logits = self(input_ids, attention_kind="prefix", source_length=source_length)

            sampled_tokens = []
            for logit in logits:
                # Decode according to the sampling strategy
                sampled_token = self.top_k_top_p_sample(
                    logit[:, -1, :], top_k=top_k, top_p=top_p, temperature=temperature
                )[0]
                sampled_tokens.append(sampled_token)
            sampled_tokens = torch.cat(sampled_tokens, dim=-1)

            # token = self.tokenizer.convert_id_to_token(sampled_tokens.cpu().numpy())
            # print(token)

            # until <SEP> token is predicted
            if torch.all(sampled_tokens == self.sep_token_tensor):
                break
            # or until the first bar of future part is reached
            if (
                future_start_bar is not None
                and future_start_bar <= sampled_tokens[self.bar_field_index] <= self.bar_vocab_size
            ):
                break

            # Append the sampled token to the input
            input_ids = torch.cat([input_ids, sampled_tokens.unsqueeze(0).unsqueeze(0)], dim=1)

        # Rearrange the sequence to the original order
        input_ids = torch.cat(
            [
                input_ids[:, :mask_token_position, :],
                input_ids[:, source_length + 1 :, :],
                input_ids[:, mask_token_position + 1 : source_length + 1, :],
            ],
            dim=1,
        )
        return input_ids

    def complete_melody(
        self,
        prompt_midi_file: Optional[Union[str, BinaryIO]] = None,
        max_length: int = 512,
        max_bar_length: Optional[int] = None,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> MidiFile:
        """
        Complete a melody from scratch or from a prompt MIDI file.
        Args:
            prompt_midi_file: The path to the prompt MIDI file.
            max_length: The maximum length of the generated sequence.
            max_bar_length: The maximum bar length of the generated sequence, optional.
            top_k: The number of highest probability vocabulary tokens to keep for top-k sampling, optional.
            top_p: The cumulative probability of highest probability vocabulary tokens to keep for top-p sampling, optional.
            temperature: The temperature used to scale the logits, optional.
        Returns:
            The generated MIDI file.
        """
        if prompt_midi_file is not None:
            midi_file = MidiFile(prompt_midi_file)
            token_ids, _ = self.tokenizer.encode(midi_file)
            token_ids = np.concatenate([[self.tokenizer.bos_token_ids], token_ids], axis=0)
            input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            # Generate from scratch, starting with a <BOS> token
            input_ids = torch.tensor(self.tokenizer.bos_token_ids, dtype=torch.long, device=self.device)
            input_ids = input_ids.unsqueeze(0).unsqueeze(0)

        # Predict for melody completion
        self.eval()
        with torch.no_grad():
            output_ids = self.predict_for_completion(
                input_ids,
                max_length=max_length,
                max_bar_length=max_bar_length,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )

        midi_file = self.tokenizer.decode(output_ids[0].cpu().numpy())
        return midi_file

    def inpaint_melody(
        self,
        prompt_midi_file: Union[str, BinaryIO],
        bar_range: Optional[Tuple[int, int]] = None,
        note_range: Optional[Tuple[int, int]] = None,
        max_length: int = 512,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> MidiFile:
        """
        Inpaint a melody from a prompt MIDI file with missing bars or notes.
        Args:
            prompt_midi_file: The path to the prompt MIDI file.
            bar_range: The range of bars to inpaint. (Either bar_range or note_range should be provided.)
            note_range: The range of notes to inpaint. (Either bar_range or note_range should be provided.)
            max_length: The maximum length of the generated sequence.
            top_k: The number of highest probability vocabulary tokens to keep for top-k sampling, optional.
            top_p: The cumulative probability of highest probability vocabulary tokens to keep for top-p sampling, optional.
            temperature: The temperature used to scale the logits, optional.
        Returns:
            The generated MIDI file.
        """
        if (bar_range is None) and (note_range is None) or (bar_range is not None) and (note_range is not None):
            raise ValueError("Either bar_range or note_range should be provided, but not both")

        midi_file = MidiFile(prompt_midi_file)
        assert len(midi_file.instruments) == 1, "Only support single track midi file for now"
        notes = midi_file.instruments[0].notes

        # Crop midi file according to the given range
        if bar_range is not None:
            note_bar = [note.start // self.tokenizer.ticks_per_bar for note in notes]
            assert 0 <= bar_range[0] < bar_range[1] <= max(note_bar) + 1, "Invalid bar range"
            start_note = next(index for index, bar in enumerate(note_bar) if bar >= bar_range[0])
            end_note = next((index for index, bar in enumerate(note_bar) if bar >= bar_range[1]), len(notes))
        elif note_range is not None:
            assert 0 <= note_range[0] < note_range[1] <= len(notes), "Invalid note range"
            start_note, end_note = note_range

        midi_file.instruments[0].notes = notes[:start_note] + notes[end_note:]
        token_ids, note_map = self.tokenizer.encode(midi_file)

        # Split index is the index of the first token of the middle part
        split_index = note_map["start"][start_note] if start_note < len(note_map) else len(token_ids)
        source_length = len(token_ids) + 1
        mask_token_position = split_index
        future_start_bar = token_ids[split_index, self.bar_field_index] if start_note < len(note_map) else None
        # Insert a [longMASK] token
        token_ids = np.concatenate(
            [
                token_ids[:split_index],
                [self.tokenizer.long_mask_token_ids],
                token_ids[split_index:],
                [self.tokenizer.sep_token_ids],
            ],
            axis=0,
        )
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        # Predict for melody completion
        self.eval()
        with torch.no_grad():
            output_ids = self.predict_for_inpainting(
                input_ids,
                source_length=source_length,
                mask_token_position=mask_token_position,
                future_start_bar=future_start_bar,
                max_length=max_length,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )

        midi_file = self.tokenizer.decode(output_ids[0].cpu().numpy())
        return midi_file

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def top_k_top_p_sample(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """Sample from a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        """
        assert temperature > 0, "t has to be strictly positive"
        if temperature != 1.0:
            logits = logits / temperature
        logits = self.top_k_top_p_filtering(
            logits, top_k=top_k, top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep
        )
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
