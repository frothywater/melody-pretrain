import json
from math import floor, log2
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from miditoolkit import Instrument, MidiFile, Note, TempoChange

Field = int
SpecialTokenField = str
AnyField = Union[Field, SpecialTokenField]

note_map_record_dtype = np.dtype([("start", np.int16), ("end", np.int16)])


class MIDITokenizer:
    kind: Optional[str] = None

    class Token(NamedTuple):
        """NamedTuple for MIDI compound token.
        One compound token represents one note. Each field is a feature of the note."""

        pass

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

        self.build_encoder_decoder()

    def define_vocabularies(self) -> None:
        raise NotImplementedError

    def build_encoder_decoder(self) -> None:
        self.encoder: Dict[str, Dict[AnyField, int]] = {
            field_name: {field: index for index, field in enumerate(self.vocabularies[field_name])}
            for field_name in self.field_names
        }
        self.decoder: Dict[str, Dict[int, AnyField]] = {
            field_name: {index: field for index, field in enumerate(self.vocabularies[field_name])}
            for field_name in self.field_names
        }

        # special tokens
        self.bos_token_str = "<BOS>"
        self.eos_token_str = "<EOS>"
        self.pad_token_str = "<PAD>"
        self.sep_token_str = "<SEP>"
        self.cls_token_str = "<CLS>"
        self.mask_token_str = "[MASK]"
        self.long_mask_token_str = "[lMASK]"
        self.bos_token = self.Token(*[self.bos_token_str] * len(self.field_names))
        self.eos_token = self.Token(*[self.eos_token_str] * len(self.field_names))
        self.pad_token = self.Token(*[self.pad_token_str] * len(self.field_names))
        self.sep_token = self.Token(*[self.sep_token_str] * len(self.field_names))
        self.cls_token = self.Token(*[self.cls_token_str] * len(self.field_names))
        self.mask_token = self.Token(*[self.mask_token_str] * len(self.field_names))
        self.long_mask_token = self.Token(*[self.long_mask_token_str] * len(self.field_names))
        self.special_token_str = [
            self.bos_token_str,
            self.eos_token_str,
            self.pad_token_str,
            self.sep_token_str,
            self.cls_token_str,
            self.mask_token_str,
            self.long_mask_token_str,
        ]

        # add special tokens to the encoder and decoder
        for field_index, field_name in enumerate(self.field_names):
            for i, token_str in enumerate(self.special_token_str):
                token_id = len(self.vocabularies[field_name]) + i
                self.encoder[field_name][token_str] = token_id
                self.decoder[field_name][token_id] = token_str
            self.field_sizes[field_index] += len(self.special_token_str)

        # convert special tokens to ids
        self.bos_token_ids = self.convert_token_to_id(self.bos_token)
        self.eos_token_ids = self.convert_token_to_id(self.eos_token)
        self.pad_token_ids = self.convert_token_to_id(self.pad_token)
        self.sep_token_ids = self.convert_token_to_id(self.sep_token)
        self.cls_token_ids = self.convert_token_to_id(self.cls_token)
        self.mask_token_ids = self.convert_token_to_id(self.mask_token)
        self.long_mask_token_ids = self.convert_token_to_id(self.long_mask_token)

    def tokenize(self, midi: MidiFile) -> Tuple[List[Token], np.ndarray]:
        """Returns:
        tokens: list of tokens.
        note_map: (num_note) with columns [start, end] mapping note index to token index."""
        raise NotImplementedError

    def detokenize(self, tokens: List[Token], velocity: int = 100) -> MidiFile:
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: List[Token]) -> np.ndarray:
        token_ids = np.zeros((len(tokens), len(self.field_names)), dtype=np.int16)
        for index, token in enumerate(tokens):
            for field_index, field_name in enumerate(self.field_names):
                field = token[field_index]
                token_ids[index, field_index] = self.encoder[field_name][field]
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

    def encode(self, midi: MidiFile) -> Tuple[np.ndarray, np.ndarray]:
        """Encode midi file to token ids.
        Args:
            midi: midi file to encode.
        Returns:
            token_ids: (length, field).
            note_map: (num_note) with columns [start, end] mapping note index to token index.
        """
        tokens, note_map = self.tokenize(midi)
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids, note_map

    def decode(self, token_ids: np.ndarray) -> MidiFile:
        """Decode token ids to midi file.
        Args:
            token_ids: (length, field)
        Returns:
            midi: decoded midi file.
        """
        tokens = self.convert_ids_to_tokens(token_ids)
        return self.detokenize(tokens)

    def pitch_shift_augument_(self, token_ids: np.ndarray, shift_range: int = 6) -> None:
        """Pitch shift augumentation. This method will modify the token_ids in place.
        Args:
            token_ids: (num_tokens, num_fields)
            shift_range: pitch shift range in semitone. The direction may be upward or downward.
        """
        raise NotImplementedError

    def _find_nearest(self, bins: List[int], value: int) -> int:
        """Find the nearest bin to the value."""
        return min(bins, key=lambda x: abs(x - value))

    def __str__(self) -> str:
        info_str = f"representation: {self.kind}, granularity={self.granularity}"
        token_size_str = ", ".join([f"{field_name}={len(d)}" for field_name, d in self.encoder.items()])
        return info_str + "\n" + token_size_str

    def save_config(self, path: str):
        if self.kind is None:
            raise NotImplementedError
        config = {
            "kind": self.kind,
            "granularity": self.granularity,
            "max_bar": self.max_bar,
            "pitch_range": [self.pitch_range.start, self.pitch_range.stop],
        }
        with open(path, "w") as f:
            json.dump(config, f)

    @staticmethod
    def from_kwargs(kind: str, **kwargs) -> "MIDITokenizer":
        if kind == "octuple":
            return OctupleTokenizer(**kwargs)
        else:
            raise ValueError(f"Unknown kind: {kind}")

    @staticmethod
    def from_config(path: str) -> "MIDITokenizer":
        with open(path) as f:
            config = json.load(f)
        kind = config.pop("kind")
        return MIDITokenizer.from_kwargs(kind, **config)


class OctupleTokenizer(MIDITokenizer):
    kind = "octuple"

    class Token(NamedTuple):
        bar: AnyField
        position: AnyField
        duration: AnyField
        pitch: AnyField
        tempo: AnyField

        def __str__(self) -> str:
            return f"[bar:{self.bar:>7}, pos:{self.position:>7}, dur:{self.duration:>7}, pit:{self.pitch:>7}, tmp:{self.tempo:>7}]"

    def define_vocabularies(self) -> None:
        self.vocabularies["bar"] = list(range(self.max_bar))
        self.vocabularies["position"] = self.position_bins
        self.vocabularies["duration"] = self.duration_bins
        self.vocabularies["pitch"] = list(self.pitch_range)
        self.vocabularies["tempo"] = self.tempo_bins

    def tokenize(self, midi: MidiFile) -> Tuple[List[Token], np.ndarray]:
        assert len(midi.instruments) == 1, "Only support single instrument midi file."

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

        current_tempo_index = 0
        tokens: List[self.Token] = []
        for note in midi.instruments[0].notes:
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

        indices = np.arange(len(tokens))
        note_map = np.zeros(len(tokens), dtype=note_map_record_dtype)
        note_map["start"] = indices + 1  # +1 for the <BOS> token
        note_map["end"] = indices + 2
        tokens = [self.bos_token] + tokens + [self.eos_token]
        return tokens, note_map

    def detokenize(self, tokens: List[Token], velocity: int = 100) -> MidiFile:
        midi = MidiFile()
        notes = []
        current_tempo = self.default_tempo
        midi.tempo_changes = [TempoChange(tempo=current_tempo, time=0)]
        for token in tokens:
            if any([field is None or field in self.special_token_str for field in token]):
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

    def pitch_shift_augument_(self, token_ids: np.ndarray, shift_range: int = 6) -> None:
        pitch_shift = np.random.randint(-shift_range, shift_range + 1)
        pitch_field_index = self.field_indices["pitch"]

        # Only adjust non special tokens
        non_special_token_mask = (token_ids[:, pitch_field_index] != self.bos_token_ids[pitch_field_index]) & (
            token_ids[:, pitch_field_index] != self.eos_token_ids[pitch_field_index]
        )
        token_ids = token_ids[non_special_token_mask]

        token_ids[:, pitch_field_index] += pitch_shift
        # Adjust the positions that are out of range
        too_low_mask = token_ids[:, pitch_field_index] < 0
        too_high_mask = token_ids[:, pitch_field_index] >= self.pitch_range.stop
        token_ids[too_low_mask, pitch_field_index] += 12
        token_ids[too_high_mask, pitch_field_index] -= 12


if __name__ == "__main__":
    # testing
    tokenizer = OctupleTokenizer()
    print("field_names:", tokenizer.field_names)
    print("field_indices:", tokenizer.field_indices)
    print("field_sizes:", tokenizer.field_sizes)
    print("vocab_sizes:", tokenizer.vocab_sizes)
    print("encoder:", tokenizer.encoder)
    midi = MidiFile("test.mid")
    token_ids = tokenizer.encode(midi)

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    for token in tokens:
        print(token)

    new_token_ids = tokenizer.encode(tokenizer.decode(token_ids))
    if np.all(token_ids == new_token_ids):
        print("Encode and decode are consistent.")
    else:
        print("Encode and decode are inconsistent!")
