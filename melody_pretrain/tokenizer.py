import json
from math import floor, log2
from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
from miditoolkit import Instrument, MidiFile, Note

Field = int
SpecialTokenField = str
AnyField = Union[Field, SpecialTokenField]


class MIDICompoundToken(NamedTuple):
    """NamedTuple for MIDI compound token.
    One compound token represents one note. Each field is a feature of the note."""

    bar: AnyField
    position: AnyField
    duration: AnyField
    pitch: AnyField


class MIDITokenizer:
    def __init__(self, granularity=64, max_bar=128, pitch_range: Tuple[int, int] = (0, 128)) -> None:
        self.granularity = granularity
        self.max_bar = max_bar

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

        # vocabularies for each field
        self.field_names = MIDICompoundToken._fields
        self.field_indices = {name: index for index, name in enumerate(self.field_names)}
        self.vocabularies: Dict[str, List[int]] = {}
        self.vocabularies["bar"] = list(range(max_bar))
        self.vocabularies["position"] = self.position_bins
        self.vocabularies["duration"] = self.duration_bins
        self.vocabularies["pitch"] = list(self.pitch_range)
        self.vocab_sizes = [len(self.vocabularies[field_name]) for field_name in self.field_names]
        self.field_sizes = list(self.vocab_sizes)  # will be modified when adding special tokens

        # create encoder and decoder for mapping between token field and id
        self.encoder: Dict[str, Dict[AnyField, int]] = {
            field_name: {field: index for index, field in enumerate(self.vocabularies[field_name])}
            for field_name in self.field_names
        }
        self.decoder: Dict[str, Dict[int, AnyField]] = {
            field_name: {index: field for index, field in enumerate(self.vocabularies[field_name])}
            for field_name in self.field_names
        }

        # add special tokens
        self.bos_token_str = "<BOS>"
        self.eos_token_str = "<EOS>"
        self.pad_token_str = "<PAD>"
        self.sep_token_str = "<SEP>"
        self.cls_token_str = "<CLS>"
        self.mask_token_str = "[MASK]"
        self.bos_token = MIDICompoundToken(*[self.bos_token_str] * len(self.field_names))
        self.eos_token = MIDICompoundToken(*[self.eos_token_str] * len(self.field_names))
        self.pad_token = MIDICompoundToken(*[self.pad_token_str] * len(self.field_names))
        self.sep_token = MIDICompoundToken(*[self.sep_token_str] * len(self.field_names))
        self.cls_token = MIDICompoundToken(*[self.cls_token_str] * len(self.field_names))
        self.mask_token = MIDICompoundToken(*[self.mask_token_str] * len(self.field_names))
        self.special_token_str = [
            self.bos_token_str,
            self.eos_token_str,
            self.pad_token_str,
            self.sep_token_str,
            self.cls_token_str,
            self.mask_token_str,
        ]

        for field_index, field_name in enumerate(self.field_names):
            for i, token_str in enumerate(self.special_token_str):
                token_id = len(self.vocabularies[field_name]) + i
                self.encoder[field_name][token_str] = token_id
                self.decoder[field_name][token_id] = token_str
            self.field_sizes[field_index] += len(self.special_token_str)

        self.bos_token_ids = self.convert_tokens_to_ids([self.bos_token])[0]
        self.eos_token_ids = self.convert_tokens_to_ids([self.eos_token])[0]
        self.pad_token_ids = self.convert_tokens_to_ids([self.pad_token])[0]
        self.sep_token_ids = self.convert_tokens_to_ids([self.sep_token])[0]
        self.cls_token_ids = self.convert_tokens_to_ids([self.cls_token])[0]
        self.mask_token_ids = self.convert_tokens_to_ids([self.mask_token])[0]
        self.special_token_id_matrix = np.array(
            [
                self.bos_token_ids,
                self.eos_token_ids,
                self.pad_token_ids,
                self.sep_token_ids,
                self.cls_token_ids,
                self.mask_token_ids,
            ]
        ).T  # (num_features, num_tokens)

    def tokenize(self, midi: MidiFile) -> List[MIDICompoundToken]:
        assert len(midi.instruments) == 1, "Only support single instrument midi file."

        tokens: List[MIDICompoundToken] = []
        for note in midi.instruments[0].notes:
            bar = (note.start // self.ticks_per_bar) % self.max_bar
            position = self._find_nearest(self.position_bins, note.start % self.ticks_per_bar)
            duration = self._find_nearest(self.duration_bins, note.end - note.start)
            tokens.append(MIDICompoundToken(bar, position, duration, note.pitch))
        return tokens

    def convert_token_to_id(self, token: MIDICompoundToken) -> np.ndarray:
        return self.convert_tokens_to_ids([token])[0]
    
    def convert_id_to_token(self, token: np.ndarray) -> MIDICompoundToken:
        return self.convert_ids_to_tokens(np.expand_dims(token, axis=0))[0]

    def convert_tokens_to_ids(self, tokens: List[MIDICompoundToken]) -> np.ndarray:
        token_ids = np.zeros((len(tokens), len(self.field_names)), dtype=np.int16)
        for index, token in enumerate(tokens):
            for field_index, field_name in enumerate(self.field_names):
                field = token[field_index]
                token_ids[index, field_index] = self.encoder[field_name][field]
        return token_ids

    def convert_ids_to_tokens(self, tokens: np.ndarray) -> List[MIDICompoundToken]:
        assert tokens.ndim == 2, "tokens should be 2D array."
        length, field_count = tokens.shape
        assert field_count == len(self.field_names), "field count should be equal to field names."

        result: List[MIDICompoundToken] = []
        for index in range(length):
            fields = []
            for field_index, field_name in enumerate(self.field_names):
                token = tokens[index, field_index]
                field = self.decoder[field_name].get(token, None)
                fields.append(field)
            result.append(MIDICompoundToken(*fields))
        return result

    def detokenize(self, tokens: List[MIDICompoundToken], velocity=100) -> MidiFile:
        midi = MidiFile()
        notes = []
        for token in tokens:
            if any([field is None or field in self.special_token_str for field in token]):
                continue
            start = token.bar * self.ticks_per_bar + token.position
            end = token.bar * self.ticks_per_bar + token.position + token.duration
            note = Note(velocity=velocity, pitch=token.pitch, start=start, end=end)
            notes.append(note)
        instrument = Instrument(program=0)
        instrument.notes.extend(notes)
        midi.instruments.append(instrument)
        return midi

    def get_bar_spans(self, token_ids: np.ndarray) -> np.ndarray:
        """Get bar spans for each token.
        Args:
            token_ids: (num_tokens, num_features)
        Returns:
            bar_spans: (num_bars, 2) array, each row is [start, end) of bar span.
        """
        num_tokens, _ = token_ids.shape
        bar_field_index = self.field_indices["bar"]
        bar_fields = token_ids[:, bar_field_index]
        bar_start_mask = np.concatenate([[True], bar_fields[:-1] != bar_fields[1:]])
        bar_start_indices = np.extract(bar_start_mask, np.arange(num_tokens, dtype=np.int16))
        bar_end_indices = np.concatenate([bar_start_indices[1:], [num_tokens]], dtype=np.int16)
        bar_spans = np.stack([bar_start_indices, bar_end_indices], axis=1)
        return bar_spans

    def encode(self, midi: MidiFile, return_bar_spans: bool = False) -> np.ndarray:
        """Encode midi file to token ids.
        Args:
            midi: midi file to encode.
        Returns:
            token_ids: (length, field)
        """
        tokens = self.tokenize(midi)
        token_ids = self.convert_tokens_to_ids(tokens)
        if return_bar_spans:
            bar_spans = self.get_bar_spans(token_ids)
            return token_ids, bar_spans
        return token_ids

    def decode(self, token_ids: np.ndarray) -> MidiFile:
        """Decode token ids to midi file.
        Args:
            token_ids: (length, field)
        Returns:
            midi: decoded midi file.
        """
        tokens = self.convert_ids_to_tokens(token_ids)
        return self.detokenize(tokens)

    def _find_nearest(self, bins: List[int], value: int) -> int:
        """Find the nearest bin to the value."""
        return min(bins, key=lambda x: abs(x - value))

    def __str__(self) -> str:
        info_str = f"representation: granularity={self.granularity}"
        token_size_str = ", ".join([f"{field_name}={len(d)}" for field_name, d in self.encoder.items()])
        return info_str + "\n" + token_size_str

    @staticmethod
    def from_config(path: str) -> "MIDITokenizer":
        with open(path) as f:
            config = json.load(f)
        return MIDITokenizer(**config)


if __name__ == "__main__":
    tokenizer = MIDITokenizer()
    print("field_names:", tokenizer.field_names)
    print("field_indices:", tokenizer.field_indices)
    print("field_sizes:", tokenizer.field_sizes)
    print("vocab_sizes:", tokenizer.vocab_sizes)
    print("encoder:", tokenizer.encoder)
    tokens = tokenizer.encode("data/test.mid")
    print(tokens)
    midi = tokenizer.decode(tokens)
    midi.dump("data/test_back.mid")
