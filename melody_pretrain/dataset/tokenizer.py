from math import floor, log2
from typing import List, Dict, NamedTuple
from miditoolkit import MidiFile, Instrument, Note
import numpy as np


class MIDITokenItem(NamedTuple):
    """NamedTuple for MIDI token. One token is one note."""
    bar: int
    position: int
    duration: int
    pitch: int


class MIDITokenizer:
    def __init__(self, granularity=64, max_bar=128, default_velocity=100, pitch_range=range(0, 128)) -> None:
        self.granularity = granularity
        self.max_bar = max_bar
        self.default_velocity = default_velocity

        self.pitch_range = pitch_range
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

        self.field_names = MIDITokenItem._fields
        # padding id=0, field=None

        # {field name: {field: token id}}
        field_to_id_dict: Dict[str, Dict[int, int]] = {}
        field_to_id_dict["bar"] = {bar: i + 1 for i, bar in enumerate(range(max_bar))}
        field_to_id_dict["position"] = {position: i + 1 for i, position in enumerate(self.position_bins)}
        field_to_id_dict["duration"] = {duration: i + 1 for i, duration in enumerate(self.duration_bins)}
        field_to_id_dict["pitch"] = {pitch: i + 1 for i, pitch in enumerate(pitch_range)}
        self.field_to_id_dict = field_to_id_dict

        # {field name: {token id: field}}
        id_to_field_dict: Dict[str, Dict[int, int]] = {}
        for field_name in self.field_names:
            id_to_field_dict[field_name] = {token_id: field for field, token_id in field_to_id_dict[field_name].items()}
            id_to_field_dict[field_name][0] = None
        self.id_to_field_dict = id_to_field_dict

    def midi_to_items(self, midi: MidiFile) -> List[MIDITokenItem]:
        assert len(midi.instruments) == 1, "Only support single instrument midi file."

        result: List[MIDITokenItem] = []
        for note in midi.instruments[0].notes:
            bar = note.start // self.ticks_per_bar
            if bar >= self.max_bar:
                continue
            position = self._find_nearest(self.position_bins, note.start % self.ticks_per_bar)
            duration = self._find_nearest(self.duration_bins, note.end - note.start)
            result.append(MIDITokenItem(bar, position, duration, note.pitch))
        return result

    def items_to_tokens(self, items: List[MIDITokenItem]) -> np.ndarray:
        result = np.zeros((len(items), len(self.field_names)), dtype=np.uint16)
        for index, item in enumerate(items):
            for field_index, field_name in enumerate(self.field_names):
                field = item[field_index]
                result[index, field_index] = self.field_to_id_dict[field_name][field]
        return result

    def tokens_to_items(self, tokens: np.ndarray) -> List[MIDITokenItem]:
        assert tokens.ndim == 2, "tokens should be 2D array."
        length, field_count = tokens.shape
        assert field_count == len(self.field_names), "field count should be equal to field names."

        result: List[MIDITokenItem] = []
        for index in range(length):
            fields = []
            for field_index, field_name in enumerate(self.field_names):
                token = tokens[index, field_index]
                field = self.id_to_field_dict[field_name][token]
                fields.append(field)
            result.append(MIDITokenItem(*fields))
        return result

    def items_to_midi(self, items: List[MIDITokenItem]) -> MidiFile:
        midi = MidiFile()
        notes = []
        for item in items:
            start = item.bar * self.ticks_per_bar + item.position
            end = item.bar * self.ticks_per_bar + item.position + item.duration
            note = Note(velocity=self.default_velocity, pitch=item.pitch, start=start, end=end)
            notes.append(note)
        instrument = Instrument(program=0)
        instrument.notes.extend(notes)
        midi.instruments.append(instrument)
        return midi

    def tokenize(self, midi_path: str) -> np.ndarray:
        """Tokenize a MIDI file.
        Result: tokens, (length, field)"""
        midi = MidiFile(midi_path)
        items = self.midi_to_items(midi)
        return self.items_to_tokens(items)

    def detokenize(self, tokens: np.ndarray) -> MidiFile:
        items = self.tokens_to_items(tokens)
        return self.items_to_midi(items)

    def _find_nearest(self, bins: List[int], value: int) -> int:
        """Find the nearest bin to the value."""
        return min(bins, key=lambda x: abs(x - value))

    def __str__(self) -> str:
        info_str = f"representation: granularity={self.granularity}"
        token_size_str = ", ".join([f"{field_name}={len(d)}" for field_name, d in self.id_to_field_dict])
        return info_str + "\n" + token_size_str


if __name__ == "__main__":
    tokenizer = MIDITokenizer()
    print(tokenizer.field_to_id_dict)
    tokens = tokenizer.tokenize("data/test.mid")
    print(tokens)
    midi = tokenizer.detokenize(tokens)
    midi.dump("data/test_back.mid")
