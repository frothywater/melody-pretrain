import json
from collections import defaultdict
from math import floor, log2
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from miditoolkit import Instrument, MidiFile, Note, TempoChange

AnyField = Union[int, str]
pad_str = "<PAD>"
none_str = "-"
note_map_record_dtype = np.dtype([("start", np.int16), ("end", np.int16)])


class MIDITokenizer:
    kind: Optional[str] = None

    class Token(NamedTuple):
        """NamedTuple for MIDI compound token.
        One compound token represents one note. Each field is a feature of the note."""

        pass

    def __init__(
        self,
        granularity: int = 64,
        max_bar: int = 128,
        pitch_range: Tuple[int, int] = (0, 128),
        add_segment_token: bool = True,
    ) -> None:
        """Initialize a MIDITokenizer instance.
        Args:
            granularity: The number of units per bar. Defaults to 64 (64-th note).
            max_bar: The maximum number of bar token to use. Exceeded ones will be mod by the number. Defaults to 128.
            pitch_range: The range of pitch token to use. Defaults to (0, 128)."""
        self.granularity = granularity
        self.max_bar = max_bar
        self.add_segment_token = add_segment_token

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
        raise NotImplementedError

    def define_special_tokens(self) -> None:
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

    def get_tokens_bar_length(self, tokens: List[Token]) -> int:
        """Get the bar length of the tokens."""
        raise NotImplementedError

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

    def _get_tempo_changes(self, midi: MidiFile) -> List[TempoChange]:
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

    def filter_overlapping_notes(self, midi: MidiFile) -> MidiFile:
        assert len(midi.instruments) == 1
        # sort notes by start time, end time (longer first), and pitch (higher first)
        notes = sorted(midi.instruments[0].notes, key=lambda x: (x.start, -x.end, -x.pitch))
        new_notes = []
        current_end = None
        for note in notes:
            if current_end is None or note.end > current_end:
                # the note is not overlapped
                new_notes.append(note)
                current_end = note.end
        midi.instruments[0].notes = new_notes
        return midi

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
            "add_segment_token": self.add_segment_token,
        }
        with open(path, "w") as f:
            json.dump(config, f)

    @staticmethod
    def from_kwargs(kind: str, **kwargs) -> "MIDITokenizer":
        if kind == "octuple":
            return OctupleTokenizer(**kwargs)
        elif kind == "cp":
            return CPTokenizer(**kwargs)
        elif kind == "remi":
            return RemiTokenizer(**kwargs)
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
            bar = self.bar if self.bar != pad_str else ""
            position = self.position if self.position != pad_str else ""
            duration = self.duration if self.duration != pad_str else ""
            pitch = self.pitch if self.pitch != pad_str else ""
            tempo = self.tempo if self.tempo != pad_str else ""
            return f"[bar:{bar:>7}, pos:{position:>7}, dur:{duration:>7}, pit:{pitch:>7}, tmp:{tempo:>7}]"

    def define_vocabularies(self) -> None:
        self.vocabularies["bar"] = list(range(self.max_bar))
        self.vocabularies["position"] = self.position_bins
        self.vocabularies["duration"] = self.duration_bins
        self.vocabularies["pitch"] = list(self.pitch_range)
        self.vocabularies["tempo"] = self.tempo_bins

    def define_special_tokens(self) -> None:
        super().define_special_tokens()
        self.seg_token_str = "<SEG>"
        self.seg_token = self.Token(*[self.seg_token_str] * len(self.field_names))
        self.special_token_str.append(self.seg_token_str)

    def set_special_token_ids(self) -> None:
        super().set_special_token_ids()
        self.seg_token_ids = self.convert_token_to_id(self.seg_token)

    def tokenize(self, midi: MidiFile) -> Tuple[List[Token], np.ndarray]:
        assert len(midi.instruments) == 1, "Only support single instrument midi file."

        notes = midi.instruments[0].notes
        if self.add_segment_token:
            segment_note_indices = self.get_segment_note_indices(midi)
        else:
            segment_note_indices = []

        current_tempo_index = 0
        tempo_changes = self._get_tempo_changes(midi)
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

        tokens = [self.bos_token] + tokens + [self.eos_token]
        note_map["start"] += 1  # +1 for the <BOS> token
        note_map["end"] += 1
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

    def pitch_shift_augument_(self, token_ids: np.ndarray, shift_range: int = 6) -> None:
        pitch_shift = np.random.randint(-shift_range, shift_range + 1)
        pitch_field_index = self.field_indices["pitch"]

        # Only adjust non special tokens
        non_special_token_mask = token_ids[:, pitch_field_index] < self.pitch_range.stop
        token_ids = token_ids[non_special_token_mask]

        token_ids[:, pitch_field_index] += pitch_shift
        # Adjust the positions that are out of range
        too_low_mask = token_ids[:, pitch_field_index] < 0
        too_high_mask = token_ids[:, pitch_field_index] >= self.pitch_range.stop
        token_ids[too_low_mask, pitch_field_index] += 12
        token_ids[too_high_mask, pitch_field_index] -= 12

    def get_tokens_bar_length(self, tokens: List[Token]) -> int:
        return max(token.bar for token in tokens if isinstance(token.bar, int))


class CPTokenizer(MIDITokenizer):
    kind = "cp"

    class Token(NamedTuple):
        family: AnyField
        metrical: AnyField = none_str
        tempo: AnyField = none_str
        pitch: AnyField = none_str
        duration: AnyField = none_str

        def __str__(self) -> str:
            family = self.family if self.family != pad_str else ""
            metrical = self.metrical if self.metrical != pad_str else ""
            tempo = self.tempo if self.tempo != pad_str else ""
            pitch = self.pitch if self.pitch != pad_str else ""
            duration = self.duration if self.duration != pad_str else ""
            return f"[{family:>8}, pos:{metrical:>7}, tmp:{tempo:>7}, pit:{pitch:>7}, dur:{duration:>7}]"

    def define_vocabularies(self) -> None:
        self.vocabularies["family"] = ["note", "metrical"]
        self.vocabularies["metrical"] = ["bar"] + self.position_bins + [none_str]
        self.vocabularies["tempo"] = self.tempo_bins + [none_str]
        self.vocabularies["pitch"] = list(self.pitch_range) + [none_str]
        self.vocabularies["duration"] = self.duration_bins + [none_str]

    def tokenize(self, midi: MidiFile) -> Tuple[List[Token], np.ndarray]:
        assert len(midi.instruments) == 1, "Only support single instrument midi file."

        current_bar, current_tempo_index = 0, 0
        previous_bar, previous_position = None, None
        previous_token_index = 0
        tempo_changes = self._get_tempo_changes(midi)
        tokens: List[self.Token] = [self.Token(family="metrical", metrical="bar")]
        note_map = np.zeros(len(midi.instruments[0].notes), dtype=note_map_record_dtype)
        for note_index, note in enumerate(midi.instruments[0].notes):
            # change current tempo if current note is after the next tempo change
            if (
                current_tempo_index < len(tempo_changes) - 1
                and note.start >= tempo_changes[current_tempo_index + 1].time
            ):
                current_tempo_index += 1

            bar = (note.start // self.ticks_per_bar) % self.max_bar
            if bar != current_bar:
                tokens += [self.Token(family="metrical", metrical="bar")] * (bar - current_bar)
                current_bar = bar

            position = self._find_nearest(self.position_bins, note.start % self.ticks_per_bar)
            tempo = self._find_nearest(self.tempo_bins, tempo_changes[current_tempo_index].tempo)
            if previous_bar != bar or previous_position != position:
                tokens.append(self.Token(family="metrical", metrical=position, tempo=tempo))

            duration = self._find_nearest(self.duration_bins, note.end - note.start)
            tokens.append(self.Token(family="note", pitch=note.pitch, duration=duration))
            note_map[note_index] = (previous_token_index, len(tokens))

            previous_token_index = len(tokens)
            previous_bar, previous_position = bar, position

        tokens = [self.bos_token] + tokens + [self.eos_token]
        note_map["start"] += 1  # +1 for the <BOS> token
        note_map["end"] += 1
        return tokens, note_map

    def detokenize(self, tokens: List[Token], velocity: int = 100) -> MidiFile:
        def is_valid(field: str) -> bool:
            return field != none_str and field not in self.special_token_str

        midi = MidiFile()
        notes = []
        current_bar, current_position = None, None
        current_tempo = self.default_tempo
        midi.tempo_changes = [TempoChange(tempo=current_tempo, time=0)]
        for token in tokens:
            if not is_valid(token.family):
                continue

            if token.family == "metrical":
                if token.metrical == "bar":
                    current_bar = current_bar + 1 if current_bar is not None else 0
                elif is_valid(token.metrical):
                    current_position = current_bar * self.ticks_per_bar + token.metrical
                    # add tempo change if tempo changes
                    if is_valid(token.tempo) and token.tempo != current_tempo:
                        current_tempo = token.tempo
                        midi.tempo_changes.append(TempoChange(tempo=current_tempo, time=current_position))
            elif token.family == "note" and is_valid(token.pitch) and is_valid(token.duration):
                note = Note(
                    velocity=velocity, pitch=token.pitch, start=current_position, end=current_position + token.duration
                )
                notes.append(note)

        instrument = Instrument(program=0)
        instrument.notes.extend(notes)
        midi.instruments.append(instrument)
        return midi

    def pitch_shift_augument_(self, token_ids: np.ndarray, shift_range: int = 6) -> None:
        pitch_shift = np.random.randint(-shift_range, shift_range + 1)
        pitch_field_index = self.field_indices["pitch"]

        # Only adjust non special tokens
        non_special_token_mask = token_ids[:, pitch_field_index] < self.pitch_range.stop
        token_ids = token_ids[non_special_token_mask]

        token_ids[:, pitch_field_index] += pitch_shift
        # Adjust the positions that are out of range
        too_low_mask = token_ids[:, pitch_field_index] < 0
        too_high_mask = token_ids[:, pitch_field_index] >= self.pitch_range.stop
        token_ids[too_low_mask, pitch_field_index] += 12
        token_ids[too_high_mask, pitch_field_index] -= 12

    def get_tokens_bar_length(self, tokens: List[Token]) -> int:
        return sum(1 for token in tokens if token.family == "metrical" and token.metrical == "bar") - 1


class RemiTokenizer(MIDITokenizer):
    kind = "remi"

    class Token(NamedTuple):
        remi: AnyField

        def __str__(self) -> str:
            return f"{self.remi:<15}"

    class _RemiToken(NamedTuple):
        bar: bool = False
        pitch: Optional[int] = None
        position: Optional[int] = None
        duration: Optional[int] = None
        tempo: Optional[int] = None

        def __str__(self) -> str:
            if self.bar:
                return "bar"
            elif self.pitch is not None:
                return f"pitch_{self.pitch}"
            elif self.position is not None:
                return f"position_{self.position}"
            elif self.duration is not None:
                return f"duration_{self.duration}"
            elif self.tempo is not None:
                return f"tempo_{self.tempo}"
            else:
                raise ValueError("Invalid remi token.")

        def to_token(self) -> "RemiTokenizer.Token":
            return RemiTokenizer.Token(str(self))

        @staticmethod
        def from_token(token: "RemiTokenizer.Token") -> "RemiTokenizer._RemiToken":
            token_str = token.remi
            if token_str == "bar":
                return RemiTokenizer._RemiToken(bar=True)
            value = int(token_str.split("_")[1])
            if token_str.startswith("pitch"):
                return RemiTokenizer._RemiToken(pitch=value)
            elif token_str.startswith("position"):
                return RemiTokenizer._RemiToken(position=value)
            elif token_str.startswith("duration"):
                return RemiTokenizer._RemiToken(duration=value)
            elif token_str.startswith("tempo"):
                return RemiTokenizer._RemiToken(tempo=value)
            else:
                raise ValueError("Invalid remi token.")

    def define_vocabularies(self) -> None:
        remi_tokens = (
            [self._RemiToken(bar=True)]
            + [self._RemiToken(pitch=pitch) for pitch in self.pitch_range]
            + [self._RemiToken(position=position) for position in self.position_bins]
            + [self._RemiToken(duration=duration) for duration in self.duration_bins]
            + [self._RemiToken(tempo=tempo) for tempo in self.tempo_bins]
        )
        self.pitch_token_id_start = 1
        self.pitch_token_id_end = 1 + len(self.pitch_range)
        self.vocabularies["remi"] = [str(remi_token) for remi_token in remi_tokens]

    def tokenize(self, midi: MidiFile) -> Tuple[List[Token], np.ndarray]:
        assert len(midi.instruments) == 1, "Only support single instrument midi file."

        current_bar, current_tempo_index = 0, 0
        previous_bar, previous_position, previous_tempo = None, None, None
        previous_token_index = 0
        tempo_changes = self._get_tempo_changes(midi)
        tokens = [self._RemiToken(bar=True)]
        note_map = np.zeros(len(midi.instruments[0].notes), dtype=note_map_record_dtype)
        for note_index, note in enumerate(midi.instruments[0].notes):
            # change current tempo if current note is after the next tempo change
            if (
                current_tempo_index < len(tempo_changes) - 1
                and note.start >= tempo_changes[current_tempo_index + 1].time
            ):
                current_tempo_index += 1

            bar = (note.start // self.ticks_per_bar) % self.max_bar
            if bar != current_bar:
                tokens += [self._RemiToken(bar=True)] * (bar - current_bar)
                current_bar = bar

            position = self._find_nearest(self.position_bins, note.start % self.ticks_per_bar)
            tempo = self._find_nearest(self.tempo_bins, tempo_changes[current_tempo_index].tempo)
            if previous_bar != bar or previous_position != position:
                tokens.append(self._RemiToken(position=position))
            if previous_tempo != tempo:
                tokens.append(self._RemiToken(tempo=tempo))

            duration = self._find_nearest(self.duration_bins, note.end - note.start)
            tokens.append(self._RemiToken(pitch=note.pitch))
            tokens.append(self._RemiToken(duration=duration))
            note_map[note_index] = (previous_token_index, len(tokens))

            previous_token_index = len(tokens)
            previous_bar, previous_position, previous_tempo = bar, position, tempo

        tokens = [token.to_token() for token in tokens]
        tokens = [self.bos_token] + tokens + [self.eos_token]
        note_map["start"] += 1  # +1 for the <BOS> token
        note_map["end"] += 1
        return tokens, note_map

    def detokenize(self, tokens: List[Token], velocity: int = 100) -> MidiFile:
        midi = MidiFile()
        midi.tempo_changes = [TempoChange(tempo=self.default_tempo, time=0)]

        notes = []
        index_to_skip_to = None
        current_bar, current_position = None, None
        for index, token in enumerate(tokens):
            if index_to_skip_to is not None and index < index_to_skip_to:
                continue
            if token.remi in self.special_token_str:
                continue
            remi_token = self._RemiToken.from_token(token)

            if remi_token.bar:
                current_bar = current_bar + 1 if current_bar is not None else 0
            elif remi_token.position is not None:
                current_position = current_bar * self.ticks_per_bar + remi_token.position
            elif remi_token.tempo is not None:
                midi.tempo_changes.append(TempoChange(tempo=remi_token.tempo, time=current_position))
            elif remi_token.pitch is not None:
                if index + 1 < len(tokens) and self._RemiToken.from_token(tokens[index + 1]).duration is not None:
                    duration = self._RemiToken.from_token(tokens[index + 1]).duration
                    note = Note(
                        velocity=velocity,
                        pitch=remi_token.pitch,
                        start=current_position,
                        end=current_position + duration,
                    )
                    notes.append(note)
                    index_to_skip_to = index + 2

        instrument = Instrument(program=0)
        instrument.notes.extend(notes)
        midi.instruments.append(instrument)
        return midi

    def pitch_shift_augument_(self, token_ids: np.ndarray, shift_range: int = 6) -> None:
        pitch_shift = np.random.randint(-shift_range, shift_range + 1)

        # Only adjust pitch tokens
        token_ids = token_ids[:, self.field_indices["remi"]]
        pitch_token_mask = (token_ids >= self.pitch_token_id_start) & (token_ids < self.pitch_token_id_end)
        token_ids = token_ids[pitch_token_mask]

        token_ids += pitch_shift
        # Adjust the positions that are out of range
        token_ids[token_ids < self.pitch_token_id_start] += 12
        token_ids[token_ids >= self.pitch_token_id_end] -= 12

    def get_tokens_bar_length(self, tokens: List[Token]) -> int:
        return sum(1 for token in tokens if token.remi == "bar") - 1


if __name__ == "__main__":
    # testing
    tokenizer = OctupleTokenizer()
    print("field_names:", tokenizer.field_names)
    print("field_indices:", tokenizer.field_indices)
    print("field_sizes:", tokenizer.field_sizes)
    print("vocab_sizes:", tokenizer.vocab_sizes)
    print("encoder:", tokenizer.encoder)
    midi = MidiFile("experiment/test/test.mid")

    token_ids, note_map = tokenizer.encode(midi)

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    for i, token in enumerate(tokens):
        print(token)

    print(note_map)
    if (
        np.all(note_map["end"][:-1] == note_map["start"][1:])
        and note_map["start"][0] == 1
        and note_map["end"][-1] == len(tokens) - 1
    ):
        print("Note map is continuous.")
    else:
        print("Note map is not continuous!")

    new_token_ids, _ = tokenizer.encode(tokenizer.decode(token_ids))
    if np.all(token_ids == new_token_ids):
        print("Encode and decode are consistent.")
    else:
        print("Encode and decode are inconsistent!")
