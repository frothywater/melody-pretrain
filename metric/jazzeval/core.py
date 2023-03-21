from typing import Optional
from .eval_metrics import (
    compute_piece_chord_progression_irregularity,
    compute_piece_groove_similarity,
    compute_piece_pitch_entropy,
    compute_structure_indicator,
)
from .utils import bar_event, convert_midi_to_events, pitch_events, pos_events


def compute_pitch_entropy(midi_file: str, window_size: int, starting_bar: Optional[int] = None, num_bars: Optional[int] = None):
    events = convert_midi_to_events(midi_file, starting_bar, num_bars)
    return compute_piece_pitch_entropy(events, window_size, bar_ev_id=bar_event, pitch_evs=pitch_events)


def compute_groove_similarity(midi_file: str, starting_bar: Optional[int] = None, num_bars: Optional[int] = None):
    events = convert_midi_to_events(midi_file, starting_bar, num_bars)
    return compute_piece_groove_similarity(events, bar_ev_id=bar_event, pos_evs=pos_events, pitch_evs=pitch_events)
