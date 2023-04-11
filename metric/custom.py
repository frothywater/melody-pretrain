from collections import defaultdict
from typing import Tuple

from miditoolkit import MidiFile
import numpy as np

bar_ticks = 1920

def get_distinct_ngram_percentage(midi_file: str, n_range: Tuple[int, int], max_bar: int = 32):
    """Compose & Embellish
    Following the popular dist-n metric used in natural language generation to evaluate the diversity of generated content,
    we compute the percentage of distinct n-grams in the pitch sequence.
    We regard 3-5, 6-10, and 11-20 contiguous notes as short, medium, and long excerpts."""
    midi = MidiFile(midi_file)
    assert len(midi.instruments) == 1
    notes = [note for note in midi.instruments[0].notes if note.start < max_bar * bar_ticks]
    
    ngrams = set()
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(notes) - n + 1):
            pitches = [note.pitch for note in notes[i:i + n]]
            ngram = tuple((pitch - pitches[0]) % 12 for pitch in pitches)
            ngrams.add(ngram)
    return len(ngrams) / len(notes)

def get_bar_pair_similarity(midi_file: str, max_bar_interval: int = 40):
    """Museformer"""
    midi = MidiFile(midi_file)
    
    # split notes into bars (start from 1)
    bar_notes = defaultdict(list)
    for note in midi.instruments[0].notes:
        bar = note.start // bar_ticks
        position = note.start - bar * bar_ticks
        duration = note.end - note.start
        bar_notes[bar + 1].append((note.pitch, position, duration))
    
    result = np.ones(max_bar_interval) * np.nan
    if len(bar_notes) < 2:
        return result
    for interval in range(1, min(max_bar_interval + 1, len(bar_notes))):
        similarities = []
        for i in range(len(bar_notes) - interval):
            bar1 = set(bar_notes[i])
            bar2 = set(bar_notes[i + interval])
            intersection = bar1 & bar2
            union = bar1 | bar2
            similarity = len(intersection) / len(union) if len(union) > 0 else 0
            similarities.append(similarity)
        result[interval - 1] = np.mean(similarities)
    return result