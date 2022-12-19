import json
import math
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from glob import glob
from multiprocessing import Pool

import numpy as np
from miditoolkit import Instrument, Marker, MidiFile, Note
from tqdm import tqdm

ticks_per_bar = 1920


def get_frequency_dict(items: list):
    """Get frequency dict from a list of items.
    Return a dict of item -> frequency."""
    freq_dict = {}
    for item in items:
        if item in freq_dict:
            freq_dict[item] += 1
        else:
            freq_dict[item] = 1
    return freq_dict


def get_ngram_prob_dict(freq_dict: dict):
    """Get ngram probability dict from a frequency dict.
    Return a dict of ngram -> probability."""
    total_freq_dict = defaultdict(int)
    for ngram, freq in freq_dict.items():
        total_freq_dict[len(ngram)] += freq

    ngram_prob_dict = deepcopy(freq_dict)
    for ngram, freq in ngram_prob_dict.items():
        ngram_prob_dict[ngram] = freq / total_freq_dict[len(ngram)]
    return ngram_prob_dict


def get_ngram_score_dict(ngram_prob_dict: dict, kind: str, method: str = "tscore", find_weakest: bool = True):
    """Get ngram score dict from a probability dict.
    Return a dict of ngram -> score.

    Args:
        kind: "pitch" or "rhythm", which kind of ngram to score.
        method: "pmi" or "tscore", which method to use.
            - pmi: Pointwise Mutual Information. Reference: PMI-Masking https://arxiv.org/abs/2010.01825
            - tscore: t-score. Reference: ERNIE-GEN https://arxiv.org/abs/2001.11314
        find_weakest: whether to find the weakest score from all spans of ngram. Reference: PMI-Masking.
    """

    count_dict = get_ngram_count_dict(ngram_prob_dict)

    # cache score for regularized ngrams
    cache = {}

    def regularize(ngram: tuple):
        if kind == "pitch":
            # relative to the first note
            ngram = tuple((ngram[i] - ngram[0]) % 12 for i in range(0, len(ngram)))
        elif kind == "rhythm":
            # relative to the first bar line
            bar_onset = ngram[0] // ticks_per_bar * ticks_per_bar
            ngram = tuple((ngram[i] - bar_onset) for i in range(0, len(ngram)))
        else:
            raise NotImplementedError(kind)
        return ngram

    def get_score(ngram: tuple):
        ngram = regularize(ngram)
        if ngram in cache:
            return cache[ngram]

        # calculate i.i.d. product
        iid_product = 1
        for k in range(len(ngram) - 1):
            if kind == "pitch":
                # unit: unordered interval
                interval = (0, (ngram[k + 1] - ngram[k]) % 12)
                iid_product *= ngram_prob_dict[interval]
            elif kind == "rhythm":
                # unit: onset & inter-onset-interval (IOI)
                bar_onset = ngram[k] // ticks_per_bar * ticks_per_bar
                unit = tuple(onset - bar_onset for onset in ngram[k : k + 2])
                iid_product *= ngram_prob_dict[unit]
            else:
                raise NotImplementedError(kind)

        # calculate score
        product = ngram_prob_dict[ngram]
        if method == "pmi":
            score = math.log2(product / iid_product)
        elif method == "tscore":
            sigma_sqr = product * (1 - product)
            count = count_dict[len(ngram)]
            score = (product - iid_product) / math.sqrt(sigma_sqr * count)
        else:
            raise NotImplementedError(method)

        cache[ngram] = score
        return score

    def get_score_from_all_spans(ngram: tuple):
        """Get score from all spans of ngram, and return the weakest score.
        Ignore span with length 2, since the smallest unit to consider is an interval.
        """
        scores = [
            get_score(ngram[start : start + n]) for n in range(3, len(ngram) + 1) for start in range(len(ngram) - n + 1)
        ]
        return min(scores)

    score_dict = {}
    for ngram in ngram_prob_dict:
        if len(ngram) <= 2:
            continue
        score_dict[ngram] = get_score_from_all_spans(ngram) if find_weakest else get_score(ngram)
    return score_dict


def extract_pitch_class_ngrams_job(midi_file: str, ngram_range: range):
    """Extract pitch class ngrams from a midi file.
    Return a list of (ngram tuple, note index)."""
    midi = MidiFile(midi_file)
    assert len(midi.instruments) == 1, "Only support single-track midi files."
    pitch_classes = [note.pitch % 12 for note in midi.instruments[0].notes]

    ngram_list = []
    for n in ngram_range:
        for i in range(len(pitch_classes) - n + 1):
            # pitch class
            ngram = [(pc - pitch_classes[i]) % 12 for pc in pitch_classes[i : i + n]]
            ngram_list.append((tuple(ngram), i))
    return ngram_list


def extract_onset_ngrams_job(midi_file: str, ngram_range: range):
    """Extract onset ngrams from a midi file.
    Return a list of (ngram tuple, note index)."""
    midi = MidiFile(midi_file)
    assert len(midi.instruments) == 1, "Only support single-track midi files."
    midi = quantize_midi(midi)
    onsets = [note.start for note in midi.instruments[0].notes]

    ngram_list = []
    for n in ngram_range:
        for i in range(len(onsets) - n + 1):
            # relative to the first bar line
            start = onsets[i] // ticks_per_bar * ticks_per_bar
            ngram = [(onset - start) for onset in onsets[i : i + n]]

            # ignore if there is a note longer than half note
            differences = [ngram[i + 1] - ngram[i] for i in range(len(ngram) - 1)]
            if len(ngram) == 1 or all(0 < d <= ticks_per_bar // 2 for d in differences):
                ngram_list.append((tuple(ngram), i))
    return ngram_list


def extract_ngram_freq_dict(midi_file: str, ngram_range: range, kind: str):
    if kind == "pitch":
        ngram_list = extract_pitch_class_ngrams_job(midi_file, ngram_range)
    elif kind == "rhythm":
        ngram_list = extract_onset_ngrams_job(midi_file, ngram_range)
    else:
        raise NotImplementedError(kind)
    return get_frequency_dict([ngram for ngram, _ in ngram_list])


def extract_ngrams(midi_dir: str, ngram_range: range, kind: str):
    midi_files = glob(midi_dir + "/**/*.mid")
    print(f"extracting {kind} ngrams from {len(midi_files)} midi files...")
    with Pool() as pool:
        futures = [
            pool.apply_async(extract_ngram_freq_dict, args=(midi_file, ngram_range, kind)) for midi_file in midi_files
        ]
        freq_dicts = [future.get() for future in tqdm(futures)]

    print("calculating ngram frequency dictionaries...")
    freq_dict = defaultdict(int)
    for d in freq_dicts:
        for k, v in d.items():
            freq_dict[k] += v
    print("total ngrams:", len(freq_dict))

    print("calculating probabilities...")
    ngram_prob_dict = get_ngram_prob_dict(freq_dict)
    ngram_count_dict = get_ngram_count_dict(ngram_prob_dict)
    for n in ngram_count_dict:
        print(f"{n}-gram: {ngram_count_dict[n]}")

    print("calculating score...")
    ngram_dict = get_ngram_score_dict(ngram_prob_dict, kind=kind)

    k = int(0.05 * len(ngram_dict))
    top_ngram_dict = get_top_k_ngrams(ngram_dict, k)
    ngram_str_dict = stringify_dict(top_ngram_dict, kind=kind)

    return ngram_dict, ngram_str_dict


# Main function 1: Extract ngrams
def extract(midi_dir: str, data_dir: str, ngram_range: range):
    """Extract ngrams from midi files."""
    os.makedirs(data_dir, exist_ok=True)
    for kind in ["pitch", "rhythm"]:
        ngram_dict, ngram_str_dict = extract_ngrams(midi_dir, ngram_range, kind=kind)
        pkl_path = os.path.join(data_dir, f"ngram_{kind}.pkl")
        print(f"saving to {pkl_path}...")
        with open(pkl_path, "wb") as f:
            pickle.dump(ngram_dict, f)
        json_path = os.path.join(data_dir, f"ngram_{kind}.json")
        with open(json_path, "w") as f:
            json.dump(ngram_str_dict, f, indent=2)


# Main function 2: Prepare ngram lexicon
def prepare_lexicon(data_dir: str, top_p: float):
    """Prepare ngram lexicon from ngram dictionaries.
    IDs are sorted by score first, then by length, then by ngram."""

    def prepare_ngram_dict(path: str):
        print(f"preparing lexicon for {path}...")
        with open(path, "rb") as f:
            ngram_dict = pickle.load(f)
        k = int(top_p * len(ngram_dict))
        top_ngram_dict = get_top_k_ngrams(ngram_dict, k)
        # label by index
        for i, ngram in enumerate(top_ngram_dict):
            top_ngram_dict[ngram] = i
        return top_ngram_dict

    pitch_ngram_dict = prepare_ngram_dict(os.path.join(data_dir, "ngram_pitch.pkl"))
    rhythm_ngram_dict = prepare_ngram_dict(os.path.join(data_dir, "ngram_rhythm.pkl"))

    # get ngram length range
    pitch_ngram_lengths = set(len(ngram) for ngram in pitch_ngram_dict)
    rhythm_ngram_lengths = set(len(ngram) for ngram in rhythm_ngram_dict)
    ngram_lengths = sorted(pitch_ngram_lengths | rhythm_ngram_lengths)
    ngram_range = range(ngram_lengths[0], ngram_lengths[-1] + 1)
    print(f"ngram lengths: {ngram_lengths}")

    lexicon = {"pitch": pitch_ngram_dict, "rhythm": rhythm_ngram_dict, "ngram_range": ngram_range}

    dest_path = os.path.join(data_dir, "lexicon.pkl")
    print(f"saving lexicon to {dest_path}...")
    with open(dest_path, "wb") as f:
        pickle.dump(lexicon, f)


# Main function 3: Get ngram labels
def get_ngram_labels(midi_file: str, lexicon_path: str):
    """Get ngrams from a midi file with a given lexicon.
    Returns two numpy arrays for pitch and rhythm data respectively.

    Each row is a ngram with the following format:
    [note index, ngram length, ngram id]

    Sorted ascending by id, then by note index.
    """
    with open(lexicon_path, "rb") as f:
        lexicon = pickle.load(f)

    def get_array(ngrams: list, kind: str) -> np.ndarray:
        rows = []
        for ngram_tuple in ngrams:
            ngram, note_index = ngram_tuple
            if ngram in lexicon[kind]:
                row = (note_index, len(ngram), lexicon[kind][ngram])
                rows.append(row)

        # sort ascending by id, then by note index
        rows.sort(key=lambda x: (x[2], x[0]))

        result = np.zeros((len(rows), 3), dtype=np.int16)
        for i, (note_index, ngram_len, ngram_id) in enumerate(rows):
            result[i, 0] = note_index
            result[i, 1] = ngram_len
            result[i, 2] = ngram_id
        return result

    ngram_range = lexicon["ngram_range"]
    pitch_ngrams = extract_pitch_class_ngrams_job(midi_file, ngram_range)
    rhythm_ngrams = extract_onset_ngrams_job(midi_file, ngram_range)
    pitch_array = get_array(pitch_ngrams, "pitch")
    rhythm_array = get_array(rhythm_ngrams, "rhythm")
    return pitch_array, rhythm_array


# Visualization


def render_midi(data_dir: str, dest_dir: str):
    """Render midi files from ngram data for visualization."""

    def render_pitch_ngrams(grouped_ngram_dict: dict, note_length: int = ticks_per_bar // 8):

        print(f"rendering pitch ngrams {dest_dir}...")
        os.makedirs(dest_dir, exist_ok=True)
        for n in grouped_ngram_dict:
            print(f"rendering {n}-grams...")
            group_notes = (n + 3) // 4 * 4
            group_length = group_notes * note_length
            midi = MidiFile()
            instrument = Instrument(program=0)
            for i, (ngram, score) in enumerate(grouped_ngram_dict[n].items()):
                start = i * group_length
                marker = Marker(text=f"{n}-{i}:{score:.2f}", time=start)
                midi.markers.append(marker)
                ngram = transpose_pitches(ngram)
                ngram = smoothen_pitches(ngram)
                for j, pitch in enumerate(ngram):
                    note = Note(
                        pitch=pitch + 60, start=start + j * note_length, end=start + (j + 1) * note_length, velocity=80
                    )
                    instrument.notes.append(note)
            midi.instruments.append(instrument)
            midi.dump(os.path.join(dest_dir, f"pitch-{n}-gram.mid"))

    def render_rhythm_ngrams(grouped_ngram_dict: dict, note_pitch: int = 65):
        print(f"rendering rhythm ngrams {dest_dir}...")
        os.makedirs(dest_dir, exist_ok=True)
        for n in grouped_ngram_dict:
            print(f"rendering {n}-grams...")
            midi = MidiFile()
            instrument = Instrument(program=0)
            pos = 0
            for i, (ngram, score) in enumerate(grouped_ngram_dict[n].items()):
                marker = Marker(text=f"{n}-{i}:{score:.2f}", time=pos)
                midi.markers.append(marker)
                durations = [ngram[i + 1] - ngram[i] for i in range(len(ngram) - 1)]
                last_duration = durations[-1]
                end = (pos + ngram[-1] + last_duration + ticks_per_bar - 1) // ticks_per_bar * ticks_per_bar
                for j, onset in enumerate(ngram):
                    duration = ngram[j + 1] - onset if j < len(ngram) - 1 else last_duration
                    note = Note(pitch=note_pitch, start=pos + onset, end=pos + onset + duration, velocity=80)
                    instrument.notes.append(note)
                pos = end
            midi.instruments.append(instrument)
            midi.dump(os.path.join(dest_dir, f"rhythm-{n}-gram.mid"))

    for kind in ["pitch", "rhythm"]:
        dict_path = os.path.join(data_dir, f"ngram_{kind}.pkl")
        with open(dict_path, "rb") as f:
            ngram_dict = pickle.load(f)

        k = int(0.05 * len(ngram_dict))
        top_ngram_dict = get_top_k_ngrams(ngram_dict, k)

        count_dict = get_ngram_count_dict(ngram_dict)
        grouped_ngram_dict = group_ngram_dict(top_ngram_dict)
        print(f"total ngrams: {len(ngram_dict)}, top {k} ngrams")
        for n in count_dict:
            selected = len(grouped_ngram_dict[n])
            total = count_dict[n]
            print(f"{n}-gram: {selected}/{total} ({selected/total:.2%})")

        if kind == "pitch":
            render_pitch_ngrams(grouped_ngram_dict)
        elif kind == "rhythm":
            render_rhythm_ngrams(grouped_ngram_dict)


# Helper functions


def get_ngram_count_dict(ngram_prob_dict: dict):
    """Get ngram count dict from a probability dict.
    Return a dict of n -> count."""
    count_dict = defaultdict(int)
    for ngram in ngram_prob_dict:
        count_dict[len(ngram)] += 1
    return count_dict


def get_top_k_ngrams(ngram_dict: dict, k: int):
    """Get top k ngrams with highest score. Sort by score first, then by length, then by ngram."""
    items = sorted(ngram_dict.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    return dict(items[:k])


def group_ngram_dict(ngram_dict: dict):
    """Group ngrams by length, and return a dict of length -> ngram -> score."""
    grouped_ngram_dict = defaultdict(dict)
    for ngram, score in ngram_dict.items():
        grouped_ngram_dict[len(ngram)][ngram] = score
    return grouped_ngram_dict


def transpose_pitches(pitches: tuple):
    """Transpose pitches to a key where there are least accidentals.
    This is utility function for visualize n-grams."""
    accidental_notes = [1, 3, 6, 8, 10]
    results = []
    for key in range(12):
        pc = [(note + key) % 12 for note in pitches]
        accidentals = sum(1 for note in pc if note in accidental_notes)
        results.append((key, accidentals))
    key, _ = min(results, key=lambda x: x[1])
    return tuple((pc + key) % 12 for pc in pitches)


def smoothen_pitches(pitches: tuple):
    """Smooth pitches so that they won't jump around.
    This is utility function for visualize n-grams."""
    result = [pitches[0]]
    for i in range(1, len(pitches)):
        interval = (pitches[i] - pitches[i - 1]) % 12
        upward = result[i - 1] + 12 + interval
        middle = result[i - 1] + interval
        downward = result[i - 1] - 12 + interval
        note = min([upward, middle, downward], key=lambda x: abs(x - result[i - 1]))
        result.append(note)
    return tuple(result)


def stringify_dict(d: dict, kind: str):
    """This is utility function for visualize n-grams."""
    note_names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

    def stringify(ngram: tuple):
        if kind == "pitch":
            return "-".join([note_names[n] for n in transpose_pitches(ngram)])
        elif kind == "rhythm":
            return "-".join([f"{n / (ticks_per_bar / 4):.2f}" for n in ngram])

    return {stringify(ngram): score for ngram, score in d.items()}


def quantize_midi(midi: MidiFile, granularity: int = 16, triplet: bool = True):
    ticks_per_unit = ticks_per_bar // granularity
    ticks_per_triplet_unit = ticks_per_unit // 3 * 4
    double_positions = set(range(0, ticks_per_bar, ticks_per_unit))
    triplet_positions = set(range(0, ticks_per_bar, ticks_per_triplet_unit))
    positions = double_positions | triplet_positions if triplet else double_positions

    notes = midi.instruments[0].notes
    # select highest note for overlapping notes
    notes = sorted(notes, key=lambda x: (x.start, -x.pitch))
    notes = [notes[0]] + [notes[i] for i in range(1, len(notes)) if notes[i].start != notes[i - 1].start]
    # quantize
    for note in notes:
        bar = note.start // ticks_per_bar
        start = note.start % ticks_per_bar
        end = note.end % ticks_per_bar
        start = min(positions, key=lambda x: abs(x - start))
        end = min(positions, key=lambda x: abs(x - end))
        note.start = bar * ticks_per_bar + start
        note.end = bar * ticks_per_bar + end
    return midi
