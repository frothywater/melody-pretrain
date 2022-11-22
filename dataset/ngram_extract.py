import json
import math
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from glob import glob
from multiprocessing import Pool

from miditoolkit import Instrument, Marker, MidiFile, Note
from tqdm import tqdm

ticks_per_bar = 1920


def transpose_pitches(pitches: tuple):
    """Transpose pitches to a key where there are least accidentals."""
    accidental_notes = [1, 3, 6, 8, 10]
    results = []
    for key in range(12):
        pc = [(note + key) % 12 for note in pitches]
        accidentals = sum(1 for note in pc if note in accidental_notes)
        results.append((key, accidentals))
    key, _ = min(results, key=lambda x: x[1])
    return tuple((pc + key) % 12 for pc in pitches)


def smoothen_pitches(pitches: tuple):
    result = [pitches[0]]
    for i in range(1, len(pitches)):
        interval = (pitches[i] - pitches[i - 1]) % 12
        upward = result[i - 1] + 12 + interval
        middle = result[i - 1] + interval
        downward = result[i - 1] - 12 + interval
        note = min([upward, middle, downward], key=lambda x: abs(x - result[i - 1]))
        result.append(note)
    return tuple(result)


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


def get_frequency_dict(items: list):
    freq_dict = {}
    for item in items:
        if item in freq_dict:
            freq_dict[item] += 1
        else:
            freq_dict[item] = 1
    return freq_dict


def get_ngram_prob_dict(freq_dict: dict):
    total_freq_dict = defaultdict(int)
    for ngram, freq in freq_dict.items():
        total_freq_dict[len(ngram)] += freq

    ngram_prob_dict = deepcopy(freq_dict)
    for ngram, freq in ngram_prob_dict.items():
        ngram_prob_dict[ngram] = freq / total_freq_dict[len(ngram)]
    return ngram_prob_dict


def get_ngram_count_dict(ngram_prob_dict: dict):
    count_dict = defaultdict(int)
    for ngram in ngram_prob_dict:
        count_dict[len(ngram)] += 1
    return count_dict


def get_ngram_score_dict(ngram_prob_dict: dict, kind: str, method: str = "tscore", find_weakest: bool = True):
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
        Ignore span with length 2, since the smallest unit to consider is an interval."""
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


def stringify_dict(d: dict, kind: str):
    note_names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

    def stringify(ngram: tuple):
        if kind == "pitch":
            return "-".join([note_names[n] for n in transpose_pitches(ngram)])
        elif kind == "rhythm":
            return "-".join([f"{n / 480:.2f}" for n in ngram])

    return {stringify(ngram): score for ngram, score in d.items()}


def get_top_k_ngrams(ngram_dict: dict, k: int):
    items = sorted(ngram_dict.items(), key=lambda x: x[1], reverse=True)
    return dict(items[:k])


def group_ngram_dict(ngram_dict: dict):
    grouped_ngram_dict = defaultdict(dict)
    for ngram, score in ngram_dict.items():
        grouped_ngram_dict[len(ngram)][ngram] = score
    return grouped_ngram_dict


def extract_pitch_class_ngrams_job(midi_file: str, ngram_range: range):
    midi = MidiFile(midi_file)
    assert len(midi.instruments) == 1, "Only support single-track midi files."
    pitch_classes = [note.pitch % 12 for note in midi.instruments[0].notes]

    ngrams = []
    for n in ngram_range:
        for i in range(len(pitch_classes) - n + 1):
            # pitch class
            ngram = [(pc - pitch_classes[i]) % 12 for pc in pitch_classes[i : i + n]]
            ngrams.append(tuple(ngram))

    return get_frequency_dict(ngrams)


def extract_onset_ngrams_job(midi_file: str, ngram_range: range):
    midi = MidiFile(midi_file)
    assert len(midi.instruments) == 1, "Only support single-track midi files."
    midi = quantize_midi(midi)
    onsets = [note.start for note in midi.instruments[0].notes]

    ngrams = []
    for n in ngram_range:
        for i in range(len(onsets) - n + 1):
            # relative to the first bar line
            start = onsets[i] // ticks_per_bar * ticks_per_bar
            ngram = [(onset - start) for onset in onsets[i : i + n]]

            # ignore if there is a note longer than half note
            differences = [ngram[i + 1] - ngram[i] for i in range(len(ngram) - 1)]
            if len(ngram) == 1 or all(0 < d <= ticks_per_bar // 2 for d in differences):
                ngrams.append(tuple(ngram))

    return get_frequency_dict(ngrams)


def extract_ngrams(midi_dir: str, ngram_range: range, kind: str):
    if kind == "pitch":
        job = extract_pitch_class_ngrams_job
    elif kind == "rhythm":
        job = extract_onset_ngrams_job

    midi_files = glob(midi_dir + "/**/*.mid")
    print(f"extracting {kind} ngrams from {len(midi_files)} midi files...")
    with Pool() as pool:
        futures = [pool.apply_async(job, args=(midi_file, ngram_range)) for midi_file in midi_files]
        freq_dicts = [future.get() for future in tqdm(futures)]

    print("merging ngram frequency dictionaries...")
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


def extract():
    midi_dir = "/mnt/nextlab/huangzhijie/midi-preprocess/data/processed/13_dataset_held50/wikifonia_melody_pcset"

    for kind in ["pitch", "rhythm"]:
        ngram_dict, ngram_str_dict = extract_ngrams(midi_dir, range(1, 8 + 1), kind=kind)
        print("saving...")
        with open(f"data/ngram_{kind}.pkl", "wb") as f:
            pickle.dump(ngram_dict, f)
        with open(f"data/ngram_{kind}.json", "w") as f:
            json.dump(ngram_str_dict, f, indent=2)


def render_midi():
    def render_pitch_ngrams(grouped_ngram_dict: dict, dest_dir: str, note_length: int = ticks_per_bar // 8):

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
            midi.dump(os.path.join(dest_dir, f"{n}-gram.mid"))

    def render_rhythm_ngrams(grouped_ngram_dict: dict, dest_dir: str, note_pitch: int = 65):
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
            midi.dump(os.path.join(dest_dir, f"{n}-gram.mid"))

    for kind in ["pitch", "rhythm"]:
        with open(f"data/ngram_{kind}.pkl", "rb") as f:
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
            render_pitch_ngrams(grouped_ngram_dict, "data/pitch")
        elif kind == "rhythm":
            render_rhythm_ngrams(grouped_ngram_dict, "data/rhythm")


if __name__ == "__main__":
    # extract()
    render_midi()
