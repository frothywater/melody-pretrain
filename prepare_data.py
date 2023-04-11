import argparse
import os
from glob import glob
from multiprocessing import Pool
from typing import Optional

import numpy as np
from miditoolkit import MidiFile
from tqdm import tqdm

from melody_pretrain.tokenizer import MIDITokenizer

bar_span_record_dtype = np.dtype([("start", np.int16), ("end", np.int16)])


def get_bar_spans(midi: MidiFile, include_empty_bar: bool) -> np.ndarray:
    bar_indices = np.array(
        [note.start // (midi.ticks_per_beat * 4) for note in midi.instruments[0].notes], dtype=np.int16
    )
    num_notes = len(bar_indices)

    if not include_empty_bar:
        bar_start_mask = np.concatenate([[True], bar_indices[:-1] != bar_indices[1:]])
        bar_start_indices = np.extract(bar_start_mask, np.arange(num_notes, dtype=np.int16))
        bar_end_indices = np.concatenate([bar_start_indices[1:], [num_notes]])

        bar_spans = np.zeros(len(bar_start_indices), dtype=bar_span_record_dtype)
        bar_spans["start"] = bar_start_indices
        bar_spans["end"] = bar_end_indices
        return bar_spans
    else:
        bar_spans = []
        current_bar, last_index = 0, 0
        for index, bar in enumerate(bar_indices):
            if bar != current_bar:
                bar_spans += [(last_index, index)] * (bar - current_bar)
                current_bar = bar
                last_index = index
        bar_spans.append((last_index, num_notes))
        return np.array(bar_spans, dtype=bar_span_record_dtype)


def get_ngrams(ngram_file: str, length: Optional[int], top_p: Optional[float]) -> np.ndarray:
    ngrams = np.load(ngram_file)

    # filter ngrams
    if length is not None:
        ngrams = ngrams[ngrams["length"] <= length]
    if top_p is not None:
        ngrams = ngrams[ngrams["rank"] <= top_p]

    # remove ngrams covered by any longer ngram (longest match first)
    ngrams = ngrams[np.argsort(ngrams["length"])[::-1]]
    ngram_mask = np.ones(len(ngrams), dtype=bool)
    for i in range(len(ngrams)):
        if not ngram_mask[i]:
            continue
        start, end = ngrams["start"][i], ngrams["end"][i]
        covered_by_current = (ngrams["start"] >= start) & (ngrams["end"] <= end)
        covered_by_current[i] = False
        ngram_mask[covered_by_current] = False
    ngrams = ngrams[ngram_mask]
    return ngrams


def prepare_data_job(
    midi_file: str,
    mixed_ngram_file: Optional[str],
    pitch_ngram_file: Optional[str],
    rhythm_ngram_file: Optional[str],
    dest_path: str,
    tokenizer: MIDITokenizer,
    include_empty_bar: bool,
    ngram_length: Optional[int],
    ngram_top_p: Optional[float],
):
    """Prepare data for a single midi file. Return the length of the encoded data."""
    midi = MidiFile(midi_file)
    data, note_map = tokenizer.encode(midi)
    results = {"data": data, "note_map": note_map}

    results["bar_spans"] = get_bar_spans(midi, include_empty_bar)

    if mixed_ngram_file:
        results["ngrams"] = get_ngrams(mixed_ngram_file, ngram_length, ngram_top_p)
    if pitch_ngram_file:
        results["pitch_ngrams"] = get_ngrams(pitch_ngram_file, ngram_length, ngram_top_p)
    if rhythm_ngram_file:
        results["rhythm_ngrams"] = get_ngrams(rhythm_ngram_file, ngram_length, ngram_top_p)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    np.savez(dest_path, **results)

    length, _ = data.shape
    return length


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--kind", type=str, required=True)
    parser.add_argument("--granularity", type=int, default=64)
    parser.add_argument("--max_bar", type=int, default=128)
    parser.add_argument("--pitch_range", type=int, nargs=2, default=(0, 128))
    parser.add_argument("--ngram_length", type=int)
    parser.add_argument("--ngram_top_p", type=float)
    parser.add_argument("--mixed_ngram_dir", type=str)
    parser.add_argument("--pitch_ngram_dir", type=str)
    parser.add_argument("--rhythm_ngram_dir", type=str)
    parser.add_argument("--include_empty_bar", action="store_true")

    args = parser.parse_args()

    # prepare file args for each midi file
    print("finding files...")
    midi_files = glob(args.midi_dir + "/**/*.mid", recursive=True)
    file_args = []
    for midi_file in midi_files:
        relpath = os.path.relpath(midi_file, args.midi_dir)
        basename = os.path.basename(midi_file)
        dest_path = os.path.join(args.dataset_dir, relpath.replace(".mid", ".npz"))
        mixed_ngram_file, pitch_ngram_file, rhythm_ngram_file = None, None, None
        if args.mixed_ngram_dir:
            mixed_ngram_file = os.path.join(args.mixed_ngram_dir, basename.replace(".mid", ".npy"))
        if args.pitch_ngram_dir:
            pitch_ngram_file = os.path.join(args.pitch_ngram_dir, basename.replace(".mid", ".npy"))
        if args.rhythm_ngram_dir:
            rhythm_ngram_file = os.path.join(args.rhythm_ngram_dir, basename.replace(".mid", ".npy"))
        file_args.append((midi_file, mixed_ngram_file, pitch_ngram_file, rhythm_ngram_file, dest_path))

    # prepare tokenizer
    tokenizer = MIDITokenizer.from_kwargs(
        kind=args.kind,
        granularity=args.granularity,
        max_bar=args.max_bar,
        pitch_range=args.pitch_range,
    )
    # save tokenizer config for later use
    config_path = os.path.join(args.dataset_dir, "tokenizer_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    tokenizer.save_config(config_path)

    print(f"preparing {len(midi_files)} midi files...")
    with Pool() as pool:
        futures = [
            pool.apply_async(
                prepare_data_job,
                args=(*file_arg, tokenizer, args.include_empty_bar, args.ngram_length, args.ngram_top_p),
            )
            for file_arg in file_args
        ]
        lengths = [future.get() for future in tqdm(futures)]

    print(f"average data length: {np.mean(lengths)}")
