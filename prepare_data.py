import argparse
import os
from glob import glob
from multiprocessing import Pool
from typing import Optional

import numpy as np
from miditoolkit import MidiFile
from tqdm import tqdm

from melody_pretrain.tokenizer import MIDITokenizer


def filter_ngrams(ngrams: np.ndarray, length: Optional[int], top_p: Optional[float]) -> np.ndarray:
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


def adapt_for_special_tokens_(ngrams: np.ndarray, num_tokens: int):
    # offset by 1 because of <BOS> token
    ngrams["start"] += 1
    ngrams["end"] += 1
    # move any ngrams on the front backward to the <BOS> token
    ngrams["start"][ngrams["start"] == 1] = 0
    # move any ngrams on the end forward to the <EOS> token
    ngrams["end"][ngrams["end"] == num_tokens - 1] = num_tokens


def get_ngrams(
    ngram_file: str, length: Optional[int], top_p: Optional[float], has_bos: bool, num_tokens: int
) -> np.ndarray:
    ngrams = np.load(ngram_file)
    ngrams = filter_ngrams(ngrams, length, top_p)
    if has_bos:
        adapt_for_special_tokens_(ngrams, num_tokens)
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
    data, bar_spans = tokenizer.encode(midi, return_bar_spans=True, include_empty_bar=include_empty_bar)
    results = {"data": data, "bar_spans": bar_spans}

    has_bos = np.all(data[0] == tokenizer.bos_token_ids)
    num_tokens = len(data)
    if mixed_ngram_file:
        results["ngrams"] = get_ngrams(mixed_ngram_file, ngram_length, ngram_top_p, has_bos, num_tokens)
    if pitch_ngram_file:
        results["pitch_ngrams"] = get_ngrams(pitch_ngram_file, ngram_length, ngram_top_p, has_bos, num_tokens)
    if rhythm_ngram_file:
        results["rhythm_ngrams"] = get_ngrams(rhythm_ngram_file, ngram_length, ngram_top_p, has_bos, num_tokens)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    np.savez(dest_path, **results)

    length, _ = data.shape
    return length


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
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
    tokenizer = MIDITokenizer(
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
