import argparse
import os
from glob import glob
from multiprocessing import Pool
from typing import Optional

import numpy as np
from miditoolkit import MidiFile
from tqdm import tqdm

from melody_pretrain.tokenizer import MIDITokenizer


def adapt_for_special_tokens_(ngrams: np.ndarray, num_tokens: int):
    # offset by 1 because of <BOS> token
    ngrams["start"] += 1
    ngrams["end"] += 1
    # move any ngrams on the front backward to the <BOS> token
    ngrams["start"][ngrams["start"] == 1] = 0
    # move any ngrams on the end forward to the <EOS> token
    ngrams["end"][ngrams["end"] == num_tokens - 1] = num_tokens


def prepare_data_job(
    midi_file: str,
    dest_path: str,
    tokenizer: MIDITokenizer,
    ngram_file: Optional[str],
    include_empty_bar: bool,
    skeleton_note_indices: Optional[np.ndarray],
):
    """Prepare data for a single midi file. Return the length of the encoded data."""
    midi = MidiFile(midi_file)
    data, bar_spans = tokenizer.encode(midi, return_bar_spans=True, include_empty_bar=include_empty_bar)
    results = {"data": data, "bar_spans": bar_spans}

    if ngram_file:
        ngrams = np.load(ngram_file)
        # take care of <BOS> token
        if np.all(data[0] == tokenizer.bos_token_ids):
            adapt_for_special_tokens_(ngrams, len(data))
        results["ngrams"] = ngrams

    if skeleton_note_indices is not None:
        # take care of <BOS> token
        if np.all(data[0] == tokenizer.bos_token_ids):
            skeleton_note_indices += 1
        results["skeleton_note_indices"] = skeleton_note_indices

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    np.savez(dest_path, **results)

    length, _ = data.shape
    return length


def prepare_data(midi_dir: str, dataset_dir: str, **kwargs):
    midi_files = glob(midi_dir + "/**/*.mid", recursive=True)
    print(f"preparing {len(midi_files)} midi files...")
    dest_paths = [
        os.path.join(dataset_dir, os.path.relpath(midi_file, midi_dir).replace(".mid", ".npz"))
        for midi_file in midi_files
    ]

    include_ngram_label = kwargs.pop("include_ngram_label")
    if include_ngram_label:
        ngram_files = [
            os.path.join(dataset_dir, "ngram", "label", os.path.basename(midi_file).replace(".mid", ".npy"))
            for midi_file in midi_files
        ]
    else:
        ngram_files = [None for _ in midi_files]

    include_empty_bar = kwargs.pop("include_empty_bar")

    skeleton_info_path = kwargs.pop("skeleton_info_path")
    if skeleton_info_path is not None:
        skeleton_info = np.load(skeleton_info_path)
        skeleton_note_indices_list = [skeleton_info[os.path.basename(midi_file)] for midi_file in midi_files]
    else:
        skeleton_note_indices_list = [None for _ in midi_files]

    tokenizer = MIDITokenizer(**kwargs)
    config_path = os.path.join(args.dataset_dir, "tokenizer_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    tokenizer.save_config(config_path)

    with Pool() as pool:
        futures = [
            pool.apply_async(
                prepare_data_job,
                args=(midi_file, dest_path, tokenizer, ngram_file, include_empty_bar, skeleton_note_indices),
            )
            for midi_file, dest_path, ngram_file, skeleton_note_indices in zip(
                midi_files, dest_paths, ngram_files, skeleton_note_indices_list
            )
        ]
        lengths = [future.get() for future in tqdm(futures)]

    print(f"average data length: {np.mean(lengths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, required=True)

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--granularity", type=int, default=64)
    parser.add_argument("--max_bar", type=int, default=128)
    parser.add_argument("--pitch_range", type=int, nargs=2, default=(0, 128))
    parser.add_argument("--include_ngram_label", action="store_true")
    parser.add_argument("--include_empty_bar", action="store_true")

    parser.add_argument("--skeleton_info_path", type=str)

    args = parser.parse_args()
    prepare_data(**vars(args))
