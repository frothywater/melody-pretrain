import argparse
import os
from glob import glob
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from miditoolkit import MidiFile

from melody_pretrain.dataset.tokenizer import MIDITokenizer
from melody_pretrain.dataset.ngram import label_ngram_job


def prepare_data_job(midi_file: str, dest_path: str, tokenizer: MIDITokenizer, lexicon_path: str):
    array_dest_path = os.path.splitext(dest_path)[0] + ".npy"
    bar_dest_path = os.path.splitext(dest_path)[0] + "_bar.npy"
    ngram_dest_path = os.path.splitext(dest_path)[0] + "_ngram.npz"

    midi = MidiFile(midi_file)
    array, bar_spans = tokenizer.encode(midi, return_bar_spans=True)
    np.save(array_dest_path, array)
    np.save(bar_dest_path, bar_spans)

    label_ngram_job(midi_file, ngram_dest_path, lexicon_path)


def prepare_data(midi_dir: str, dataset_dir: str, **kwargs):
    midi_files = glob(midi_dir + "/**/*.mid", recursive=True)
    dest_paths = [os.path.join(dataset_dir, os.path.relpath(midi_file, midi_dir)) for midi_file in midi_files]
    lexicon_path = os.path.join(dataset_dir, "ngram_data", "lexicon.pkl")

    print(f"preparing {len(midi_files)} midi files...")
    tokenizer = MIDITokenizer(**kwargs)
    with Pool() as pool:
        futures = [
            pool.apply_async(prepare_data_job, args=(midi_file, dest_path, tokenizer, lexicon_path))
            for midi_file, dest_path in zip(midi_files, dest_paths)
        ]
        _ = [future.get() for future in tqdm(futures)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, required=True)

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--granularity", type=int, default=64)
    parser.add_argument("--max_bar", type=int, default=128)
    parser.add_argument("--pitch_range", type=int, nargs=2, default=(0, 128))
    args = parser.parse_args()
    args.pitch_range = range(*args.pitch_range)

    prepare_data(**vars(args))
