import argparse
import os
from glob import glob
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from miditoolkit import MidiFile

from melody_pretrain.dataset.tokenizer import MIDITokenizer
from melody_pretrain.dataset.ngram import get_ngram_labels


def prepare_data_job(midi_file: str, dest_path: str, tokenizer: MIDITokenizer, lexicon_path: str):
    midi = MidiFile(midi_file)
    data, bar_spans = tokenizer.encode(midi, return_bar_spans=True)
    pitch_ngrams, rhythm_ngrams = get_ngram_labels(midi_file, lexicon_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    np.savez(dest_path, data=data, bar_spans=bar_spans, pitch_ngrams=pitch_ngrams, rhythm_ngrams=rhythm_ngrams)


def prepare_data(midi_dir: str, dataset_dir: str, **kwargs):
    midi_files = glob(midi_dir + "/**/*.mid", recursive=True)
    dest_paths = [
        os.path.join(dataset_dir, os.path.relpath(midi_file, midi_dir)[:-4] + ".npz") for midi_file in midi_files
    ]
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
