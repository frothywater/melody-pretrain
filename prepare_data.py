import argparse
import os
from glob import glob
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from melody_pretrain.dataset.tokenizer import MIDITokenizer


def prepare_data_job(midi_file: str, dest_path: str, tokenizer: MIDITokenizer):
    array = tokenizer.tokenize(midi_file)
    np.save(dest_path, array)


def prepare_data(midi_dir: str, dest_dir: str, **kwargs):
    midi_files = glob(midi_dir + "/**/*.mid", recursive=True)
    dest_paths = [
        os.path.join(dest_dir, os.path.relpath(midi_file, midi_dir))[:-4] + ".npy" for midi_file in midi_files
    ]

    print(f"preparing {len(midi_files)} midi files...")
    tokenizer = MIDITokenizer(**kwargs)
    with Pool() as pool:
        futures = [
            pool.apply_async(prepare_data_job, args=(midi_file, dest_path, tokenizer))
            for midi_file, dest_path in zip(midi_files, dest_paths)
        ]
        _ = [future.get() for future in tqdm(futures)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--granularity", type=int, default=64)
    parser.add_argument("--max_bar", type=int, default=128)
    parser.add_argument("--pitch_range", type=int, nargs=2, default=(0, 128))
    args = parser.parse_args()
    args.pitch_range = range(*args.pitch_range)

    prepare_data(**vars(args))
