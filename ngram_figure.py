import os
import pickle
from glob import glob
from multiprocessing import Pool
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
sns.set_style("ticks")

min_ngram_length = 2
max_ngram_length = 12
ngram_length_count = max_ngram_length - min_ngram_length + 1
length_groups = [(2, 5), (6, 9), (10, 12)]
length_group_names = ["2-5", "6-9", "10-12"]
pitch_ngram_dir = "experiment/dataset/melodynet/ngram/ngram_pitch"
rhythm_ngram_dir = "experiment/dataset/melodynet/ngram/ngram_rhythm"


def get_ngram_sets(ngram_file: str):
    ngram_sets = {i: set() for i in range(min_ngram_length, max_ngram_length + 1)}
    with open(ngram_file, "rb") as f:
        ngrams: List[Tuple[Tuple, int]] = pickle.load(f)
        for ngram, _ in ngrams:
            ngram_sets[len(ngram)].add(ngram)
    return ngram_sets


def get_ngram_count_by_file(dir: str, max_files: int = None):
    ngram_files = glob(os.path.join(dir, "*.pkl"))
    np.random.shuffle(ngram_files)
    if max_files is not None:
        ngram_files = ngram_files[:max_files]

    print(f"loading ngrams from {len(ngram_files)} files...")
    all_ngram_sets = {i: set() for i in range(min_ngram_length, max_ngram_length + 1)}
    ngram_count_by_length = np.zeros((len(ngram_files), ngram_length_count), dtype=np.int64)
    with Pool() as pool:
        futures = [pool.apply_async(get_ngram_sets, (ngram_file,)) for ngram_file in ngram_files]
        for i, future in tqdm(enumerate(futures), total=len(futures)):
            one_ngram_sets = future.get()
            for length, one_ngram_set in one_ngram_sets.items():
                all_ngram_sets[length].update(one_ngram_set)
                ngram_count_by_length[i, length - min_ngram_length] = len(all_ngram_sets[length])
    return ngram_count_by_length


if __name__ == "__main__":
    # pitch_ngram_count = get_ngram_count_by_file(pitch_ngram_dir)
    # np.save("experiment/result/pitch_ngram_count.npy", pitch_ngram_count)
    # rhythm_ngram_count = get_ngram_count_by_file(rhythm_ngram_dir)
    # np.save("experiment/result/rhythm_ngram_count.npy", rhythm_ngram_count)

    pitch_ngram_count = np.load("experiment/result/pitch_ngram_count.npy")
    rhythm_ngram_count = np.load("experiment/result/rhythm_ngram_count.npy")
    data = pd.DataFrame()
    num_files = pitch_ngram_count.shape[0]
    file_count = np.arange(num_files) + 1
    for (start, end), name in zip(length_groups, length_group_names):
        pitch_count = np.sum(pitch_ngram_count[:, start - min_ngram_length : end - min_ngram_length + 1], axis=1)
        rhythm_count = np.sum(rhythm_ngram_count[:, start - min_ngram_length : end - min_ngram_length + 1], axis=1)
        assert len(pitch_count) == len(rhythm_count) == num_files
        pitch_data = pd.DataFrame({"file_count": file_count, "ngram_count": pitch_count, "kind": "pitch", "length": name})
        rhythm_data = pd.DataFrame({"file_count": file_count, "ngram_count": rhythm_count, "kind": "rhythm", "length": name})
        data = pd.concat([data, pitch_data, rhythm_data], axis=0)
    
    g = sns.FacetGrid(data, col="length", hue="kind", sharey=False, col_order=length_group_names, hue_order=["pitch", "rhythm"])
    g.map(sns.lineplot, "file_count", "ngram_count")
    g.set(xlabel="Number of files", ylabel="Number of ngrams")
    g.add_legend()
    for ax in g.axes.flat:
        ax.ticklabel_format(style="sci", scilimits=(0, 0))
    g.savefig("experiment/result/ngram_count.png")
