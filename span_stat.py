import numpy as np
from glob import glob

filenames = glob("experiment/dataset/pretrain_base/train/**.npz")

def job(filename: str):
    file = np.load(filename)
    bar_spans = file["bar_spans"]
    pitch_ngrams = file["pitch_ngrams"]
    rhythm_ngrams = file["rhythm_ngrams"]

    mean_bar_notes = np.mean(bar_spans[:, 1] - bar_spans[:, 0])
    mean_ngram_notes = np.mean(np.concatenate([pitch_ngrams[:, 1], rhythm_ngrams[:, 1]]))
    
    return mean_bar_notes, mean_ngram_notes

from multiprocessing import Pool
from tqdm import tqdm

with Pool(8) as p:
    bar_notes_list, ngram_notes_list = zip(*tqdm(p.imap(job, filenames), total=len(filenames)))

print("mean bar notes:", np.mean(bar_notes_list))
print("mean ngram notes:", np.mean(ngram_notes_list))
