from functools import partial
from glob import glob
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

dataset_dir = "experiment/dataset/melodynet"


def get_noise_bars(bar_spans: np.ndarray, num_tokens: int, corruption_rate: float) -> np.ndarray:
    num_bars = len(bar_spans)
    num_noise_tokens = int(round(num_tokens * corruption_rate))
    num_noise_tokens = min(max(num_noise_tokens, 1), num_tokens - 1)

    # Randomly select bars until we have enough noise bars
    random_bar_indices = np.arange(num_bars)
    np.random.shuffle(random_bar_indices)
    noise_bar_indices = []
    noise_bar_lengths = []
    current_noise_tokens = 0
    for index in random_bar_indices:
        if current_noise_tokens >= num_noise_tokens:
            break
        noise_bar_indices.append(index)
        start, end = bar_spans[index]
        current_noise_tokens += end - start
        noise_bar_lengths.append(end - start)

    noise_rate = current_noise_tokens / num_tokens
    mean_bar_length = np.mean(noise_bar_lengths)
    return noise_rate, mean_bar_length


def get_noise_ngrams(ngrams: np.ndarray, num_tokens: int, corruption_rate: float) -> np.ndarray:
    num_noise_tokens = int(round(num_tokens * corruption_rate))
    num_noise_tokens = min(max(num_noise_tokens, 1), num_tokens - 1)

    if len(ngrams) == 0:
        return np.nan, np.nan

    permutation = np.random.permutation(len(ngrams))
    permutation = permutation[ngrams[permutation]["length"].argsort()[::-1]]

    current_noise_tokens = 0
    covered_indices = np.zeros(num_tokens, dtype=bool)
    noise_ngram_indices = []
    noise_ngram_lengths = []
    for index in permutation:
        start = ngrams["start"][index]
        end = ngrams["end"][index]
        length = end - start
        if current_noise_tokens >= num_noise_tokens:
            break
        if np.any(covered_indices[start : start + length]):
            continue
        noise_ngram_indices.append(index)
        noise_ngram_lengths.append(length)
        covered_indices[start : start + length] = True
        current_noise_tokens += length

    noise_rate = current_noise_tokens / num_tokens
    mean_ngram_length = np.mean(noise_ngram_lengths) if len(noise_ngram_lengths) > 0 else np.nan
    return noise_rate, mean_ngram_length


def get_sample_rate_job(filename: str, corruption_rate: float, max_length: int = 256):
    file = np.load(filename)
    data = file["data"]
    bar_spans = file["bar_spans"]
    ngrams = file["ngrams"]

    def preprocess_spans(spans: np.ndarray, start: int, end: int):
        starts = spans["start"]
        ends = spans["end"]
        spans = spans[(starts >= start) & (ends <= end)]
        spans["start"] -= start
        spans["end"] -= start

    if len(data) > max_length:
        start = np.random.randint(0, len(data) - max_length)
        end = start + max_length
        preprocess_spans(bar_spans, start, end)
        preprocess_spans(ngrams, start, end)

    length = min(len(data), max_length)
    noise_bar_rate, mean_bar_length = get_noise_bars(bar_spans, length, corruption_rate)
    noise_ngram_rate, mean_ngram_length = get_noise_ngrams(ngrams, length, corruption_rate)
    return (
        noise_bar_rate,
        noise_ngram_rate,
        mean_bar_length,
        mean_ngram_length,
    )


def get_sample_rate(dir: str, corruption_rate: float, file_count: int = 30000):
    filenames = glob(f"{dir}/train/**.npz")
    if file_count is not None:
        filenames = filenames[:file_count]

    with Pool() as p:
        (
            noise_bar_rates,
            noise_ngram_rates,
            mean_bar_lengths,
            mean_ngram_lengths,
        ) = zip(*tqdm(p.imap(partial(get_sample_rate_job, corruption_rate=corruption_rate), filenames)))

    ngram_fallback_rate = np.count_nonzero(np.array(noise_ngram_rates) < corruption_rate) / len(noise_ngram_rates)

    print("noise_bar_rates:", np.nanmean(noise_bar_rates))
    print("noise_ngram_rates:", np.nanmean(noise_ngram_rates))
    print("mean_bar_lengths:", np.nanmean(mean_bar_lengths))
    print("mean_ngram_lengths:", np.nanmean(mean_ngram_lengths))
    print("ngram_fallback_rate:", ngram_fallback_rate)


if __name__ == "__main__":
    rate = 0.8
    print("rate:", rate)
    get_sample_rate(dataset_dir, rate)
