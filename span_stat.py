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
    # print("num_noise_tokens:", num_noise_tokens)

    # Randomly select noise ngrams
    ngrams = ngrams.copy()
    permutation = np.random.permutation(len(ngrams))
    permutation = permutation[ngrams[permutation, 1].argsort()[::-1]]

    current_noise_tokens = 0
    covered_indices = np.zeros(num_tokens, dtype=bool)
    noise_ngram_indices = []
    noise_ngram_lengths = []
    for index in permutation:
        start, length, _ = ngrams[index]
        if current_noise_tokens >= num_noise_tokens:
            break
        if np.any(covered_indices[start : start + length]):
            # print(f"skipping [{start}:{start + length}]={length}")
            continue
        noise_ngram_indices.append(index)
        covered_indices[start : start + length] = True
        current_noise_tokens += length
        noise_ngram_lengths.append(length)
        # print(f"select [{start}:{start + length}]={length}, {current_noise_tokens=}")

    noise_rate = current_noise_tokens / num_tokens
    mean_ngram_length = np.mean(noise_ngram_lengths) if len(noise_ngram_lengths) > 0 else np.nan
    return noise_rate, mean_ngram_length


def get_sample_rate_job(filename: str, corruption_rate: float, max_length: int = 256, sample_count: int = 5):
    file = np.load(filename)
    data = file["data"]
    bar_spans = file["bar_spans"]
    pitch_ngrams = file["pitch_ngrams"]
    rhythm_ngrams = file["rhythm_ngrams"]

    def sample_once(bar_spans: np.ndarray, pitch_ngrams: np.ndarray, rhythm_ngrams: np.ndarray):
        def preprocess_ngram_spans(ngram_spans: np.ndarray, start: int, end: int):
            starts = ngram_spans[:, 0]
            lengths = ngram_spans[:, 1]
            ends = starts + lengths
            ngram_spans = ngram_spans[(starts >= start) & (ends <= end)]
            ngram_spans[:, 0] -= start

        bar_spans_ = bar_spans.copy()
        pitch_ngrams_ = pitch_ngrams.copy()
        rhythm_ngrams_ = rhythm_ngrams.copy()

        if len(data) > max_length:
            start = np.random.randint(0, len(data) - max_length)
            end = start + max_length
            bar_spans_ = bar_spans_[(bar_spans_[:, 0] >= start) & (bar_spans_[:, 1] <= end)]
            bar_spans_ = bar_spans_ - start
            preprocess_ngram_spans(pitch_ngrams_, start, end)
            preprocess_ngram_spans(rhythm_ngrams_, start, end)

        length = min(len(data), max_length)
        noise_bar_rate, mean_bar_length = get_noise_bars(bar_spans_, length, corruption_rate)
        noise_pitch_ngram_rate, mean_pitch_ngram_length = get_noise_ngrams(pitch_ngrams_, length, corruption_rate)
        noise_rhythm_ngram_rate, mean_rhythm_ngram_length = get_noise_ngrams(rhythm_ngrams_, length, corruption_rate)
        return (
            noise_bar_rate,
            noise_pitch_ngram_rate,
            noise_rhythm_ngram_rate,
            mean_bar_length,
            mean_pitch_ngram_length,
            mean_rhythm_ngram_length,
        )

    results = [sample_once(bar_spans, pitch_ngrams, rhythm_ngrams) for _ in range(sample_count)]
    results = tuple(zip(*results))
    results = [np.mean(data) if any(x != np.nan for x in data) else np.nan for data in results]
    return results


def get_sample_rate(dir: str, corruption_rate: float):
    filenames = glob(f"{dir}/train/**.npz")

    with Pool() as p:
        (
            noise_bar_rates,
            noise_pitch_ngram_rates,
            noise_rhythm_ngram_rates,
            mean_bar_lengths,
            mean_pitch_ngram_lengths,
            mean_rhythm_ngram_lengths,
        ) = zip(*tqdm(p.imap(partial(get_sample_rate_job, corruption_rate=corruption_rate), filenames)))

    noise_pitch_ngram_rates = np.array(noise_pitch_ngram_rates)
    noise_rhythm_ngram_rates = np.array(noise_rhythm_ngram_rates)
    pitch_ngram_fallback_rate = np.nanmean(noise_pitch_ngram_rates < corruption_rate)
    rhythm_ngram_fallback_rate = np.nanmean(noise_rhythm_ngram_rates < corruption_rate)

    print("noise_bar_rates:", np.nanmean(noise_bar_rates))
    print("noise_pitch_ngram_rates:", np.nanmean(noise_pitch_ngram_rates))
    print("noise_rhythm_ngram_rates:", np.nanmean(noise_rhythm_ngram_rates))
    print("mean_bar_lengths:", np.nanmean(mean_bar_lengths))
    print("mean_pitch_ngram_lengths:", np.nanmean(mean_pitch_ngram_lengths))
    print("mean_rhythm_ngram_lengths:", np.nanmean(mean_rhythm_ngram_lengths))
    print("pitch_ngram_fallback_rate:", pitch_ngram_fallback_rate)
    print("rhythm_ngram_fallback_rate:", rhythm_ngram_fallback_rate)


if __name__ == "__main__":
    rate = 0.8
    print("rate:", rate)
    get_sample_rate(dataset_dir, rate)
