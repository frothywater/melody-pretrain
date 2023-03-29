import json
import math
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from miditoolkit import MidiFile
from tqdm import tqdm

ticks_per_bar = 1920


class Ngram:
    @staticmethod
    def from_notes(
        pitches: Optional[List[int]] = None,
        positions: Optional[List[int]] = None,
        durations: Optional[List[int]] = None,
    ) -> Optional[Tuple]:
        raise NotImplementedError

    @staticmethod
    def split(ngram: Tuple) -> List[Tuple]:
        raise NotImplementedError


class PitchClassNgram(Ngram):
    @staticmethod
    def from_notes(pitches: List[int], **kwargs) -> Tuple:
        return tuple((pitch - pitches[0]) % 12 for pitch in pitches)

    @staticmethod
    def split(ngram: Tuple) -> List[Tuple]:
        return [(0, (ngram[i + 1] - ngram[i]) % 12) for i in range(len(ngram) - 1)]


class BarOnsetNgram(Ngram):
    @staticmethod
    def from_notes(positions: List[int], **kwargs) -> Optional[Tuple]:
        differences = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        if any(difference >= ticks_per_bar for difference in differences):
            return None
        bar_start = positions[0] // ticks_per_bar * ticks_per_bar
        return tuple(position - bar_start for position in positions)

    @staticmethod
    def split(ngram: Tuple) -> List[Tuple]:
        result = []
        for i in range(len(ngram) - 1):
            bar_start = ngram[i] // ticks_per_bar * ticks_per_bar
            result.append((ngram[i] - bar_start, ngram[i + 1] - bar_start))
        return result


class MixedNgram(Ngram):
    @staticmethod
    def from_notes(pitches: List[int], positions: List[int], **kwargs) -> Optional[Tuple]:
        differences = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        if any(difference >= ticks_per_bar for difference in differences):
            return None
        # bar_start = positions[0] // ticks_per_bar * ticks_per_bar
        bar_start = positions[0]

        pcs = ((pitch - pitches[0]) % 12 for pitch in pitches)
        onsets = (position - bar_start for position in positions)
        return tuple(zip(pcs, onsets))

    @staticmethod
    def split(ngram: Tuple) -> List[Tuple]:
        result = []
        pcs, onsets = zip(*ngram)
        for i in range(len(ngram) - 1):
            # bar_start = onsets[i] // ticks_per_bar * ticks_per_bar
            bar_start = onsets[i]
            pc_unit = (0, (pcs[i + 1] - pcs[i]) % 12)
            onset_unit = (onsets[i] - bar_start, onsets[i + 1] - bar_start)
            result.append(tuple(zip(pc_unit, onset_unit)))
        return result


class NgramExtractor:
    def __init__(self, n_range: Tuple[int, int], ngram_type: Type[Ngram]) -> None:
        assert n_range[0] >= 2
        self.n_range = range(n_range[0], n_range[1] + 1)
        self.ngram_type = ngram_type

    def save_config(self, dest_path: str):
        config = {"n_range": (self.n_range.start, self.n_range.stop - 1), "ngram_type": self.ngram_type.__name__}
        with open(dest_path, "w") as f:
            json.dump(config, f)

    @staticmethod
    def from_config(config_path: str) -> "NgramExtractor":
        with open(config_path, "r") as f:
            config = json.load(f)
        n_range = tuple(config["n_range"])
        ngram_type = globals()[config["ngram_type"]]
        return NgramExtractor(n_range, ngram_type)

    def get_ngrams(self, midi_file: str) -> Dict[Tuple, int]:
        midi = MidiFile(midi_file)
        assert len(midi.instruments) == 1
        notes = midi.instruments[0].notes
        ngrams = []
        indices = []
        for n in range(2, self.n_range.stop):
            for i in range(len(notes) - n + 1):
                ngram = self.ngram_type.from_notes(
                    pitches=[note.pitch for note in notes[i : i + n]],
                    positions=[note.start for note in notes[i : i + n]],
                    durations=[note.end - note.start for note in notes[i : i + n]],
                )
                if ngram is not None:
                    ngrams.append(ngram)
                    indices.append(i)

        result = {ngram: index for ngram, index in zip(ngrams, indices)}
        return result

    def get_ngram_frequency(self, ngram_file: str):
        with open(ngram_file, "rb") as f:
            ngrams = pickle.load(f)
        frequency = defaultdict(int)
        frequency_by_length = defaultdict(int)
        for ngram in ngrams:
            frequency[ngram] += 1
            frequency_by_length[len(ngram)] += 1
        return frequency, frequency_by_length

    def extract_ngrams_file(self, midi_file: str, dest_path: str):
        ngrams = self.get_ngrams(midi_file)
        with open(dest_path, "wb") as f:
            pickle.dump(ngrams, f)

    def extract_ngrams(self, midi_files: List[str], dest_dir: str):
        print(f"extracting ngrams from {len(midi_files)} files to {dest_dir}...")
        os.makedirs(dest_dir, exist_ok=True)
        dest_paths = [
            os.path.join(dest_dir, os.path.basename(midi_file).replace(".mid", ".pkl")) for midi_file in midi_files
        ]
        with Pool() as pool:
            futures = [
                pool.apply_async(self.extract_ngrams_file, (midi_file, dest_path))
                for midi_file, dest_path in zip(midi_files, dest_paths)
            ]
            _ = [future.get() for future in tqdm(futures)]

    def build_lexicon(self, ngram_files: List[str], dest_path: str):
        print(f"loading ngrams from {len(ngram_files)} files...")
        with Pool() as pool:
            futures = [pool.apply_async(self.get_ngram_frequency, (ngram_file,)) for ngram_file in ngram_files]
            results = [future.get() for future in tqdm(futures)]
        frequency_list, frequency_by_length_list = zip(*results)

        print("combining frequencies...")
        frequency = defaultdict(int)
        frequency_by_length = defaultdict(int)
        for freq, freq_by_length in zip(frequency_list, frequency_by_length_list):
            for ngram, count in freq.items():
                frequency[ngram] += count
            for length, count in freq_by_length.items():
                frequency_by_length[length] += count
        print(f"total ngrams: {len(frequency)}")

        print("calculating probs...")
        for ngram, count in frequency.items():
            frequency[ngram] = count / frequency_by_length[len(ngram)]

        print("calculating scores...")
        scores = self.get_ngram_scores(frequency, frequency_by_length)

        print(f"saving to {dest_path}...")
        with open(dest_path, "wb") as f:
            pickle.dump(scores, f)

    def get_ngram_scores(
        self,
        prob: Dict[Tuple, float],
        count_by_length: Dict[int, float],
        mode: Union[Literal["t-test"], Literal["pmi"]] = "t-test",
    ):
        """
        Args:
            mode: "pmi" or "tscore", which method to use.
                - t-test: Reference: ERNIE-GEN https://arxiv.org/abs/2001.11314
                - pmi: Pointwise Mutual Information. Reference: PMI-Masking https://arxiv.org/abs/2010.01825
        """

        def _get_score(ngram: Tuple) -> float:
            iid_product = 1
            for unit in self.ngram_type.split(ngram):
                iid_product *= prob[unit]
            product = prob[ngram]

            if mode == "t-test":
                sigma_sqr = product * (1 - product)
                total_count = count_by_length[len(ngram)]
                return (product - iid_product) / math.sqrt(sigma_sqr * total_count)
            elif mode == "pmi":
                return math.log2(product / iid_product)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        scores: Dict[Tuple, float] = {}
        for ngram in tqdm(prob):
            if len(ngram) in self.n_range:
                scores[ngram] = _get_score(ngram)
        return scores

    def get_ngram_labels(self, ngram_file: str, lexicon: Dict[Tuple, float]) -> np.ndarray:
        with open(ngram_file, "rb") as f:
            ngrams: Dict[Tuple, int] = pickle.load(f)
        ngrams = {ngram: index for ngram, index in ngrams.items() if len(ngram) in self.n_range}

        result = np.zeros(len(ngrams), dtype=[("start", "i4"), ("end", "i4"), ("length", "i4"), ("score", "f4")])
        result["start"] = list(ngrams.values())
        result["length"] = [len(ngram) for ngram in ngrams]
        result["end"] = result["start"] + result["length"]
        result["score"] = [lexicon[ngram] for ngram in ngrams]
        result.sort(order=["start", "length", "score"], axis=0)
        return result

    def prepare_ngram_labels(self, ngram_files: List[str], lexicon_path: str, dest_dir: str):
        print(f"loading lexicon from {lexicon_path}...")
        with open(lexicon_path, "rb") as f:
            lexicon: Dict[Tuple, float] = pickle.load(f)

        print(f"preparing ngram labels for {len(ngram_files)} files to {dest_dir}...")
        os.makedirs(dest_dir, exist_ok=True)
        dest_paths = [
            os.path.join(dest_dir, os.path.basename(ngram_file).replace(".pkl", ".npy")) for ngram_file in ngram_files
        ]

        for ngram_file, dest_path in tqdm(zip(ngram_files, dest_paths), total=len(ngram_files)):
            ngram_labels = self.get_ngram_labels(ngram_file, lexicon)
            np.save(dest_path, ngram_labels)
