from multiprocessing import Pool
from typing import Dict, List
import os

# make sure numpy doesn't use multiple threads so that we can exploit multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from glob import glob

import muspy
import numpy as np
import pandas as pd

from .custom import get_distinct_ngram_percentage, get_bar_pair_similarity
from .jazzeval import compute_piece_groove_similarity, compute_piece_pitch_entropy, convert_midi_to_events
from .mgeval import extract_feature as mgeval_extract_feature
from .mgeval import metrics as mgeval_metrics
from .utils import cross_valid, overlap_area

# metric shapes
mgeval_metric_shapes = {
    "total_pitch_class_histogram": (12,),
    "pitch_class_transition_matrix": (12, 12),
    "note_length_hist": (12,),
    "note_length_transition_matrix": (12, 12),
}

# metric names
mgeval_scalar_metric_names = [
    "pitch_range",
    "avg_pitch_interval",
    "avg_IOI",
]
mgeval_array_metric_names = [
    "total_pitch_class_histogram",
    "pitch_class_transition_matrix",
    "note_length_hist",
    "note_length_transition_matrix",
]
mgeval_metric_names = mgeval_scalar_metric_names + mgeval_array_metric_names

jazzeval_metric_names = ["pitch_entropy_1", "pitch_entropy_4", "groove_similarity"]
muspy_metric_names = ["scale_consistency", "empty_beat_rate", "empty_measure_rate"]

custom_scalar_metric_names = ["distinct_ngram_percentage_short", "distinct_ngram_percentage_medium", "distinct_ngram_percentage_long"]
custom_array_metric_names = ["bar_pair_similarity"]
custom_metric_names = custom_scalar_metric_names + custom_array_metric_names

absolute_metric_names = mgeval_scalar_metric_names + jazzeval_metric_names + muspy_metric_names + custom_scalar_metric_names
oa_metric_names = mgeval_metric_names
average_error_metric_names = custom_array_metric_names

all_metric_names = mgeval_metric_names + jazzeval_metric_names + muspy_metric_names + custom_metric_names


def compute_metric(midi_file: str, metric_name: str):
    if metric_name in mgeval_metric_names:
        feature = mgeval_extract_feature(midi_file)
        # skip empty files, return 0 for now
        if len(feature["pretty_midi"].instruments) == 0:
            if metric_name in mgeval_metric_shapes:
                return np.zeros(mgeval_metric_shapes[metric_name])
            else:
                return 0
        return getattr(mgeval_metrics(), metric_name)(feature)
    elif metric_name in jazzeval_metric_names:
        events = convert_midi_to_events(midi_file)
        # skip empty files, return 0 for now
        if len(events) == 0:
            return 0
        if metric_name == "pitch_entropy_1":
            return compute_piece_pitch_entropy(events, 1)
        elif metric_name == "pitch_entropy_4":
            return compute_piece_pitch_entropy(events, 4)
        elif metric_name == "groove_similarity":
            return compute_piece_groove_similarity(events)
    elif metric_name in muspy_metric_names:
        midi = muspy.read_midi(midi_file)
        # skip empty files, return 0 for now
        if len(midi.tracks) == 0:
            return 0
        if metric_name == "scale_consistency":
            return muspy.metrics.scale_consistency(midi)
        elif metric_name == "empty_beat_rate":
            return muspy.metrics.empty_beat_rate(midi)
        elif metric_name == "empty_measure_rate":
            return muspy.metrics.empty_measure_rate(midi, 1920)
    elif metric_name in custom_metric_names:
        if metric_name == "distinct_ngram_percentage_short":
            return get_distinct_ngram_percentage(midi_file, n_range=(3, 5))
        elif metric_name == "distinct_ngram_percentage_medium":
            return get_distinct_ngram_percentage(midi_file, n_range=(6, 10))
        elif metric_name == "distinct_ngram_percentage_long":
            return get_distinct_ngram_percentage(midi_file, n_range=(11, 20))
        elif metric_name == "bar_pair_similarity":
            return get_bar_pair_similarity(midi_file)
    raise ValueError(f"Unknown metric: {metric_name}")


def compute_metrics(midi_files: List[str], metric_name: str):
    result = np.stack([compute_metric(midi_file, metric_name) for midi_file in midi_files])
    print(f"computed {metric_name} for {os.path.dirname(midi_files[0])}")
    return result


def compute_oa(generated_metrics: np.ndarray, test_metrics: np.ndarray):
    inter = cross_valid(generated_metrics, test_metrics)
    intra_generated = cross_valid(generated_metrics, generated_metrics)
    oa = overlap_area(intra_generated, inter)
    return oa


def compute_average_error(generated_metrics: np.ndarray, test_metrics: np.ndarray):
    # metrics: (num_files, num_bar_interval) of [similarity, count]
    # 1. average over files
    generated_metrics = generated_metrics["similarity"].sum(axis=0) / generated_metrics["count"].sum(axis=0)
    test_metrics = test_metrics["similarity"].sum(axis=0) / test_metrics["count"].sum(axis=0)
    # 2. mean over bar intervals
    return np.mean(np.abs(generated_metrics - test_metrics))


def compute_absolute_metrics(dir: str):
    midi_files = glob(os.path.join(dir, "*.mid"), recursive=True)
    print(f"absolute metrics in {dir} for {len(midi_files)} files")
    data = {}
    for metric_name in absolute_metric_names:
        data[metric_name] = compute_metrics(midi_files, metric_name).mean()
    return pd.DataFrame(data, index=[0])


def compute_oa_metrics(generated_dir: str, test_dir: str):
    generated_files = glob(os.path.join(generated_dir, "*.mid"), recursive=True)
    test_files = glob(os.path.join(test_dir, "*.mid"), recursive=True)
    print(f"OA metrics in {generated_dir} and {test_dir} for {len(generated_files)} and {len(test_files)} files")
    data = {}
    for metric_name in oa_metric_names:
        generated_metrics = compute_metrics(generated_files, metric_name)
        test_metrics = compute_metrics(test_files, metric_name)
        data[metric_name] = compute_oa(generated_metrics, test_metrics)
    return pd.DataFrame(data, index=[0])


def compute_average_error_metrics(generated_dir: str, test_dir: str):
    generated_files = glob(os.path.join(generated_dir, "*.mid"), recursive=True)
    test_files = glob(os.path.join(test_dir, "*.mid"), recursive=True)
    print(f"average error metrics in {generated_dir} and {test_dir} for {len(generated_files)} and {len(test_files)} files")
    data = {}
    for metric_name in average_error_metric_names:
        generated_metrics = compute_metrics(generated_files, metric_name)
        test_metrics = compute_metrics(test_files, metric_name)
        data[metric_name] = compute_average_error(generated_metrics, test_metrics)
    return pd.DataFrame(data, index=[0])


def compute_all_metrics_for_models(test_files: List[str], generated_files: Dict[str, List[str]]):
    # compute all metrics in parallel
    files_list = [test_files] + list(generated_files.values())
    args_list = [(files, metric_name) for files in files_list for metric_name in all_metric_names]
    with Pool() as pool:
        results = pool.starmap(compute_metrics, args_list)
    # collect results
    num_metrics = len(all_metric_names)
    num_models = len(generated_files)
    test_metrics_list = results[:num_metrics]
    generated_metrics_list = [results[num_metrics * (i + 1) : num_metrics * (i + 2)] for i in range(num_models)]

    # compute mean
    mean_list = []
    for generated_metrics in generated_metrics_list:
        for metric_name in absolute_metric_names:
            i = all_metric_names.index(metric_name)
            mean_list.append(generated_metrics[i].mean())
    test_mean = [test_metrics.mean() for test_metrics in test_metrics_list]
    mean_list = np.array(mean_list).reshape(num_models, len(absolute_metric_names))

    # compute OA
    args_list = []
    for generated_metrics in generated_metrics_list:
        for metric_name in oa_metric_names:
            i = all_metric_names.index(metric_name)
            args_list.append((generated_metrics[i], test_metrics_list[i]))
    with Pool() as pool:
        results = pool.starmap(compute_oa, args_list)
    oa_list = np.array(results).reshape(num_models, len(oa_metric_names))
    
    # compute average error
    error_list = []
    for model_index, generated_metrics in enumerate(generated_metrics_list):
        for metric_name in average_error_metric_names:
            i = all_metric_names.index(metric_name)
            error_list.append(compute_average_error(generated_metrics[i], test_metrics_list[i]))
    error_list = np.array(error_list).reshape(num_models, len(average_error_metric_names))
    
    # gather as DataFrame
    data = []
    for metric_index, metric_name in enumerate(absolute_metric_names):
        data.append({"model": "test", "metric": metric_name, "kind": "mean", "value": test_mean[metric_index]})
    for model_index, generated_dir in enumerate(generated_files):
        for metric_index, metric_name in enumerate(absolute_metric_names):
            data.append({"model": generated_dir, "metric": metric_name, "kind": "mean", "value": mean_list[model_index, metric_index]})
    for model_index, generated_dir in enumerate(generated_files):
        for metric_index, metric_name in enumerate(oa_metric_names):
            data.append({"model": generated_dir, "metric": metric_name, "kind": "oa", "value": oa_list[model_index, metric_index]})
    for model_index, generated_dir in enumerate(generated_files):
        for metric_index, metric_name in enumerate(average_error_metric_names):
            data.append({"model": generated_dir, "metric": metric_name, "kind": "error", "value": error_list[model_index, metric_index]})
    return pd.DataFrame(data)
