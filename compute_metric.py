import glob
import os
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import LeaveOneOut

from metric.mgeval import core, utils

sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
sns.set_style("ticks")

metric_labels = {
    # "total_used_pitch": "PC",
    # "total_used_note": "NC",
    "total_pitch_class_histogram": "PCH",
    "pitch_class_transition_matrix": "PCTM",
    "pitch_range": "PR",
    "avg_pitch_interval": "PI",
    "avg_IOI": "IOI",
    "note_length_hist": "NLH",
    "note_length_transition_matrix": "NLTM",
}
metric_shapes = {
    # "total_used_pitch": (1,),
    # "total_used_note": (1,),
    "total_pitch_class_histogram": (12,),
    "pitch_class_transition_matrix": (12, 12),
    "pitch_range": (1,),
    "avg_pitch_interval": (1,),
    "avg_IOI": (1,),
    "note_length_hist": (12,),
    "note_length_transition_matrix": (12, 12),
}
metric_names = metric_labels.keys()


def compute_mgeval_feature(files: List[str], metric: str):
    result = np.zeros((len(files),) + metric_shapes[metric])
    for i, file in enumerate(files):
        feature = core.extract_feature(file)
        if metric == "total_used_note":
            metric_result = np.sum(core.metrics().bar_used_note(feature, num_bar=32))
        elif metric == "total_used_pitch":
            metric_result = np.sum(core.metrics().bar_used_pitch(feature, num_bar=32))
        else:
            metric_result = getattr(core.metrics(), metric)(feature)
        result[i] = metric_result
    return result


def cross_valid(set1: np.ndarray, set2: np.ndarray):
    loo = LeaveOneOut()
    num_samples = len(set1)
    loo.get_n_splits(np.arange(num_samples))
    result = np.zeros((num_samples, num_samples))
    for _, test_index in loo.split(np.arange(num_samples)):
        result[test_index[0]] = utils.c_dist(set1[test_index], set2)
    return result.flatten()


def compute_group(test_feature: np.ndarray, generated_files: str, model: str, task: str, metric: str, sample: int):
    generated_feature = compute_mgeval_feature(generated_files, metric)
    inter = cross_valid(generated_feature, test_feature)
    intra_gen = cross_valid(generated_feature, generated_feature)

    data = []
    # data.append({"kind": "mean", "value": np.mean(intra_gen)})
    # data.append({"kind": "std", "value": np.std(intra_gen)})
    # data.append({"kind": "kldiv", "value": utils.kl_dist(intra_gen, inter)})
    data.append({"kind": "oa", "value": utils.overlap_area(intra_gen, inter)})

    print(f"completed: {model=}, {task=}, {metric=}, {sample=}")
    df = pd.DataFrame(data)
    df["model"] = model
    df["task"] = task
    df["metric"] = metric
    df["sample"] = sample
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, required=True)
    args = parser.parse_args()

    generated_dir = os.path.join(args.experiment_dir, "generated")
    models = os.listdir(generated_dir)
    first_model_dir = os.path.join(generated_dir, models[0])
    # tasks = os.listdir(first_model_dir)
    tasks = ["clm"]
    times_repeated = 5
    first_task_dir = os.path.join(first_model_dir, tasks[0])
    basenames = set(name.split("+")[0] for name in os.listdir(first_task_dir) if name.endswith(".mid"))

    test_dir = os.path.join(args.dataset_dir, "midi", "test")
    test_files = [os.path.join(test_dir, basename + ".mid") for basename in basenames]

    for file in test_files:
        if not os.path.exists(file):
            print("test file doesn't exist:", file)
            return
    
    # compute test files first
    print(f"test, {len(test_files)} files")
    with Pool(processes=4) as p:
        test_features = p.starmap(compute_mgeval_feature, [(test_files, metric) for metric in metric_names])
    test_features = {metric: test_feature for metric, test_feature in zip(metric_names, test_features)}

    args_list = []
    for model in models:
        for task in tasks:
            print(f"{model=}, {task=}, {times_repeated=}")
            task_dir = os.path.join(generated_dir, model, task)
            task_files = glob.glob(task_dir + "/**/*.mid", recursive=True)
            for i in range(times_repeated):
                group_files = [file for file in task_files if file.replace(".mid", "").split("+")[1] == str(i)]
                if len(group_files) != len(basenames):
                    print("some generated file doesn't exist.")
                    return
                args_list += [(test_features[metric], group_files, model, task, metric, i) for metric in metric_names]

    # compute generated files
    with Pool(processes=4) as p:
        dataframes = p.starmap(compute_group, [args for args in args_list])

    all_data = pd.concat(dataframes)
    groupby = all_data.groupby(["model", "task", "metric", "kind"])
    mean = groupby["value"].mean()
    std = groupby["value"].std()
    df = pd.DataFrame({"mean": mean, "std": std}).reset_index()
    csv_path = os.path.join(args.experiment_dir, "result", "oa.csv")
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
