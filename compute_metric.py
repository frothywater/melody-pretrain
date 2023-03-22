import glob
import os
from argparse import ArgumentParser

from metric.metrics import compute_all_metrics_for_models


def split_sample_number(file: str):
    basename = os.path.basename(file).replace(".mid", "")
    name, sample_number = basename.split("+")
    return name, int(sample_number)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, required=True)
    args = parser.parse_args()
    tasks = ["clm"]

    # get basenames and the number of times each file is repeated
    generated_dir = os.path.join(args.experiment_dir, "generated")
    test_dir = os.path.join(args.dataset_dir, "midi", "test")
    models = os.listdir(generated_dir)
    first_model_dir = os.path.join(generated_dir, models[0])
    first_task_dir = os.path.join(first_model_dir, tasks[0])
    basenames = set(split_sample_number(file)[0] for file in os.listdir(first_task_dir) if file.endswith(".mid"))
    first_basename = next(iter(basenames))
    # assume that all the generated files are repeated the same number of times
    times_repeated = len(
        [file for file in os.listdir(first_task_dir) if split_sample_number(file)[0] == first_basename]
    )

    # check if files exist
    test_files = [os.path.join(test_dir, basename + ".mid") for basename in basenames]
    for file in test_files:
        if not os.path.exists(file):
            return print("test file doesn't exist:", file)
    generated_files_dict = {}
    for model in models:
        for task in tasks:
            task_dir = os.path.join(generated_dir, model, task)
            task_files = glob.glob(task_dir + "/**/*.mid", recursive=True)
            for i in range(times_repeated):
                group_files = [file for file in task_files if split_sample_number(file)[1] == i]
                if len(group_files) != len(basenames):
                    return print("some generated file doesn't exist.")
                generated_files_dict[f"{model}/{task}/{i}"] = group_files

    # compute metrics
    df = compute_all_metrics_for_models(test_files, generated_files_dict)

    # divide generated files into groups
    df["sample"] = df["model"].apply(lambda x: x.split("/")[2] if x != "test" else None)
    df["task"] = df["model"].apply(lambda x: x.split("/")[1] if x != "test" else None)
    df["model"] = df["model"].apply(lambda x: x.split("/")[0] if x != "test" else "test")

    csv_path = os.path.join(args.experiment_dir, "result", "oa.csv")
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
