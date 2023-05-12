import os
from argparse import ArgumentParser
from glob import glob
from typing import Dict, List

from metric.metrics import compute_all_metrics_for_models


def split_sample_number(file: str):
    basename = os.path.basename(file).replace(".mid", "")
    name, sample_number = basename.split("+")
    return name, int(sample_number)


def main():
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--generated_dir", type=str)
    parser.add_argument("--generated_group_dir", type=str)
    parser.add_argument("--dest_path", type=str, required=True)
    parser.add_argument("--force_filename", action="store_true")
    args = parser.parse_args()

    if args.generated_dir is None and args.generated_group_dir is None:
        raise ValueError("Either --generated_dir or --generated_group_dir must be specified")

    test_files = glob(args.test_dir + "/**/*.mid", recursive=True)
    test_basenames = [os.path.basename(file).replace(".mid", "") for file in test_files]

    # scan generated files
    print("Scanning generated files...")
    generated_by_model: Dict[str, List[str]] = {}
    if args.generated_dir is not None:
        generated_files = glob(args.generated_dir + "/**/*.mid", recursive=True)
        last_path_component = os.path.basename(os.path.normpath(args.generated_dir))
        generated_by_model[last_path_component] = generated_files
    elif args.generated_group_dir is not None:
        for dir in os.listdir(args.generated_group_dir):
            model_dir = os.path.join(args.generated_group_dir, dir)
            generated_by_model[dir] = glob(model_dir + "/**/*.mid", recursive=True)

    def group_by_name_and_sample_number(files: List[str]):
        result: Dict[str, Dict[int, str]] = {}
        if args.force_filename:
            for file in files:
                name, sample_number = split_sample_number(file)
                if name not in result:
                    result[name] = {}
                result[name][sample_number] = file
        else:
            # if don't care about correspondence, group by the number of test files
            if len(files) % len(test_files) != 0:
                raise ValueError(
                    f"Number of generated files {len(files)} is not a multiple of test files {len(test_files)}"
                )
            sample_count = len(files) // len(test_files)
            for file_index in range(len(test_files)):
                result[file_index] = {
                    sample_index: files[sample_index * len(test_files) + file_index]
                    for sample_index in range(sample_count)
                }
        return result

    # group generated files by name and sample number
    # {model: {name: {sample_number: file}}}
    print("Grouping generated files by name and sample number...")
    generated_dict = {model: group_by_name_and_sample_number(files) for model, files in generated_by_model.items()}

    # check if all generated files are sampled the same number of times
    print("Checking if all generated files are sampled the same number of times...")
    first_model = next(iter(generated_dict.values()))
    first_name = next(iter(first_model.values()))
    times_sampled = len(first_name)
    for model, model_dict in generated_dict.items():
        for name, group in model_dict.items():
            if len(group) != times_sampled:
                raise ValueError(
                    f"Generated files for {model}/{name} are not sampled the same number of times {times_sampled}"
                )

    def check_group(grouped_generated_files: Dict[str, Dict[int, str]]):
        if args.force_filename:
            # check if all test files have a corresponding generated file
            generated_names = set(grouped_generated_files.keys())
            if generated_names != set(test_basenames):
                raise ValueError(f"Generated names {generated_names} do not match test names {test_basenames}")

    # check if all groups are valid
    print("Checking if all groups are valid...")
    for group in generated_dict.values():
        check_group(group)

    # pack generated files by all models and sample number into a dict
    generated_files_dict = {
        f"{model}/{sample_number}": [model_dict[name][sample_number] for name in model_dict]
        for model, model_dict in generated_dict.items()
        for sample_number in range(times_sampled)
    }

    # compute metrics
    df = compute_all_metrics_for_models(test_files, generated_files_dict)

    # split model name and sample number into separate columns
    df[["model", "sample"]] = df["model"].str.split("/", 1, expand=True)

    csv_path = os.path.join(args.dest_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
