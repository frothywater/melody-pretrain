import glob
import os
from argparse import ArgumentParser
from typing import Dict

from miditoolkit import MidiFile

from metric.metrics import compute_all_metrics_for_models


def split_sample_number(file: str):
    basename = os.path.basename(file).replace(".mid", "")
    name, sample_number = basename.split("+")
    return name, int(sample_number)


def main():
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--dest_path", type=str, required=True)
    parser.add_argument("--force_filename", action="store_true")
    parser.add_argument("--check_empty", action="store_true")
    args = parser.parse_args()

    test_files = glob.glob(args.test_dir + "/**/*.mid", recursive=True)
    generated_files = glob.glob(args.generated_dir + "/**/*.mid", recursive=True)
    test_basenames = [os.path.basename(file).replace(".mid", "") for file in test_files]

    # group generated files by name and sample number
    grouped_generated_files: Dict[str, Dict[int, str]] = {}
    for file in generated_files:
        name, sample_number = split_sample_number(file)
        if name not in grouped_generated_files:
            grouped_generated_files[name] = {}
        grouped_generated_files[name][sample_number] = file

    # check if all generated files are sampled the same number of times
    times_sampled = len(next(iter(grouped_generated_files.values())))
    for name, group in grouped_generated_files.items():
        if len(group) != times_sampled:
            raise ValueError(f"Generated files for {name} are not sampled the same number of times {times_sampled}")

    if args.force_filename:
        # check if all test files have a corresponding generated file
        generated_names = set(grouped_generated_files.keys())
        if generated_names != set(test_basenames):
            raise ValueError(f"Generated names {generated_names} do not match test names {test_basenames}")
    else:
        # check if number of one generated file group is equal to number of test files
        if len(grouped_generated_files) != len(test_files):
            raise ValueError(
                f"Number of generated files {len(grouped_generated_files)} does not match number of test files {len(test_files)}"
            )

    # check empty midi files
    if args.check_empty:
        for group in grouped_generated_files.values():
            for file in group.values():
                midi = MidiFile(file)
                if len(midi.instruments) == 0 or len(midi.instruments[0].notes) <= 1:
                    print(f"Warning: generated file {file} is empty")

    generated_files_dict = {
        sample_number: [grouped_generated_files[name][sample_number] for name in grouped_generated_files]
        for sample_number in range(times_sampled)
    }

    # compute metrics
    df = compute_all_metrics_for_models(test_files, generated_files_dict)

    df = df.rename(columns={"model": "group"})
    csv_path = os.path.join(args.dest_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
