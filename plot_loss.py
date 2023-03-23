import os
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
sns.set_style("ticks")


def get_test_data(experiment_dir: str):
    from tbparse import SummaryReader

    def get_model_variables_and_task(event_file: str):
        path = os.path.relpath(event_file, os.path.join(experiment_dir, "model"))
        variables = path.split("/")[0].split("_")
        task = path.split("/")[1]
        return variables, task

    event_files = glob(f"{experiment_dir}/**/events.out.tfevents.*", recursive=True)

    results = []
    num_variables = None
    for event_file in event_files:
        variables, task = get_model_variables_and_task(event_file)
        if num_variables is None:
            num_variables = len(variables)
        else:
            assert num_variables == len(variables)

        scalars = SummaryReader(event_file).scalars

        # ppl = scalars[scalars.tag == "perplexity"].value
        # if not ppl.empty:
        #     ppl = ppl.iloc[-1]
        #     results.append((*variables, task, ppl))

        if "finetune" in task:
            val_loss = scalars[scalars.tag == "val_loss"].value
            if not val_loss.empty:
                val_loss = val_loss.iloc[-1]
                results.append((*variables, task, val_loss))

    columns = [f"var_{i}" for i in range(num_variables)] + ["task", "value"]
    data = pd.DataFrame(results, columns=columns)
    data.sort_values(by=columns[:-1], inplace=True)
    return data


def save_figure(data: pd.DataFrame, dest_path: str, label_0="var_0", label_1="var_1", label_value="value"):
    if label_0 is not None:
        data = data.rename(columns={"var_0": label_0})
    if label_1 is not None:
        data = data.rename(columns={"var_1": label_1})
    if label_value is not None:
        data = data.rename(columns={"value": label_value})

    task_sorted = sorted(data.task.unique())
    var_0_order = sorted(data[label_0].unique())
    var_1_order = sorted(data[label_1].unique())
    g = sns.FacetGrid(data, col="task", hue=label_0, sharey=False, col_order=task_sorted, hue_order=var_0_order)
    g.map(sns.pointplot, label_1, label_value, order=var_1_order)
    g.add_legend()
    g.savefig(dest_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=False)
    args = parser.parse_args()

    figure_path = os.path.join(args.experiment_dir, "result", "loss.png")
    csv_path = os.path.join(args.experiment_dir, "result", "loss.csv")
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    # data = pd.read_csv(csv_path)
    data = get_test_data(args.experiment_dir)
    data.to_csv(csv_path, index=False)
    save_figure(data, figure_path, label_0="model", label_1="rate", label_value="loss")


if __name__ == "__main__":
    main()
