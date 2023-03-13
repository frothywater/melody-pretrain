import os
from argparse import ArgumentParser
from glob import glob
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

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
    g = sns.FacetGrid(data, col="task", hue=label_0, sharey=False, col_order=task_sorted, hue_order=var_0_order)
    g.map(sns.lineplot, label_1, label_value)
    g.add_legend()
    g.savefig(dest_path)


# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--experiment_dir", type=str, required=True)
#     args = parser.parse_args()

#     figure_path = os.path.join(args.experiment_dir, "result", "data.png")
#     csv_path = os.path.join(args.experiment_dir, "result", "data.csv")
#     os.makedirs(os.path.dirname(figure_path), exist_ok=True)

#     data = pd.read_csv(csv_path)
#     data = get_test_data(args.experiment_dir)
#     data.to_csv(csv_path, index=False)
#     save_figure(data, figure_path, label_0="model", label_1="rate", label_value="loss")

def main():
    df1 = pd.read_csv("experiment/ablation_infilling/result/data.csv")
    df1["method"] = "infilling"
    df2 = pd.read_csv("experiment/ablation_recovery/result/data.csv")
    df2["method"] = "recovery"

    df = pd.concat([df1, df2])
    df.rename(columns={"var_0": "masking", "var_1": "rate", "value": "loss"}, inplace=True)
    df.sort_values(by=["method", "task", "masking", "rate"], inplace=True)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    sns.lineplot(data=df[df.task == "finetune_clm"], x="rate", y="loss", hue="masking", style="method", ax=axs[0])
    sns.lineplot(data=df[df.task == "finetune_infilling"], x="rate", y="loss", hue="masking", style="method", ax=axs[1])
    axs[0].set_title("clm")
    axs[1].set_title("infilling")

    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.98, .92))

    fig.tight_layout()
    fig.savefig("experiment/ablation.png")
    df.to_csv("experiment/ablation.csv", index=False)

if __name__ == "__main__":
    main()
