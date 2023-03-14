import os
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
sns.set_style("ticks")


def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=False)
    args = parser.parse_args()

    csv_path = os.path.join(args.experiment_dir, "result", "oa.csv")
    figure_path = os.path.join(args.experiment_dir, "result", "oa.png")
    data = pd.read_csv(csv_path)
    models = data["model"].unique().sort()

    g = sns.FacetGrid(data, col="metric", row="task", sharey=False, ylim=(0.5, None))
    g.map(sns.barplot, "model", "mean", order=models)
    g.add_legend()
    g.tight_layout()
    g.savefig(figure_path)


if __name__ == "__main__":
    main()
