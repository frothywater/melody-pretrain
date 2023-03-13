def _get_dataframe(metrics_dict: Dict[str, MetricsGroup], kind: str):
    data = []
    for model_name, outer in metrics_dict.items():
        for metric_name, inner in outer.items():
            if kind in inner:
                metric_label = metric_labels[metric_name]
                data.append({"Type": model_name, "Metric": metric_label, "Value": inner[kind]})
    return pd.DataFrame(data)


def _get_aspect(metrics_dict: dict, kind: str):
    series_count = len(metrics_dict) + (1 if kind == "mean" else 0)
    return 0.1 * series_count + 0.2


def save_overlap_figure(generated_metrics_dict: Dict[str, MetricsGroup], dest_path: str):
    df = _get_dataframe(generated_metrics_dict, "overlap")
    aspect = _get_aspect(generated_metrics_dict, "overlap")
    order = generated_metrics_dict.keys()

    g = sns.FacetGrid(df, col="Metric", hue="Type", ylim=[0.5, 1.0], col_wrap=4, aspect=aspect, palette="muted")
    g.map(sns.barplot, "Type", "Value", order=order)

    g.add_legend()
    g.set(xlabel=None)
    g.set_titles(template="{col_name}")
    g.set_xticklabels(rotation=90)
    g.set_ylabels("Overlapped Area")

    g.savefig(dest_path)


def save_absolute_figure(test_metrics: MetricsGroup, generated_metrics_dict: Dict[str, MetricsGroup], dest_path: str):
    data = {"test": test_metrics, **generated_metrics_dict}
    df = _get_dataframe(data, "mean")
    aspect = _get_aspect(generated_metrics_dict, "mean")
    order = data.keys()

    g = sns.FacetGrid(df, col="Metric", hue="Type", sharey=False, col_wrap=4, aspect=aspect, palette="muted")
    g.map(sns.barplot, "Type", "Value", order=order)

    g.add_legend()
    g.set(xlabel=None)
    g.set_titles(template="{col_name}")
    g.set_xticklabels(rotation=90)
    g.set_ylabels("Mean Value")

    g.savefig(dest_path)