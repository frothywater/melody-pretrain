import os
import yaml

from typing import Optional, Union, List
from argparse import ArgumentParser


def get_pretrain_config(dataset_dir: str, task: str, kind: str, corruption_rate: float):
    config = {"model": {"dataset_dir": dataset_dir}, "data": {"dataset_dir": dataset_dir}, "task": []}

    def get_task_config(
        task: str, masking: Union[str, List[str]], corruption_rate: float, weight: Optional[float] = None, **kwargs
    ):
        if task == "infilling":
            task_name = "InfillingTask"
        elif task == "recovery":
            task_name = "RecoveryTask"
        elif task == "rewriting":
            task_name = "RewritingTask"
        else:
            raise ValueError(f"Unknown task: {task}")
        result = {"class_path": task_name, "init_args": {"kind": masking, "corruption_rate": corruption_rate, **kwargs}}
        if weight is not None:
            result["init_args"]["weight"] = weight
        return result

    if kind == "span":
        config["task"].append(get_task_config(task, "span", corruption_rate))
    elif kind == "bar":
        config["data"]["load_bar_data"] = True
        config["task"].append(get_task_config(task, "bar", corruption_rate))
    elif kind == "single":
        config["task"].append(get_task_config(task, "single", corruption_rate))
    elif kind == "ngram":
        config["data"]["load_ngram_data"] = True
        config["task"].append(get_task_config(task, "pitch_ngram", corruption_rate))
        config["task"].append(get_task_config(task, "rhythm_ngram", corruption_rate))
    elif kind == "skeleton":
        config["data"]["load_skeleton_data"] = True
        config["task"].append(get_task_config(task, "skeleton", corruption_rate))
    elif kind == "ngram-single":
        config["data"]["load_ngram_data"] = True
        config["task"].append(get_task_config(task, "pitch_ngram", corruption_rate, weight=0.5))
        config["task"].append(get_task_config(task, "rhythm_ngram", corruption_rate, weight=0.5))
        config["task"].append(get_task_config(task, "single", corruption_rate))
    elif kind == "single-skeleton":
        config["data"]["load_skeleton_data"] = True
        config["task"].append(get_task_config(task, "single", corruption_rate))
        config["task"].append(get_task_config(task, "skeleton", corruption_rate, field_specific_masking=True))
    else:
        raise ValueError(f"Unknown kind: {kind}")

    return config


def get_model_script(
    experiment_name: str,
    config_path: str,
    experiment_dir: str,
    pretrain_steps: int = 3000,
    finetune_steps: int = 1000,
    ckpt_path: str = "lightning_logs/version_0/checkpoints",
):
    def get_command(stage: str, model_dir: str, ckpt_path: Optional[str] = None):
        if stage == "pretrain":
            subcommand = "fit"
            config_path_ = config_path
        elif stage == "finetune_clm":
            subcommand = "fit"
            config_path_ = "config/finetune/clm.yaml"
        elif stage == "finetune_infilling":
            subcommand = "fit"
            config_path_ = "config/finetune/infilling.yaml"
        elif stage == "test":
            subcommand = "test"
            config_path_ = "config/predict/test_clm.yaml"
        elif stage == "generate_clm":
            subcommand = "predict"
            config_path_ = "config/predict/generate_clm.yaml"
        elif stage == "generate_infilling":
            subcommand = "predict"
            config_path_ = "config/predict/generate_infilling.yaml"
        else:
            raise ValueError(f"Unknown stage: {stage}")

        result = f"python main.py {subcommand}"
        if "generate" not in stage:
            result += " --config config/trainer.yaml"
        result += " --config config/model/base.yaml"
        result += f" --config {config_path_}"
        result += f" --trainer.default_root_dir {model_dir}"
        if ckpt_path is not None:
            ckpt_key = "ckpt_path" if stage == "test" or "generate" in stage else "load_from_checkpoint"
            result += f" --{ckpt_key} {ckpt_path}"
        if "generate" in stage:
            subdir = stage.split("_")[1]
            result += f" --trainer.callbacks.output_dir {experiment_dir}/generated/{experiment_name}/{subdir}"

        return result

    pretrain_dir = f"{experiment_dir}/model/{experiment_name}/pretrain"
    finetune_clm_dir = f"{experiment_dir}/model/{experiment_name}/finetune_clm"
    finetune_infilling_dir = f"{experiment_dir}/model/{experiment_name}/finetune_infilling"

    pretrain_ckpt_path = f"{pretrain_dir}/{ckpt_path}/step={pretrain_steps}.ckpt"
    finetune_clm_ckpt_path = f"{finetune_clm_dir}/{ckpt_path}/step={finetune_steps}.ckpt"
    finetune_infilling_ckpt_path = f"{finetune_infilling_dir}/{ckpt_path}/step={finetune_steps}.ckpt"

    lines = [
        f"# {experiment_name}",
        get_command("pretrain", pretrain_dir),
        get_command("finetune_clm", finetune_clm_dir, pretrain_ckpt_path),
        get_command("finetune_infilling", finetune_infilling_dir, pretrain_ckpt_path),
        get_command("test", finetune_clm_dir, finetune_clm_ckpt_path),
    ]
    predict_lines = [
        get_command("generate_infilling", finetune_infilling_dir, finetune_infilling_ckpt_path),
        get_command("generate_clm", finetune_clm_dir, finetune_clm_ckpt_path),
    ]
    return "\n".join(lines), " &\n".join(predict_lines)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, required=True)
    args = parser.parse_args()

    script_path = f"{args.experiment_dir}/script/run.sh"
    predict_script_path = f"{args.experiment_dir}/script/generate.sh"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    scripts = []
    predict_scripts = []

    task = "recovery" if "recovery" in args.experiment_dir else "infilling"
    # kinds = ["span", "bar", "single", "ngram", "ngram-single"]
    kinds = ["single-skeleton", "skeleton"]
    # corruption_rates = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    corruption_rates = [0.8, 0.7]
    for kind in kinds:
        for corruption_rate in corruption_rates:
            experiment_name = f"{kind}_{int(corruption_rate * 100)}"
            config_path = f"{args.experiment_dir}/script/{experiment_name}.yaml"

            config = get_pretrain_config(args.dataset_dir, task, kind, corruption_rate)
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            script, predict_script = get_model_script(experiment_name, config_path, args.experiment_dir)
            scripts.append(script)
            if corruption_rate == 0.8:
                predict_scripts.append(predict_script)

    # scripts.append(f"python plot_loss.py --experiment_dir {args.experiment_dir}")
    with open(script_path, "w") as f:
        f.write("\n".join(scripts))
    os.system(f"chmod +x {script_path}")

    # predict_script_path.append(f"python compute_metric.py --experiment_dir {args.experiment_dir} --dataset_dir {args.dataset_dir}")
    with open(predict_script_path, "w") as f:
        f.write(" &\n".join(predict_scripts))
    os.system(f"chmod +x {predict_script_path}")


if __name__ == "__main__":
    main()
