import os
import yaml

from typing import Optional
from argparse import ArgumentParser


def get_infilling_pretrain_task_config(dataset_dir: str, kind: str, corruption_rate: float):
    def get_task_config(kind: str, corruption_rate: float):
        return {"class_path": "InfillingTask", "init_args": {"kind": kind, "corruption_rate": corruption_rate}}

    config = {"model": {"dataset_dir": dataset_dir}, "data": {"dataset_dir": dataset_dir}, "task": []}

    if kind == "span":
        config["task"].append(get_task_config("span", corruption_rate))
    elif kind == "bar":
        config["data"]["load_bar_data"] = True
        config["task"].append(get_task_config("bar", corruption_rate))
    elif kind == "single":
        config["task"].append(get_task_config("single", corruption_rate))
    elif kind == "ngram":
        config["data"]["load_ngram_data"] = True
        config["task"].append(get_task_config("pitch_ngram", corruption_rate))
        config["task"].append(get_task_config("rhythm_ngram", corruption_rate))
    else:
        raise ValueError(f"Unknown kind: {kind}")

    return config


def get_experiment_script(
    experiment_name: str,
    config_path: str,
    experiment_dir: str,
    pretrain_steps: int = 3000,
    finetune_steps: int = 1000,
    ckpt_path: str = "lightning_logs/version_0/checkpoints",
):
    def get_command(stage: str, model_dir: str, ckpt_path: Optional[str] = None):
        subcommand = "fit" if stage != "test" else "test"

        if stage == "pretrain":
            config_path_ = config_path
        elif stage == "finetune_clm":
            config_path_ = "config/finetune/clm.yaml"
        elif stage == "finetune_infilling":
            config_path_ = "config/finetune/infilling.yaml"
        elif stage == "test":
            config_path_ = "config/predict/test_clm.yaml"

        result = f"python main.py {subcommand}"
        result += " --config config/trainer.yaml"
        result += " --config config/model/base.yaml"
        result += f" --config {config_path_}"
        result += f" --trainer.default_root_dir {model_dir}"
        if ckpt_path is not None:
            ckpt_key = "ckpt_path" if stage == "test" else "load_from_checkpoint"
            result += f" --{ckpt_key} {ckpt_path}"
        
        return result

    pretrain_dir = f"{experiment_dir}/{experiment_name}/pretrain"
    finetune_clm_dir = f"{experiment_dir}/{experiment_name}/finetune_clm"
    finetune_infilling_dir = f"{experiment_dir}/{experiment_name}/finetune_infilling"
    pretrain_ckpt_path = f"{pretrain_dir}/{ckpt_path}/step={pretrain_steps}.ckpt"
    finetune_clm_ckpt_path = f"{finetune_clm_dir}/{ckpt_path}/step={finetune_steps}.ckpt"

    lines = [
        f"# {experiment_name}",
        get_command("pretrain", pretrain_dir),
        get_command("finetune_clm", finetune_clm_dir, pretrain_ckpt_path),
        get_command("finetune_infilling", finetune_infilling_dir, pretrain_ckpt_path),
        get_command("test", finetune_clm_dir, finetune_clm_ckpt_path),
    ]
    return "\n".join(lines)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, required=True)
    args = parser.parse_args()

    kinds = ["ngram", "bar", "span", "single"]
    corruption_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    script_path = f"{args.output_dir}/run.sh"
    os.makedirs(args.output_dir, exist_ok=True)

    scripts = []
    for kind in kinds:
        for corruption_rate in reversed(corruption_rates):
            experiment_name = f"{kind}_{int(corruption_rate * 100)}"
            config_path = f"{args.output_dir}/{experiment_name}.yaml"

            config = get_infilling_pretrain_task_config(args.dataset_dir, kind, corruption_rate)
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            script = get_experiment_script(experiment_name, config_path, args.experiment_dir)
            scripts.append(script)

    with open(script_path, "w") as f:
        f.write("\n".join(scripts))
    
    os.system(f"chmod +x {script_path}")


if __name__ == "__main__":
    main()
