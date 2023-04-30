import json
import os
from argparse import ArgumentParser

import torch
import yaml

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_dir = config["model"]["init_args"]["dataset_dir"]
    tokenizer_config_path = os.path.join(dataset_dir, "tokenizer_config.json")

    with open(tokenizer_config_path) as f:
        tokenizer_config = json.load(f)

    hyperparameters = {
        "granularity": tokenizer_config["granularity"],
        "max_bar": tokenizer_config["max_bar"],
        "pitch_range": tokenizer_config["pitch_range"],
        "num_layers": config["model"]["init_args"]["num_layers"],
        "num_heads": config["model"]["init_args"]["num_heads"],
        "model_dim": config["model"]["init_args"]["model_dim"],
        "feedforward_dim": config["model"]["init_args"]["feedforward_dim"],
        "embedding_dim": config["model"]["init_args"]["embedding_dim"],
        "dropout": config["model"]["init_args"]["dropout"],
    }
    print(hyperparameters)

    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint["state_dict"]
    for key in list(state_dict):
        new_key = key.replace("melody_pretrain_model.", "")
        state_dict[new_key] = state_dict.pop(key)
        print(new_key)

    new_checkpoint = {"hyperparameters": hyperparameters, "state_dict": state_dict}
    torch.save(new_checkpoint, args.output_path)
