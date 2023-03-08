from typing import Optional, Union, List

import torch
from lightning.pytorch.cli import LightningCLI

from melody_pretrain.dataset import MelodyPretrainDataModule
from melody_pretrain.task import TrainingTask

class CustomLightningCLI(LightningCLI):
    """Lightning CLI does not support loading from checkpoint from now, so we need to override it."""

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--load_from_checkpoint", type=Optional[str], default=None)
        parser.add_argument("--task", type=Union[None, TrainingTask, List[TrainingTask]])

    def before_fit(self) -> None:
        # Set training tasks
        tasks = self.parser.instantiate_classes(self.config).fit.task
        if tasks is None:
            raise ValueError("Task is not specified.")
        if not isinstance(tasks, list):
            tasks = [tasks]
        for task in tasks:
            task_name = task.task_name
            data_collator = task.get_data_collator()
            self.datamodule.register_task(task_name, data_collator)
            self.model.register_task(task)

        # Load from checkpoint
        load_from_checkpoint_path = self.config.fit.load_from_checkpoint
        if load_from_checkpoint_path is not None:
            checkpoint = torch.load(load_from_checkpoint_path)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("Loaded model from checkpoint:", load_from_checkpoint_path)


def cli_main():
    torch.set_float32_matmul_precision("high")
    _ = CustomLightningCLI(datamodule_class=MelodyPretrainDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
