from typing import Optional, Union, List

import torch
from lightning.pytorch.cli import LightningCLI

from melody_pretrain.dataset import MelodyPretrainDataModule
from melody_pretrain.task import TrainingTask

class CustomLightningCLI(LightningCLI):
    """Lightning CLI does not support loading from checkpoint from now, so we need to override it."""

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--load_from_checkpoint", type=Optional[str], default=None)
        parser.add_argument("--task", type=Union[TrainingTask, List[TrainingTask]])

    def setup_tasks(self) -> None:
        config = self.parser.instantiate_classes(self.config)
        tasks = getattr(config, self.subcommand).task
        if tasks is None:
            return
        if not isinstance(tasks, list):
            tasks = [tasks]
        task_names = [task.task_name for task in tasks]
        assert len(task_names) == len(set(task_names)), "Task names must be unique."
        for task in tasks:
            task_name = task.task_name
            data_collator = task.get_data_collator()
            self.datamodule.register_task(task_name, data_collator)
            if self.subcommand == "fit":
                self.model.register_task(task)

    def before_fit(self) -> None:
        self.setup_tasks()

        # Load from checkpoint
        load_from_checkpoint_path = self.config.fit.load_from_checkpoint
        if load_from_checkpoint_path is not None:
            checkpoint = torch.load(load_from_checkpoint_path)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("Loaded model from checkpoint:", load_from_checkpoint_path)

    def before_test(self) -> None:
        self.setup_tasks()
    
    def before_predict(self) -> None:
        self.setup_tasks()

def cli_main():
    torch.set_float32_matmul_precision("high")
    _ = CustomLightningCLI(datamodule_class=MelodyPretrainDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
