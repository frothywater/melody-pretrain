import inspect
from typing import Optional
from lightning.pytorch.cli import LightningCLI
from melody_pretrain.dataset import MelodyPretrainDataModule


class CustomLightningCLI(LightningCLI):
    """Lightning CLI does not support loading from checkpoint from now, so we need to override it."""
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--load_from_checkpoint", type=Optional[str], default=None)

    def before_fit(self) -> None:
        load_from_checkpoint_path = self.config.fit.load_from_checkpoint
        if load_from_checkpoint_path is not None:
            model_class = type(self.model)
            model_config = self.config.fit.model
            model_args = model_config.init_args if "init_args" in model_config else model_config
            self.model = model_class.load_from_checkpoint(
                load_from_checkpoint_path,
                **model_args,
            )
            print("Loaded model from checkpoint:", load_from_checkpoint_path)


def cli_main():
    _ = CustomLightningCLI(datamodule_class=MelodyPretrainDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
