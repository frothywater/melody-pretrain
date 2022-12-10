import inspect
from typing import Optional
from lightning.pytorch.cli import LightningCLI
from melody_pretrain.model import MelodyPretrainModel
from melody_pretrain.dataset import MelodyPretrainDataModule


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--load_from_checkpoint", type=Optional[str], default=None)

    def before_fit(self) -> None:
        load_from_checkpoint_path = self.config["fit"]["load_from_checkpoint"]
        if load_from_checkpoint_path is not None:
            self.model = self._model_class.load_from_checkpoint(
                load_from_checkpoint_path,
                **self.config["fit"]["model"],
            )
            print("Loaded model from checkpoint:", load_from_checkpoint_path)


def cli_main():
    _ = CustomLightningCLI(MelodyPretrainModel, MelodyPretrainDataModule)


if __name__ == "__main__":
    cli_main()
