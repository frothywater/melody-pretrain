from lightning.pytorch.cli import LightningCLI
from melody_pretrain.model import MelodyPretrainModel
from melody_pretrain.dataset import MelodyPretrainDataModule


def cli_main():
    _ = LightningCLI(MelodyPretrainModel, MelodyPretrainDataModule)


if __name__ == "__main__":
    cli_main()
