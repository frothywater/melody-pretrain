import os
from argparse import ArgumentParser

from lightning import Trainer
from lightning.lite.utilities.seed import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, StochasticWeightAveraging
from lightning.pytorch.strategies import DDPStrategy

from melody_pretrain.dataset import DataCollatorForCausalLanguageModeling, MelodyPretrainDataModule
from melody_pretrain.model import MelodyPretrainModel
from melody_pretrain.tokenizer import MIDITokenizer


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser = MelodyPretrainDataModule.add_argparse_args(parser)
    parser = MelodyPretrainModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # trainer
    seed_everything(args.seed, workers=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, min_delta=args.min_delta)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    swa = StochasticWeightAveraging(swa_lrs=1e-2)
    trainer: Trainer = Trainer.from_argparse_args(
        args,
        callbacks=[early_stopping, lr_monitor, swa],
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    # tokenizer
    tokenizer_config_path = os.path.join(args.dataset_dir, "tokenizer_config.json")
    tokenizer = MIDITokenizer.from_config(tokenizer_config_path)

    # data
    data_collator = DataCollatorForCausalLanguageModeling(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        random_crop=True,
    )
    data_module = MelodyPretrainDataModule.from_argparse_args(args, data_collator=data_collator)
    # Need setup data module manually to get the total steps for configuring the scheduler
    data_module.setup("fit")
    total_steps = (
        len(data_module.train_dataloader())
        * trainer.max_epochs
        // (trainer.num_devices * trainer.accumulate_grad_batches)
    )

    # model
    if args.checkpoint_path is not None:
        print(f"Load checkpoint from {args.checkpoint_path}")
        model = MelodyPretrainModel.load_from_checkpoint(tokenizer=tokenizer, total_steps=total_steps, **vars(args))
    else:
        print("Train from scratch.")
        model = MelodyPretrainModel(tokenizer=tokenizer, total_steps=total_steps, **vars(args))

    # train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()