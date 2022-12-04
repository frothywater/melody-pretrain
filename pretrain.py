from argparse import ArgumentParser
import os
from lightning import Trainer
from lightning.lite.utilities.seed import seed_everything
# from lightning.lite.strategies import DDPStrategy
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from melody_pretrain.dataset.tokenizer import MIDITokenizer
from melody_pretrain.dataset.data_module import (
    DataCollatorForPrefixMaskedLanguageModeling,
    MelodyPretrainDataModule,
    MultiTargetInfillingMasking,
    RandomNgramMasking,
    SingleSpanMasking,
)
from melody_pretrain.model import MelodyPretrainModel


def main():
    parser = ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--patience", type=int, default=10)
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
    # strategy = DDPStrategy(find_unused_parameters=False)
    trainer: Trainer = Trainer.from_argparse_args(
        args,
        callbacks=[early_stopping, lr_monitor],
        gradient_clip_val=1.0,
        # strategy=strategy,
    )

    # tokenizer
    tokenizer_config_path = os.path.join(args.dataset_dir, "tokenizer_config.json")
    tokenizer = MIDITokenizer.from_config(tokenizer_config_path)

    # data
    masking = MultiTargetInfillingMasking(
        (
            RandomNgramMasking(tokenizer, corruption_rate=0.2, extra_data_field_name="pitch_ngrams"),
            RandomNgramMasking(tokenizer, corruption_rate=0.2, extra_data_field_name="rhythm_ngrams"),
            SingleSpanMasking(tokenizer, corruption_rate=0.5),
        ),
        probabilities=(0.4, 0.4, 0.2),
    )
    data_collator = DataCollatorForPrefixMaskedLanguageModeling(
        tokenizer=tokenizer,
        masking=masking,
        seq_len=args.seq_len,
        random_crop=True,
    )
    data_module = MelodyPretrainDataModule(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        data_collator=data_collator,
        load_ngram_data=True,
    )

    # model
    model = MelodyPretrainModel(tokenizer=tokenizer, **vars(args))

    # train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
