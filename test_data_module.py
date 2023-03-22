import torch

from melody_pretrain.dataset import (
    DataBatch,
    DataCollatorForCausalLanguageModeling,
    DataCollatorForFixedInfilling,
    DataCollatorForInfilling,
    DataCollatorForMaskedLanguageModeling,
    DataCollatorForRecovery,
    FixedBarMasking,
    MelodyPretrainDataModule,
    RandomBarMasking,
    RandomNgramMasking,
    RandomSkeletonUnitMasking,
    RandomSpanMasking,
    SingleSpanMasking,
)
from melody_pretrain.ngram import get_lexicon_size
from melody_pretrain.tokenizer import MIDITokenizer

if __name__ == "__main__":
    # debug
    torch.set_printoptions(linewidth=1000)

    tokenizer = MIDITokenizer()
    print(tokenizer)

    # pitch_size, rhythm_size = get_lexicon_size("experiment/dataset/melodynet/ngram_data/lexicon.pkl")
    # print(f"pitch_size: {pitch_size}, rhythm_size: {rhythm_size}")

    data_collator = DataCollatorForRecovery(
        # RandomSpanMasking(corruption_rate=0.5),
        # RandomBarMasking(corruption_rate=0.5),
        # SingleSpanMasking(corruption_rate=0.5),
        RandomNgramMasking(corruption_rate=0.5, extra_data_field_name="pitch_ngrams"),
        # FixedBarMasking(6, 4, 6),
        seq_len=128,
        random_crop=False,
    )
    data_module = MelodyPretrainDataModule(
        dataset_dir="experiment/dataset/wikifonia",
        batch_size=1,
        num_workers=0,
        load_ngram_data=True,
        load_bar_data=True,
    )
    data_module.register_task("train", data_collator)
    data_module.setup("train")

    def print_batch(batch: DataBatch):
        print("filename:", batch.filenames[0])
        input_tokens = tokenizer.convert_ids_to_tokens(batch.input_ids.squeeze(0).numpy())
        label_tokens = tokenizer.convert_ids_to_tokens(batch.label_ids.squeeze(0).numpy())
        for input_token, label_token in zip(input_tokens, label_tokens):
            print(input_token, "->", label_token)
        # print("attention_mask:")
        # print(batch.attention_mask)
        # print("padding_mask:")
        # print(batch.padding_mask)

    for i, batch in enumerate(data_module.train_dataloader()):
        if i > 10:
            break
        print_batch(batch)
