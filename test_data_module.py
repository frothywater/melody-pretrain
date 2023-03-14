import torch

from melody_pretrain.dataset import (
    DataBatch,
    DataCollatorForCausalLanguageModeling,
    DataCollatorForInfilling,
    DataCollatorForPrefixMaskedLanguageModeling,
    DataCollatorForRecovery,
    FixedBarMasking,
    MelodyPretrainDataModule,
    RandomBarMasking,
    RandomNgramMasking,
    RandomSkeletonUnitMasking,
    SingleSpanMasking,
)
from melody_pretrain.ngram import get_lexicon_size
from melody_pretrain.tokenizer import MIDITokenizer

if __name__ == "__main__":
    # debug
    torch.set_printoptions(linewidth=1000)

    tokenizer = MIDITokenizer()
    print(tokenizer)

    pitch_size, rhythm_size = get_lexicon_size("experiment/dataset/melodynet/ngram_data/lexicon.pkl")
    print(f"pitch_size: {pitch_size}, rhythm_size: {rhythm_size}")

    data_collator = DataCollatorForRecovery(
        RandomSkeletonUnitMasking(corruption_rate=0.5),
        seq_len=30,
        random_crop=True,
    )
    data_module = MelodyPretrainDataModule(
        dataset_dir="experiment/dataset/melodynet",
        batch_size=1,
        num_workers=0,
        load_ngram_data=True,
        load_bar_data=True,
        load_skeleton_data=True,
    )
    data_module.register_task("recovery", data_collator)

    data_module.setup("train")

    for batch in data_module.train_dataloader():
        print("input_ids:")
        print(batch.input_ids)
        print("label_ids:")
        print(batch.label_ids)
        # print("span_indices:")
        # print(batch.span_indices)
        # print("ngram_type:")
        # print(batch.ngram_type)
        # print("ngram_ids:")
        # print(batch.ngram_ids)
        # print("padding_mask:")
        # print(batch.padding_mask)
        # print("attention_mask:")
        # print(batch.attention_mask)
        break
