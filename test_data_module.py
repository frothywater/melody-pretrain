import torch

from melody_pretrain.dataset import (
    DataBatch,
    DataCollatorForCausalLanguageModeling,
    DataCollatorForInfilling,
    DataCollatorForMultipleTasks,
    DataCollatorForPrefixMaskedLanguageModeling,
    FixedBarMasking,
    MelodyPretrainDataModule,
    RandomBarMasking,
    RandomNgramMasking,
    SingleSpanMasking,
)
from melody_pretrain.ngram import get_lexicon_size
from melody_pretrain.tokenizer import MIDITokenizer

if __name__ == "__main__":
    # debug
    torch.set_printoptions(linewidth=1000)

    tokenizer = MIDITokenizer()
    print(tokenizer)

    pitch_size, rhythm_size = get_lexicon_size("experiment/dataset/lmd/ngram_data/lexicon.pkl")
    print(f"pitch_size: {pitch_size}, rhythm_size: {rhythm_size}")

    ngram_data_collator = DataCollatorForPrefixMaskedLanguageModeling(
        # masking=RandomBarMasking(corruption_rate=0.3),
        masking=RandomNgramMasking(corruption_rate=0.3, extra_data_field_name="pitch_ngrams"),
        seq_len=25,
        random_crop=True,
        ngram_classification=True,
        permutated_infilling=True,
    )
    single_span_data_collator = DataCollatorForPrefixMaskedLanguageModeling(
        masking=SingleSpanMasking(corruption_rate=0.5),
        seq_len=25,
        random_crop=True,
    )
    data_collator = DataCollatorForMultipleTasks(
        collators=[ngram_data_collator, single_span_data_collator],
        task_names=["ngram", "single_span"],
    )

    data_module = MelodyPretrainDataModule(
        dataset_dir="experiment/dataset/lmd",
        data_collator=data_collator,
        batch_size=1,
        num_workers=0,
        load_ngram_data=True,
        load_bar_data=True,
    )

    data_module.setup("test")

    for batches in data_module.test_dataloader():
        for name, batch in batches.items():
            print(f"task: {name}")
            print("ngram_types:")
            print(batch.ngram_types)
            print("input_ids:")
            print(batch.input_ids)
            print("label_ids:")
            print(batch.label_ids)
            # print("padding_mask:")
            # print(batch.padding_mask)
            # print("span_indices:")
            # print(batch.span_indices)
            # print("ngram_ids:")
            # print(batch.ngram_ids)
            # print("attention_mask:")
            # print(batch.attention_mask)
        break
