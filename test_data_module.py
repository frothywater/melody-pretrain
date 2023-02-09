import torch

from melody_pretrain.dataset import (
    DataCollatorForCausalLanguageModeling,
    DataCollatorForPrefixMaskedLanguageModeling,
    DataCollatorForInfilling,
    DatasetBatch,
    MelodyPretrainDataModule,
    MultiTargetInfillingMasking,
    RandomNgramMasking,
    SingleSpanMasking,
    FixedBarMasking,
)
from melody_pretrain.tokenizer import MIDITokenizer
from melody_pretrain.ngram import get_lexicon_size

if __name__ == "__main__":
    # debug
    torch.set_printoptions(linewidth=1000)

    tokenizer = MIDITokenizer()
    print(tokenizer)

    pitch_size, rhythm_size = get_lexicon_size("experiment/dataset/lmd/ngram_data/lexicon.pkl")
    print(f"pitch_size: {pitch_size}, rhythm_size: {rhythm_size}")

    masking = RandomNgramMasking(corruption_rate=0.3, extra_data_field_name="pitch_ngrams")
    data_collator = DataCollatorForPrefixMaskedLanguageModeling(
        masking=masking,
        seq_len=25,
        random_crop=True,
        ngram_classification=True,
        ngram_field_specific_masking=True,
        span_independent_infilling=True,
        permutated_infilling=True,
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

    for batch in data_module.val_dataloader():
        print("ngram_types:")
        print(batch.ngram_types)
        print("input_ids:")
        print(batch.input_ids)
        print("label_ids:")
        print(batch.label_ids)
        # print("padding_mask:")
        # print(batch.padding_mask)
        print("span_indices:")
        print(batch.span_indices)
        print("ngram_ids:")
        print(batch.ngram_ids)
        print("attention_mask:")
        print(batch.attention_mask)
        break
