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
    tokenizer = MIDITokenizer()
    print(tokenizer)

    pitch_size, rhythm_size = get_lexicon_size("experiment/dataset/lmd/ngram_data/lexicon.pkl")
    print(f"pitch_size: {pitch_size}, rhythm_size: {rhythm_size}")

    masking = MultiTargetInfillingMasking(
        (
            RandomNgramMasking(corruption_rate=0.2, extra_data_field_name="pitch_ngrams", return_ngram_ids=True),
            RandomNgramMasking(corruption_rate=0.2, extra_data_field_name="rhythm_ngrams", return_ngram_ids=True),
            # SingleSpanMasking(corruption_rate=0.5),
        ),
        probabilities=(0.5, 0.5),
    )
    data_collator = DataCollatorForPrefixMaskedLanguageModeling(
        masking=masking,
        seq_len=20,
        random_crop=True,
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
        batch: DatasetBatch
        print(batch.input_ids)
        print(batch.label_ids)
        print(batch.padding_mask)
        print(batch.attention_mask)
        print(batch.extra_label_ids)
        print(batch.ngram_types)
        break
