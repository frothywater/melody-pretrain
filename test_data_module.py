from melody_pretrain.dataset.data_module import (
    MultiTargetInfillingMasking,
    RandomNgramMasking,
    SingleSpanMasking,
    DataCollatorForCausalLanguageModeling,
    DataCollatorForPrefixMaskedLanguageModeling,
    MelodyPretrainDataModule,
    DatasetBatch,
)

from melody_pretrain.dataset.tokenizer import MIDITokenizer

if __name__ == "__main__":
    # debug
    tokenizer = MIDITokenizer()

    # masking = MultiTargetInfillingMasking(
    #     (
    #         RandomNgramMasking(tokenizer, corruption_rate=0.2, extra_data_field_name="pitch_ngrams"),
    #         RandomNgramMasking(tokenizer, corruption_rate=0.2, extra_data_field_name="rhythm_ngrams"),
    #         SingleSpanMasking(tokenizer, corruption_rate=0.5),
    #     ),
    #     probabilities=(0.4, 0.4, 0.2),
    # )
    # data_collator = DataCollatorForPrefixMaskedLanguageModeling(
    #     tokenizer=tokenizer,
    #     masking=masking,
    #     seq_len=20,
    #     random_crop=True,
    # )
    data_collator = DataCollatorForCausalLanguageModeling(
        tokenizer=tokenizer,
        seq_len=20,
        random_crop=True,
    )
    data_module = MelodyPretrainDataModule(
        dataset_dir="experiment/dataset/pretrain_small",
        data_collator=data_collator,
        batch_size=5,
        num_workers=0,
        load_ngram_data=True,
    )

    data_module.setup("")

    dataloader = data_module.val_dataloader()

    for batch in dataloader:
        batch: DatasetBatch
        print(batch.input_ids)
        print(batch.label_ids)
        print(batch.padding_mask)
        print(batch.attention_mask)
        break
