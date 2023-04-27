import os

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
    RandomSpanMasking,
    SingleSpanMasking,
)
from melody_pretrain.tokenizer import MIDITokenizer

dataset_dir = "experiment/dataset/melodynet"


def get_data_module(mask: str):
    if mask == "span":
        masking = RandomSpanMasking(corruption_rate=0.5)
    elif mask == "bar":
        masking = RandomBarMasking(corruption_rate=0.5)
    elif mask == "single":
        masking = SingleSpanMasking(corruption_rate=0.5)
    elif mask == "ngram":
        masking = RandomNgramMasking(corruption_rate=0.5, extra_data_field_name="pitch_ngrams")
    elif mask == "fixed_bar":
        masking = FixedBarMasking(6, 4, 6)
    data_collator = DataCollatorForInfilling(
        masking=masking,
        seq_len=50,
        random_crop=False,
        permutated_infilling=True,
    )
    data_module = MelodyPretrainDataModule(
        dataset_dir=dataset_dir,
        batch_size=1,
        load_ngram_data=mask == "ngram",
        load_bar_data=mask == "bar" or mask == "fixed_bar",
        debug=True,
    )
    data_module.register_task("train", data_collator)
    data_module.setup("train")
    return data_module


if __name__ == "__main__":
    torch.set_printoptions(linewidth=1000)
    tokenizer_config_path = os.path.join(dataset_dir, "tokenizer_config.json")
    tokenizer = MIDITokenizer.from_config(tokenizer_config_path)
    print(tokenizer)

    def print_batch(batch: DataBatch, file=None):
        print("filename:", batch.filenames[0], file=file)
        input_tokens = tokenizer.convert_ids_to_tokens(batch.input_ids.squeeze(0).numpy())
        label_tokens = tokenizer.convert_ids_to_tokens(batch.label_ids.squeeze(0).numpy())
        for i, (input_token, label_token) in enumerate(zip(input_tokens, label_tokens)):
            print(i, input_token, "->", label_token, file=file)
        print("attention_kind:", batch.attention_kind, file=file)
        print("length:", batch.lengths[0], file=file)
        print("positional_ids:", batch.positional_ids[0], file=file)
        if batch.source_lengths is not None:
            print("source_length:", batch.source_lengths[0], file=file)

    maskings = ["span", "bar", "single", "ngram"]
    for mask in maskings:
        print(f"masking: {mask}")
        data_module = get_data_module(mask)
        log_file = f"experiment/test/{mask}.txt"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            for i, batch in enumerate(data_module.train_dataloader()):
                if i > 100:
                    break
                print_batch(batch, file=f)
