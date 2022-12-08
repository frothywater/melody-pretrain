# Melody Pretrain

1. Prepare n-gram lexicon.
    ```bash
    python lexicon.py prepare \
    --length 8 --top_p 0.1 \
    --midi_dir ../midi-preprocess/data/done \
    --dataset_dir experiment/dataset/pretrain_small

    python lexicon.py render \
    --dataset_dir experiment/dataset/pretrain_small
    ```
2. Prepare dataset. (Keep tokenizer configs the same between pretrain and finetune stages.)
    ```bash
    python prepare_data.py \
    --midi_dir ../midi-preprocess/data/done \
    --dataset_dir experiment/dataset/pretrain_small \
    --granularity 64 --max_bar 128 --pitch_range 0 128
    ```
3. Pretrain.
    ```bash
    python main.py fit --config config/trainer.yaml \
    --config config/model_small.yaml --config config/pretrain_prefix_multi_target.yaml
    ```
4. Finetune.
    ```bash
    python main.py fit --config config/trainer.yaml \
    --config config/model_small.yaml --config config/finetune_clm.yaml
    ```

## Dependencies

- `torch`
- `lightning`
- `jsonargparse[signatures]`
- `miditoolkit`
