# Melody Pretrain

## Commands

### 1 Prepare n-gram lexicon
```bash
python lexicon.py prepare \
--length 8 --top_p 0.1 \
--midi_dir experiment/dataset/pretrain_base/midi \
--dataset_dir experiment/dataset/pretrain_base

python lexicon.py render \
--dataset_dir experiment/dataset/pretrain_base
```

### 2 Prepare dataset
(Keep tokenizer configs the same between pretrain and finetune stages.)
```bash
python prepare_data.py \
--midi_dir experiment/dataset/pretrain_base_ngram/midi \
--dataset_dir experiment/dataset/pretrain_base_ngram \
--granularity 64 --max_bar 128 --pitch_range 0 128

python prepare_data.py \
--midi_dir ../midi-preprocess/data/wikifonia \
--dataset_dir experiment/dataset/wikifonia \
--granularity 64 --max_bar 128 --pitch_range 0 128
```

### 3 Pretrain
```bash
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/span.yaml

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/bar.yaml

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/ngram.yaml

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/single.yaml

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_advanced/ngram.yaml

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_advanced/ngram_plus.yaml

python main.py fit --config config/model/base.yaml --config config/pretrain_advanced/ngram_explicit.yaml
```

### 4 Finetune
1. Causal lanugage modeling.
    ```bash
    # span
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_span --load_from_checkpoint experiment/model/pretrain_span/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # bar
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_bar --load_from_checkpoint experiment/model/pretrain_bar/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # ngram
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram --load_from_checkpoint experiment/model/pretrain_ngram/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # single span
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_single --load_from_checkpoint experiment/model/pretrain_single/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # ngram with multi-target
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_multi --load_from_checkpoint experiment/model/pretrain_ngram_multi/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # ngram with multi-target plus
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_plus --load_from_checkpoint experiment/model/pretrain_ngram_plus/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # explicit
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_explicit --load_from_checkpoint experiment/model/pretrain_ngram_explicit/lightning_logs/version_0/checkpoints/step=5000.ckpt/fp32.ckpt
    ```
2. Infilling.
    ```bash
    # span
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_span --load_from_checkpoint experiment/model/pretrain_span/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # bar
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_bar --load_from_checkpoint experiment/model/pretrain_bar/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # ngram
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram --load_from_checkpoint experiment/model/pretrain_ngram/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # single span
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_single --load_from_checkpoint experiment/model/pretrain_single/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # ngram with multi-target
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram_multi --load_from_checkpoint experiment/model/pretrain_ngram_multi/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # ngram with multi-target plus
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram_plus --load_from_checkpoint experiment/model/pretrain_ngram_plus/lightning_logs/version_0/checkpoints/step=5000.ckpt

    # explicit
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_explicit --load_from_checkpoint experiment/model/pretrain_ngram_explicit/lightning_logs/version_0/checkpoints/step=5000.ckpt/fp32.ckpt
    ```

### 5 Train from scratch
(for comparison)
```bash

```

### 6 Test
1. Causal lanugage modeling.
    ```bash
    
    ```
2. Infilling.
    ```bash
    
    ```

### 7 Inference
1. Causal lanugage modeling.
    ```bash
    
    ```
2. Infilling.
    ```bash
    
    ```

## Dependencies

- `torch`
- `lightning`
- `jsonargparse[signatures]`
- `miditoolkit`
