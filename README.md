# Melody Pretrain

## Commands

### 1 Prepare n-gram lexicon
```bash
python lexicon.py prepare \
--length 8 --top_p 0.1 \
--midi_dir ../midi-preprocess/data/done \
--dataset_dir experiment/dataset/pretrain_small

python lexicon.py render \
--dataset_dir experiment/dataset/pretrain_small
```

### 2 Prepare dataset
(Keep tokenizer configs the same between pretrain and finetune stages.)
```bash
python prepare_data.py \
--midi_dir ../midi-preprocess/data/done \
--dataset_dir experiment/dataset/pretrain_small \
--granularity 64 --max_bar 128 --pitch_range 0 128
```

### 3 Pretrain
```bash
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_span.yaml --trainer.devices "1,"
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_bar.yaml --trainer.devices "2,"
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_ngram.yaml --trainer.devices "3,"
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_ngram_multi.yaml --trainer.devices "4,"
```

### 4 Finetune
1. Causal lanugage modeling.
    ```bash
    # bar
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_bar --load_from_checkpoint experiment/model/pretrain_bar/lightning_logs/version_1/checkpoints/epoch=60-train_loss=0.172-val_loss=0.153-step=5000.ckpt --trainer.devices "1,"

    # span
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_span --load_from_checkpoint experiment/model/pretrain_span/lightning_logs/version_1/checkpoints/epoch=60-train_loss=0.228-val_loss=0.278-step=5000.ckpt --trainer.devices "2,"

    # ngram
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram --load_from_checkpoint experiment/model/pretrain_ngram/lightning_logs/version_1/checkpoints/epoch=60-train_loss=0.121-val_loss=0.151-step=5000.ckpt --trainer.devices "3,"

    # ngram with multi-target
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_multi --load_from_checkpoint experiment/model/pretrain_ngram_multi/lightning_logs/version_1/checkpoints/epoch=60-train_loss=0.054-val_loss=0.110-step=5000.ckpt --trainer.devices "4,"
    ```
2. Infilling.
    ```bash
    # bar
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_bar --load_from_checkpoint experiment/model/pretrain_bar/lightning_logs/version_1/checkpoints/epoch=240-step=20000.ckpt --trainer.devices "0,"

    # span
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_span --load_from_checkpoint experiment/model/pretrain_span/lightning_logs/version_1/checkpoints/epoch=240-step=20000.ckpt --trainer.devices "1,"

    # ngram
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram --load_from_checkpoint experiment/model/pretrain_ngram/lightning_logs/version_1/checkpoints/epoch=240-step=20000.ckpt --trainer.devices "2,"

    # ngram with multi-target
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram_multi --load_from_checkpoint experiment/model/pretrain_ngram_multi/lightning_logs/version_1/checkpoints/epoch=240-step=20000.ckpt --trainer.devices "3,"

### 5 Train from scratch
(for comparison)
```bash
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/from_scratch_clm.yaml --trainer.default_root_dir experiment/model/from_scratch_clm --trainer.devices "4,"

python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/from_scratch_infilling.yaml --trainer.default_root_dir experiment/model/from_scratch_infilling --trainer.devices "5,"
```

### 6 Test
1. Causal lanugage modeling.
    ```bash
    # bar
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_bar --ckpt_path experiment/model/finetune_clm_bar/lightning_logs/version_0/checkpoints/epoch=55-train_loss=0.501-val_loss=0.567-step=500.ckpt --trainer.devices "1,"

    # span
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_span --ckpt_path experiment/model/finetune_clm_span/lightning_logs/version_0/checkpoints/epoch=55-train_loss=0.608-val_loss=0.685-step=500.ckpt --trainer.devices "2,"

    # ngram
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram --ckpt_path experiment/model/finetune_clm_ngram/lightning_logs/version_0/checkpoints/epoch=55-train_loss=0.521-val_loss=0.590-step=500.ckpt --trainer.devices "3,"

    # ngram with multi-target
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_multi --ckpt_path experiment/model/finetune_clm_ngram_multi/lightning_logs/version_0/checkpoints/epoch=55-train_loss=0.470-val_loss=0.549-step=500.ckpt --trainer.devices "4,"

    # from scratch
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model/from_scratch_clm --ckpt_path experiment/model/from_scratch_clm/lightning_logs/version_0/checkpoints/epoch=43-train_loss=0.523-val_loss=0.658.ckpt --trainer.devices "5,"
    ```
2. Infilling.
    ```bash
    # bar
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_bar --ckpt_path experiment/model/finetune_infilling_bar/lightning_logs/version_2/checkpoints/epoch=81-val_loss=0.235.ckpt --trainer.devices "0,"

    # span
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_span --ckpt_path experiment/model/finetune_infilling_span/lightning_logs/version_2/checkpoints/epoch=62-val_loss=0.182.ckpt --trainer.devices "1,"

    # ngram
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram --ckpt_path experiment/model/finetune_infilling_ngram/lightning_logs/version_2/checkpoints/epoch=51-val_loss=0.263.ckpt --trainer.devices "2,"

    # ngram with multi-target
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram_multi --ckpt_path experiment/model/finetune_infilling_ngram_multi/lightning_logs/version_2/checkpoints/epoch=09-val_loss=0.232.ckpt --trainer.devices "3,"

    # from scratch
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_infilling.yaml --trainer.default_root_dir experiment/model/from_scratch_infilling --ckpt_path experiment/model/from_scratch_infilling/lightning_logs/version_2/checkpoints/epoch=234-val_loss=0.334.ckpt --trainer.devices "4,"
    ```

### 7 Inference
1. Causal lanugage modeling.
    ```bash
    # bar
    python main.py predict --config config/model_small.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_bar --ckpt_path experiment/model/finetune_clm_bar/lightning_logs/version_3/checkpoints/epoch=111-step=1000.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_bar --trainer.devices "0,"

    # span
    python main.py predict --config config/model_small.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_span --ckpt_path experiment/model/finetune_clm_span/lightning_logs/version_3/checkpoints/epoch=111-step=1000.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_span --trainer.devices "1,"

    # ngram
    python main.py predict --config config/model_small.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram --ckpt_path experiment/model/finetune_clm_ngram/lightning_logs/version_3/checkpoints/epoch=111-step=1000.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_ngram --trainer.devices "2,"

    # ngram with multi-target
    python main.py predict --config config/model_small.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_multi --ckpt_path experiment/model/finetune_clm_ngram_multi/lightning_logs/version_3/checkpoints/epoch=111-step=1000.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_ngram_multi --trainer.devices "3,"

    # from scratch
    python main.py predict --config config/model_small.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/from_scratch_clm --ckpt_path experiment/model/from_scratch_clm/lightning_logs/version_2/checkpoints/epoch=15-val_loss=0.591.ckpt --trainer.callbacks.output_dir experiment/output/from_scratch_clm --trainer.devices "4,"
    ```
2. Infilling.
    ```bash
    # bar
    python main.py predict --config config/model_small.yaml --config config/generate_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_bar --ckpt_path experiment/model/finetune_infilling_bar/lightning_logs/version_2/checkpoints/epoch=81-val_loss=0.235.ckpt --trainer.callbacks.output_dir experiment/output/finetune_infilling_bar --trainer.devices "0,"

    # span
    python main.py predict --config config/model_small.yaml --config config/generate_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_span --ckpt_path experiment/model/finetune_infilling_span/lightning_logs/version_2/checkpoints/epoch=62-val_loss=0.182.ckpt --trainer.callbacks.output_dir experiment/output/finetune_infilling_span --trainer.devices "1,"

    # ngram
    python main.py predict --config config/model_small.yaml --config config/generate_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram --ckpt_path experiment/model/finetune_infilling_ngram/lightning_logs/version_2/checkpoints/epoch=51-val_loss=0.263.ckpt --trainer.callbacks.output_dir experiment/output/finetune_infilling_ngram --trainer.devices "2,"

    # ngram with multi-target
    python main.py predict --config config/model_small.yaml --config config/generate_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram_multi --ckpt_path experiment/model/finetune_infilling_ngram_multi/lightning_logs/version_2/checkpoints/epoch=09-val_loss=0.232.ckpt --trainer.callbacks.output_dir experiment/output/finetune_infilling_ngram_multi --trainer.devices "3,"

    # from scratch
    python main.py predict --config config/model_small.yaml --config config/generate_infilling.yaml --trainer.default_root_dir experiment/model/from_scratch_infilling --ckpt_path experiment/model/from_scratch_infilling/lightning_logs/version_0/checkpoints/epoch=86-val_loss=0.297.ckpt --trainer.callbacks.output_dir experiment/output/from_scratch_infilling --trainer.devices "4,"
    ```

## Dependencies

- `torch`
- `lightning`
- `jsonargparse[signatures]`
- `miditoolkit`
