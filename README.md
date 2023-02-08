# Melody Pretrain

## Commands

### 1 Prepare n-gram lexicon
```bash
python lexicon.py prepare \
--length 8 --top_p 0.02 \
--midi_dir experiment/dataset/lmd/midi \
--dataset_dir experiment/dataset/lmd

python lexicon.py render \
--dataset_dir experiment/dataset/lmd
```

### 2 Prepare dataset
(Keep tokenizer configs the same between pretrain and finetune stages.)
```bash
python prepare_data.py \
--midi_dir experiment/dataset/lmd/midi \
--dataset_dir experiment/dataset/lmd \
--granularity 64 --max_bar 128 --pitch_range 0 128
```

### 3 Pretrain
```bash
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_bar.yaml --trainer.devices "0,"

python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_span.yaml --trainer.devices "1,"

python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_ngram.yaml --trainer.devices "2,"

python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_ngram_multi.yaml --trainer.devices "3,"

python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_single.yaml --trainer.devices "1,"

python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_ngram_explicit.yaml --trainer.devices "1,2,3,4" --ckpt_path experiment/model-lmd/pretrain_ngram_explicit/lightning_logs/version_1/checkpoints/epoch=69-train_loss=0.413-val_loss=0.534-step=700.ckpt
```

### 4 Finetune
1. Causal lanugage modeling.
    ```bash
    # bar
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model-lmd/finetune_clm_bar --load_from_checkpoint experiment/model-lmd/pretrain_bar/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.086-val_loss=0.188-step=5000.ckpt --trainer.devices "1,"

    # span
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model-lmd/finetune_clm_span --load_from_checkpoint experiment/model-lmd/pretrain_span/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.269-val_loss=0.262-step=5000.ckpt --trainer.devices "2,"

    # ngram
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model-lmd/finetune_clm_ngram --load_from_checkpoint experiment/model-lmd/pretrain_ngram/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.043-val_loss=0.171-step=5000.ckpt --trainer.devices "3,"

    # ngram with multi-target
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model-lmd/finetune_clm_ngram_multi --load_from_checkpoint experiment/model-lmd/pretrain_ngram_multi/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.041-val_loss=0.125-step=5000.ckpt --trainer.devices "4,"

    # single span
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model-lmd/finetune_clm_single --load_from_checkpoint experiment/model-lmd/pretrain_single/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.129-val_loss=0.255-step=5000.ckpt --trainer.devices "5,"
    ```
2. Infilling.
    ```bash
    # bar
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model-lmd/finetune_infilling_bar --load_from_checkpoint experiment/model-lmd/pretrain_bar/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.086-val_loss=0.188-step=5000.ckpt --trainer.devices "1,"

    # span
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model-lmd/finetune_infilling_span --load_from_checkpoint experiment/model-lmd/pretrain_span/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.269-val_loss=0.262-step=5000.ckpt --trainer.devices "2,"

    # ngram
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model-lmd/finetune_infilling_ngram --load_from_checkpoint experiment/model-lmd/pretrain_ngram/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.043-val_loss=0.171-step=5000.ckpt --trainer.devices "3,"

    # ngram with multi-target
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model-lmd/finetune_infilling_ngram_multi --load_from_checkpoint experiment/model-lmd/pretrain_ngram_multi/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.041-val_loss=0.125-step=5000.ckpt --trainer.devices "4,"

    # single span
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model-lmd/finetune_infilling_single --load_from_checkpoint experiment/model-lmd/pretrain_single/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.129-val_loss=0.255-step=5000.ckpt --trainer.devices "6,"

### 5 Train from scratch
(for comparison)
```bash
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/from_scratch_clm.yaml --trainer.default_root_dir experiment/model-lmd/from_scratch_clm --trainer.devices "4,"

python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/from_scratch_infilling.yaml --trainer.default_root_dir experiment/model-lmd/from_scratch_infilling --trainer.devices "4,"
```

### 6 Test
1. Causal lanugage modeling.
    ```bash
    # bar
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model-lmd/finetune_clm_bar --ckpt_path experiment/model-lmd/finetune_clm_bar/lightning_logs/version_0/checkpoints/epoch=51-train_loss=0.667-val_loss=0.563.ckpt --trainer.devices "0,"

    # span
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model-lmd/finetune_clm_span --ckpt_path experiment/model-lmd/finetune_clm_span/lightning_logs/version_0/checkpoints/epoch=47-train_loss=0.609-val_loss=0.577.ckpt --trainer.devices "1,"

    # ngram
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model-lmd/finetune_clm_ngram --ckpt_path experiment/model-lmd/finetune_clm_ngram/lightning_logs/version_0/checkpoints/epoch=50-train_loss=0.595-val_loss=0.547.ckpt --trainer.devices "2,"

    # ngram with multi-target
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model-lmd/finetune_clm_ngram_multi --ckpt_path experiment/model-lmd/finetune_clm_ngram_multi/lightning_logs/version_0/checkpoints/epoch=48-train_loss=0.508-val_loss=0.511.ckpt --trainer.devices "3,"

    # from scratch
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model-lmd/from_scratch_clm --ckpt_path experiment/model-lmd/from_scratch_clm/lightning_logs/version_0/checkpoints/epoch=65-train_loss=0.446-val_loss=0.543.ckpt --trainer.devices "4,"
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
