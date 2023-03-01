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
--midi_dir experiment/dataset/pretrain_base/midi \
--dataset_dir experiment/dataset/pretrain_base \
--granularity 64 --max_bar 128 --pitch_range 0 128

python prepare_data.py \
--midi_dir /mnt/nextlab/xinda/MDP/data/process/paper_mlm_v20230225_dedup_split/finetune_wikifonia \
--dataset_dir experiment/dataset/wikifonia \
--granularity 64 --max_bar 128 --pitch_range 0 128
```

### 3 Pretrain
```bash
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/span.yaml --trainer.devices "0,"

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/bar.yaml --trainer.devices "1,"

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/ngram.yaml --trainer.devices "2,"

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/single.yaml --trainer.devices "0,"

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_advanced/ngram.yaml --trainer.devices "1,"

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_advanced/ngram_plus.yaml --trainer.devices "2,"

python main.py fit --config config/model/base.yaml --config config/pretrain_advanced/ngram_explicit.yaml --trainer.devices "2,3"
```

### 4 Finetune
1. Causal lanugage modeling.
    ```bash
    # span
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_span --load_from_checkpoint experiment/model/pretrain_span/lightning_logs/version_0/checkpoints/epoch=19-train_loss=0.110-val_loss=0.000-step=5000.ckpt --trainer.devices "0,"

    # bar
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_bar --load_from_checkpoint experiment/model/pretrain_bar/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.086-val_loss=0.188-step=5000.ckpt --trainer.devices "1,"

    # ngram
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_random --load_from_checkpoint experiment/model/pretrain_ngram_random/lightning_logs/version_0/checkpoints/epoch=19-train_loss=0.128-val_loss=0.000-step=5000.ckpt --trainer.devices "1,"

    # ngram with multi-target
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram --load_from_checkpoint experiment/model/pretrain_ngram/lightning_logs/version_0/checkpoints/epoch=19-train_loss=0.099-val_loss=0.000-step=5000.ckpt --trainer.devices "2,"

    # ngram with multi-target plus
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_plus --load_from_checkpoint experiment/model/pretrain_ngram_plus/lightning_logs/version_0/checkpoints/epoch=19-train_loss=0.154-val_loss=0.000-step=5000.ckpt --trainer.devices "3,"

    # single span
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_single --load_from_checkpoint experiment/model/pretrain_single/lightning_logs/version_0/checkpoints/epoch=19-train_loss=0.104-val_loss=0.000-step=5000.ckpt --trainer.devices "4,"

    # explicit
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_explicit --load_from_checkpoint experiment/model/pretrain_ngram_explicit/lightning_logs/version_1/checkpoints/epoch=135-train_loss=0.183-val_loss=0.143-step=5000.ckpt/fp32.ckpt --trainer.devices "2,"
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
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-lmd-advanced/finetune_infilling_ngram_multi --load_from_checkpoint experiment/model-lmd-advanced/pretrain_ngram_multi/lightning_logs/version_1/checkpoints/epoch=131-train_loss=0.036-val_loss=0.165-step=5000.ckpt --trainer.devices "3,"

    # ngram with multi-target plus
    python main.py fit --config config/trainer.yaml --config config/model_base.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model-lmd/finetune_infilling_ngram_multi_plus --load_from_checkpoint experiment/model-lmd/pretrain_ngram_multi_plus/lightning_logs/version_1/checkpoints/epoch=131-train_loss=0.112-val_loss=0.183-step=5000.ckpt --trainer.devices "3,"

    # single span
    python main.py fit --config config/trainer.yaml --config config/model_base.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model-lmd/finetune_infilling_single --load_from_checkpoint experiment/model-lmd/pretrain_single/lightning_logs/version_0/checkpoints/epoch=131-train_loss=0.092-val_loss=0.254-step=5000.ckpt --trainer.devices "3,"

    # explicit
    python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-lmd-advanced/finetune_infilling_explicit --load_from_checkpoint experiment/model-lmd-advanced/pretrain_ngram_explicit/lightning_logs/version_1/checkpoints/epoch=135-train_loss=0.183-val_loss=0.143-step=5000.ckpt/fp32.ckpt --trainer.devices "4,"
    ```

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
    python main.py predict --config config/model/base.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/model-lmd-advanced/finetune_clm_ngram_multi --ckpt_path experiment/model-lmd-advanced/finetune_clm_ngram_multi/lightning_logs/version_1/checkpoints/epoch=222-train_loss=0.334-val_loss=0.557-step=2000.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_ngram --trainer.devices "0,"

    # single
    python main.py predict --config config/model_base.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_single --ckpt_path experiment/model-lmd/finetune_clm_single/lightning_logs/version_0/checkpoints/epoch=222-train_loss=0.266-val_loss=0.630-step=2000.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_single --trainer.devices "3,"

    # explicit
    python main.py predict --config config/model/base.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/model-lmd-advanced/finetune_clm_explicit --ckpt_path experiment/model-lmd-advanced/finetune_clm_explicit/lightning_logs/version_1/checkpoints/epoch=222-train_loss=0.348-val_loss=0.547-step=2000.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_explicit --trainer.devices "0,"

    # from scratch
    python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/model-lmd-masking/from_scratch_clm --ckpt_path experiment/model-lmd-masking/from_scratch_clm/lightning_logs/version_0/checkpoints/epoch=222-train_loss=0.247-val_loss=0.764-step=2000.ckpt --trainer.callbacks.output_dir experiment/output/from_scratch_clm --trainer.devices "1,"
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
    python main.py predict --config config/model/base.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/model-lmd-advanced/finetune_infilling_ngram_multi --ckpt_path experiment/model-lmd-advanced/finetune_infilling_ngram_multi/lightning_logs/version_1/checkpoints/epoch=285-train_loss=0.141-val_loss=0.181-step=2000.ckpt --trainer.callbacks.output_dir experiment/output/finetune_infilling_ngram --trainer.devices "0,"

    # single
    python main.py predict --config config/model/base.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_single --ckpt_path experiment/model-lmd/finetune_infilling_single/lightning_logs/version_0/checkpoints/epoch=285-train_loss=0.112-val_loss=0.184-step=2000.ckpt --trainer.callbacks.output_dir experiment/output/finetune_infilling_single --trainer.devices "3,"

    # explicit
    python main.py predict --config config/model/base.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/model-lmd-advanced/finetune_infilling_explicit --ckpt_path experiment/model-lmd-advanced/finetune_infilling_explicit/lightning_logs/version_1/checkpoints/epoch=285-train_loss=0.140-val_loss=0.183-step=2000.ckpt --trainer.callbacks.output_dir experiment/output/finetune_infilling_explicit --trainer.devices "0,"

    # from scratch
    python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/model-lmd-masking/from_scratch_infilling --ckpt_path experiment/model-lmd-masking/from_scratch_infilling/lightning_logs/version_0/checkpoints/epoch=399-train_loss=0.168-val_loss=0.263-step=2000.ckpt --trainer.callbacks.output_dir experiment/output/from_scratch_infilling --trainer.devices "1,"
    ```

## Dependencies

- `torch`
- `lightning`
- `jsonargparse[signatures]`
- `miditoolkit`
