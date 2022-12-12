# Melody Pretrain

## 1 Prepare n-gram lexicon
```bash
python lexicon.py prepare \
--length 8 --top_p 0.1 \
--midi_dir ../midi-preprocess/data/done \
--dataset_dir experiment/dataset/pretrain_small

python lexicon.py render \
--dataset_dir experiment/dataset/pretrain_small
```

## 2 Prepare dataset
(Keep tokenizer configs the same between pretrain and finetune stages.)
```bash
python prepare_data.py \
--midi_dir ../midi-preprocess/data/done \
--dataset_dir experiment/dataset/pretrain_small \
--granularity 64 --max_bar 128 --pitch_range 0 128
```

## 3 Pretrain
```bash
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_span.yaml --trainer.devices "0,"
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_bar.yaml --trainer.devices "1,"
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_ngram.yaml --trainer.devices "2,"
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/pretrain_ngram_multi.yaml --trainer.devices "3,"
```

## 4 Finetune
1. Causal lanugage modeling.
    ```bash
    # bar
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_bar --load_from_checkpoint experiment/model/pretrain_bar/lightning_logs/version_0/checkpoints/epoch=105-step=8798.ckpt --trainer.devices "0,"

    # span
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_span --load_from_checkpoint experiment/model/pretrain_span/lightning_logs/version_0/checkpoints/epoch=169-step=14110.ckpt --trainer.devices "1,"

    # ngram
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram --load_from_checkpoint experiment/model/pretrain_ngram/lightning_logs/version_1/checkpoints/epoch=87-step=7304.ckpt --trainer.devices "2,"

    # ngram with multi-target
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_multi --load_from_checkpoint experiment/model/pretrain_ngram_multi/lightning_logs/version_0/checkpoints/epoch=128-step=10707.ckpt --trainer.devices "3,"
    ```
2. Infilling.
    ```bash
    # bar
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_bar --load_from_checkpoint experiment/model/pretrain_bar/lightning_logs/version_0/checkpoints/epoch=105-step=8798.ckpt --trainer.devices "0,"

    # span
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_span --load_from_checkpoint experiment/model/pretrain_span/lightning_logs/version_0/checkpoints/epoch=169-step=14110.ckpt --trainer.devices "1,"

    # ngram
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram --load_from_checkpoint experiment/model/pretrain_ngram/lightning_logs/version_1/checkpoints/epoch=87-step=7304.ckpt --trainer.devices "2,"

    # ngram with multi-target
    python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram_multi --load_from_checkpoint experiment/model/pretrain_ngram_multi/lightning_logs/version_0/checkpoints/epoch=128-step=10707.ckpt --trainer.devices "3,"

## 5 Train from scratch
(for comparison)
```bash
python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_clm.yaml --trainer.default_root_dir experiment/model/from_scratch_clm --trainer.devices "4,"

python main.py fit --config config/trainer.yaml --config config/model_small.yaml --config config/finetune_infilling.yaml --trainer.default_root_dir experiment/model/from_scratch_infilling --trainer.devices "5,"
```

## 6 Test
1. Causal lanugage modeling.
    ```bash
    # bar
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_bar --ckpt_path experiment/model/finetune_clm_bar/lightning_logs/version_0/checkpoints/epoch=18-step=950.ckpt --trainer.devices "0,"

    # span
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_span --ckpt_path experiment/model/finetune_clm_span/lightning_logs/version_0/checkpoints/epoch=14-val_loss=0.538.ckpt --trainer.devices "1,"

    # ngram
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram --ckpt_path experiment/model/finetune_clm_ngram/lightning_logs/version_0/checkpoints/epoch=18-val_loss=0.540.ckpt --trainer.devices "2,"

    # ngram with multi-target
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_multi --ckpt_path experiment/model/finetune_clm_ngram_multi/lightning_logs/version_0/checkpoints/epoch=16-step=850.ckpt --trainer.devices "3,"

    # from scratch
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_clm.yaml --trainer.default_root_dir experiment/model/from_scratch_clm --ckpt_path experiment/model/from_scratch_clm/lightning_logs/version_0/checkpoints/epoch=31-step=1600.ckpt --trainer.devices "4,"
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
    python main.py test --config config/trainer.yaml --config config/model_small.yaml --config config/test_infilling.yaml --trainer.default_root_dir experiment/model/from_scratch_infilling --ckpt_path experiment/model/from_scratch_infilling/lightning_logs/version_0/checkpoints/epoch=86-val_loss=0.297.ckpt --trainer.devices "4,"
    ```

## 7 Inference
1. Causal lanugage modeling.
    ```bash
    # bar
    python main.py predict --config config/model_small.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_bar --ckpt_path experiment/model/finetune_clm_bar/lightning_logs/version_0/checkpoints/epoch=18-step=950.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_bar --trainer.devices "0,"

    # span
    python main.py predict --config config/model_small.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_span --ckpt_path experiment/model/finetune_clm_span/lightning_logs/version_0/checkpoints/epoch=14-val_loss=0.538.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_span --trainer.devices "1,"

    # ngram
    python main.py predict --config config/model_small.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram --ckpt_path experiment/model/finetune_clm_ngram/lightning_logs/version_0/checkpoints/epoch=18-val_loss=0.540.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_ngram --trainer.devices "2,"

    # ngram with multi-target
    python main.py predict --config config/model_small.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_multi --ckpt_path experiment/model/finetune_clm_ngram_multi/lightning_logs/version_0/checkpoints/epoch=16-step=850.ckpt --trainer.callbacks.output_dir experiment/output/finetune_clm_ngram_multi --trainer.devices "3,"

    # from scratch
    python main.py predict --config config/model_small.yaml --config config/generate_clm.yaml --trainer.default_root_dir experiment/model/from_scratch_clm --ckpt_path experiment/model/from_scratch_clm/lightning_logs/version_0/checkpoints/epoch=31-step=1600.ckpt --trainer.callbacks.output_dir experiment/output/from_scratch_clm --trainer.devices "4,"
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
