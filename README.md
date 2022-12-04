# Melody Pretrain

```bash
python lexicon.py prepare \
--length 8 --top_p 0.1 \
--midi_dir ../midi-preprocess/data/done \
--dataset_dir experiment/dataset/pretrain_small

python lexicon.py render \
--dataset_dir experiment/dataset/pretrain_small

python prepare_data.py \
--midi_dir ../midi-preprocess/data/done \
--dataset_dir experiment/dataset/pretrain_small \
--granularity 64 --max_bar 128 --pitch_range 0 128

python pretrain.py \
--dataset_dir experiment/dataset/pretrain_small \
--default_root_dir experiment/model/pretrain_small \
--max_epochs 200 --batch_size 16 --seq_len 512 --accumulate_grad_batches 8 \
--accelerator gpu --devices 1
```
