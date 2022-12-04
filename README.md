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

python prepare_data.py \
--midi_dir ../midi-preprocess/data/wikifonia \
--dataset_dir experiment/dataset/wikifonia \
--granularity 64 --max_bar 128 --pitch_range 0 128

python pretrain.py \
--dataset_dir experiment/dataset/pretrain_small --load_ngram_data True \
--default_root_dir experiment/model/pretrain_small \
--model_dim 512 --feedforward_dim 2048 --num_layers 8 --num_heads 8 \
--max_epochs 100 --batch_size 16 --seq_len 512 \
--lr 4e-4 --warmup_percent 0.1 \
--accumulate_grad_batches 8 --gradient_clip_val 1.0 --patience 10 \
--accelerator gpu --devices 2 --num_workers 4 --precision 16
# --fast_dev_run 100 --profiler simple

python finetune_clm.py \
--dataset_dir experiment/dataset/wikifonia \
--default_root_dir experiment/model/clm_from_scratch \
--model_dim 512 --feedforward_dim 2048 --num_layers 8 --num_heads 8 \
--max_epochs 100 --batch_size 32 --seq_len 512 \
--lr 4e-4 --warmup_percent 0.1 \
--accumulate_grad_batches 4 --gradient_clip_val 1.0 --patience 10 \
--accelerator gpu --devices 1 --num_workers 4 --precision 16

python finetune_clm.py \
--checkpoint_path 'experiment/model/pretrain_small/lightning_logs/version_0/checkpoints/epoch=37-step=798.ckpt' \
--dataset_dir experiment/dataset/wikifonia \
--default_root_dir experiment/model/finetune_clm \
--model_dim 512 --feedforward_dim 2048 --num_layers 8 --num_heads 8 \
--max_epochs 100 --batch_size 32 --seq_len 512 \
--lr 4e-4 --warmup_percent 0.1 \
--accumulate_grad_batches 4 --gradient_clip_val 1.0 --patience 10 \
--accelerator gpu --devices "1," --num_workers 4 --precision 16
```
