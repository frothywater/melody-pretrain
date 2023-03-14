# Melody Pretrain

## Commands

### 1 Prepare n-gram lexicon
```bash
python lexicon.py prepare \
--length 8 --top_p 0.2 \
--midi_dir ../dataset/melodynet \
--dataset_dir experiment/dataset/melodynet

python lexicon.py prepare \
--length 8 --top_p 0.2 \
--midi_dir ../dataset/lmd \
--dataset_dir experiment/dataset/lmd

python lexicon.py render \
--dataset_dir experiment/dataset/melodynet
```

### 2 Prepare dataset
(Keep tokenizer configs the same between pretrain and finetune stages.)
```bash
python prepare_data.py \
--midi_dir ../dataset/melodynet \
--dataset_dir experiment/dataset/melodynet \
--granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_label \
--skeleton_info_path skeleton_note_indices.npz

python prepare_data.py \
--midi_dir ../dataset/lmd \
--dataset_dir experiment/dataset/lmd \
--granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_label

python prepare_data.py \
--midi_dir ../dataset/wikifonia \
--dataset_dir experiment/dataset/wikifonia \
--granularity 64 --max_bar 128 --pitch_range 0 128 --include_empty_bar
```

### 3 Pretrain
```bash
python generate_script.py --dataset_dir experiment/dataset/melodynet \
--experiment_dir experiment/ablation_infilling
python generate_script.py --dataset_dir experiment/dataset/melodynet \
--experiment_dir experiment/ablation_recovery

experiment/ablation_infilling/script/run.sh
experiment/ablation_recovery/script/run.sh

experiment/ablation_infilling/script/generate.sh
experiment/ablation_recovery/script/generate.sh

python plot_loss.py --experiment_dir experiment/ablation_infilling
python plot_loss.py --experiment_dir experiment/ablation_recovery
python compute_metric.py --experiment_dir experiment/ablation_infilling --dataset_dir experiment/dataset/wikifonia
python compute_metric.py --experiment_dir experiment/ablation_recovery --dataset_dir experiment/dataset/wikifonia
python plot_metric.py --experiment_dir experiment/ablation_infilling
python plot_metric.py --experiment_dir experiment/ablation_recovery
```

## Dependencies

- `torch`
- `lightning`
- `jsonargparse[signatures]`
- `miditoolkit`
