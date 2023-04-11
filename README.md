# Melody Pretrain

## Commands

### 1 Prepare n-gram lexicon
```bash
python lexicon.py extract --length 12 --ngram_kind pitch_class --midi_dir ../dataset/melodynet --ngram_dir experiment/dataset/melodynet/ngram/ngram_pitch
python lexicon.py extract --length 12 --ngram_kind bar_onset --midi_dir ../dataset/melodynet --ngram_dir experiment/dataset/melodynet/ngram/ngram_rhythm

python lexicon.py build --length 12 --ngram_kind pitch_class --ngram_dir experiment/dataset/melodynet/ngram/ngram_pitch --lexicon_path experiment/dataset/melodynet/ngram/lexicon_pitch.pkl
python lexicon.py build --length 12 --ngram_kind bar_onset --ngram_dir experiment/dataset/melodynet/ngram/ngram_rhythm --lexicon_path experiment/dataset/melodynet/ngram/lexicon_rhythm.pkl

python lexicon.py prepare --length 12 --ngram_kind pitch_class --ngram_dir experiment/dataset/melodynet/ngram/ngram_pitch --lexicon_path experiment/dataset/melodynet/ngram/lexicon_pitch.pkl --label_dir experiment/dataset/melodynet/ngram/label_pitch
python lexicon.py prepare --length 12 --ngram_kind bar_onset --ngram_dir experiment/dataset/melodynet/ngram/ngram_rhythm --lexicon_path experiment/dataset/melodynet/ngram/lexicon_rhythm.pkl --label_dir experiment/dataset/melodynet/ngram/label_rhythm
```

### 2 Prepare dataset
(Keep tokenizer configs the same between pretrain and finetune stages.)
```bash
python prepare_data.py --granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_length 12 --ngram_top_p 0.3 \
--midi_dir ../dataset/melodynet --dataset_dir experiment/dataset/melodynet \
--pitch_ngram_dir experiment/dataset/melodynet/ngram/label_pitch \
--rhythm_ngram_dir experiment/dataset/melodynet/ngram/label_rhythm

python prepare_data.py --granularity 64 --max_bar 128 --pitch_range 0 128 --include_empty_bar \
--midi_dir ../dataset/wikifonia --dataset_dir experiment/dataset/wikifonia
```

### 3 Pretrain
```bash
python generate_script.py --dataset_dir experiment/dataset/melodynet --experiment_dir experiment/ablation_mask
experiment/ablation_mask/script/run.sh

experiment/ablation_mask/script/generate.sh
python plot_loss.py --experiment_dir experiment/ablation_mask_step
python compute_metric.py --experiment_dir experiment/ablation_mask --dataset_dir experiment/dataset/wikifonia
python plot_metric.py --experiment_dir experiment/ablation_mask
```

## Dependencies

- `torch`
- `lightning`
- `jsonargparse[signatures]`
- `miditoolkit`
