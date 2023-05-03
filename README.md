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
python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_length 12 --ngram_top_p 0.3 \
--midi_dir ../dataset/melodynet_old --dataset_dir experiment/dataset/melodynet \
--pitch_ngram_dir experiment/dataset/melodynet/ngram/label_pitch \
--rhythm_ngram_dir experiment/dataset/melodynet/ngram/label_rhythm

python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_length 12 --ngram_top_p 0.3 \
--midi_dir ../dataset/melodynet_old --dataset_dir experiment/dataset/melodynet_ngram_mixed \
--mixed_ngram_dir experiment/dataset/melodynet/ngram/label_mixed;\
python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_length 8 --ngram_top_p 0.3 \
--midi_dir ../dataset/melodynet_old --dataset_dir experiment/dataset/melodynet_ngram_8 \
--pitch_ngram_dir experiment/dataset/melodynet/ngram/label_pitch \
--rhythm_ngram_dir experiment/dataset/melodynet/ngram/label_rhythm;\
python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_length 4 --ngram_top_p 0.3 \
--midi_dir ../dataset/melodynet_old --dataset_dir experiment/dataset/melodynet_ngram_4 \
--pitch_ngram_dir experiment/dataset/melodynet/ngram/label_pitch \
--rhythm_ngram_dir experiment/dataset/melodynet/ngram/label_rhythm

python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --include_empty_bar \
--midi_dir ../dataset/wikifonia --dataset_dir experiment/dataset/wikifonia

python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --include_empty_bar \
--midi_dir experiment/dataset/infilling_comparison/midi --dataset_dir experiment/dataset/infilling_comparison
```

### 3 Pretrain
```bash
python generate_script.py --dataset_dir experiment/dataset/melodynet --experiment_dir experiment/ablation_infilling
python generate_script.py --dataset_dir experiment/dataset/melodynet --experiment_dir experiment/ablation_recovery
python generate_script.py --dataset_dir experiment/dataset/melodynet --experiment_dir experiment/final

# experiment/ablation_recovery/script/run.sh
experiment/ablation_infilling/script/run.sh
experiment/ablation_other/script/run.sh
experiment/ablation_ngram/script/run.sh
experiment/final/script/run.sh

experiment/final/script/generate.sh

# python plot_loss.py --experiment_dir experiment/ablation_recovery
python plot_loss.py --experiment_dir experiment/ablation_infilling
python plot_loss.py --experiment_dir experiment/ablation_other
python plot_loss.py --experiment_dir experiment/ablation_ngram

python crop_midi.py --src_dir experiment/dataset/infilling_comparison/midi/test --dest_dir experiment/dataset/infilling_comparison/midi/test_4bar --starting_bar 6 --num_bars 4
python crop_midi.py --src_dir experiment/final/generated/infilling --dest_dir experiment/final/generated/infilling_4bar --starting_bar 6 --num_bars 4

python compute_metric.py --test_dir ../dataset/clm_test --generated_dir experiment/final/generated/clm_epoch10 --dest_path experiment/final/result/clm_epoch10.csv
python compute_metric.py --test_dir ../dataset/clm_test --generated_dir experiment/final/generated/clm_epoch20 --dest_path experiment/final/result/clm_epoch20.csv
python compute_metric.py --test_dir ../dataset/infilling_test --generated_dir experiment/final/generated/infilling_epoch15 --dest_path experiment/final/result/infilling_epoch15.csv --force_filename
python compute_metric.py --test_dir ../dataset/infilling_test --generated_dir experiment/final/generated/infilling_epoch30 --dest_path experiment/final/result/infilling_epoch30.csv --force_filename

python compute_metric.py --test_dir ../dataset/clm_test --generated_dir ../music-transformer/experiment/generated --dest_path experiment/result/music_transformer.csv
```

### 4 Deploy
```bash
python production/convert_checkpoint.py --checkpoint_path experiment/ablation_infilling/model/ngram-multi-single_60/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt --config_path experiment/ablation_infilling/model/ngram-multi-single_60/finetune_clm/lightning_logs/version_0/config.yaml --output_path experiment/production/melodyglm_finetuned_completion_small.ckpt
```

## Dependencies

- `torch`
- `lightning`
- `jsonargparse[signatures]`
- `miditoolkit`
