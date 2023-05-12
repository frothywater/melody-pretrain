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
python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_length 12 --ngram_top_p 0.3 --add_segment_token \
--midi_dir ../dataset/melodynet_old --dataset_dir experiment/dataset/melodynet \
--pitch_ngram_dir experiment/dataset/melodynet/ngram/label_pitch \
--rhythm_ngram_dir experiment/dataset/melodynet/ngram/label_rhythm

python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_length 12 --ngram_top_p 0.3 --add_segment_token \
--midi_dir ../dataset/melodynet_old --dataset_dir experiment/dataset/melodynet_ngram_mixed \
--mixed_ngram_dir experiment/dataset/melodynet/ngram/label_mixed;\
python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_length 8 --ngram_top_p 0.3 --add_segment_token \
--midi_dir ../dataset/melodynet_old --dataset_dir experiment/dataset/melodynet_ngram_8 \
--pitch_ngram_dir experiment/dataset/melodynet/ngram/label_pitch \
--rhythm_ngram_dir experiment/dataset/melodynet/ngram/label_rhythm;\
python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_length 4 --ngram_top_p 0.3 --add_segment_token \
--midi_dir ../dataset/melodynet_old --dataset_dir experiment/dataset/melodynet_ngram_4 \
--pitch_ngram_dir experiment/dataset/melodynet/ngram/label_pitch \
--rhythm_ngram_dir experiment/dataset/melodynet/ngram/label_rhythm

python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --ngram_length 12 --ngram_top_p 0.3 \
--midi_dir ../dataset/melodynet_old --dataset_dir experiment/dataset/melodynet_noseg \
--pitch_ngram_dir experiment/dataset/melodynet/ngram/label_pitch \
--rhythm_ngram_dir experiment/dataset/melodynet/ngram/label_rhythm

python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --include_empty_bar --add_segment_token \
--midi_dir ../dataset/wikifonia --dataset_dir experiment/dataset/wikifonia
python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --include_empty_bar \
--midi_dir ../dataset/wikifonia --dataset_dir experiment/dataset/wikifonia_noseg
python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --include_empty_bar --add_segment_token \
--midi_dir experiment/dataset/infilling/midi --dataset_dir experiment/dataset/infilling
python prepare_data.py --kind octuple --granularity 64 --max_bar 128 --pitch_range 0 128 --include_empty_bar \
--midi_dir experiment/dataset/infilling/midi --dataset_dir experiment/dataset/infilling_noseg
```

### 3 Pretrain
```bash
script/ablation_infilling/pretrain.sh
script/ablation_other/pretrain.sh
script/ablation_ngram/pretrain.sh
script/base_model/pretrain.sh
```

### 4 Finetune
```bash
script/ablation_infilling/finetune.sh
script/ablation_other/finetune.sh
script/ablation_ngram/finetune.sh
script/base_model/finetune.sh

script/base_model/finetune.sh; script/base_model/generate.sh;\
script/ablation_other/finetune.sh; script/ablation_other/test.sh; script/ablation_other/generate.sh;\
script/ablation_ngram/finetune.sh; script/ablation_ngram/test.sh; script/ablation_ngram/generate.sh;\
script/ablation_infilling/finetune.sh; script/ablation_infilling/test.sh; script/ablation_infilling/generate.sh
```

### 5 Predict
```bash
script/ablation_infilling/generate.sh
script/ablation_other/generate.sh
script/ablation_ngram/generate.sh
script/base_model/generate.sh
```

### 6 Evaluate
```bash
script/ablation_infilling/test.sh
script/ablation_other/test.sh
script/ablation_ngram/test.sh

python plot_loss.py --experiment_dir experiment/ablation_infilling
python plot_loss.py --experiment_dir experiment/ablation_other
python plot_loss.py --experiment_dir experiment/ablation_ngram

python compute_metric.py --test_dir ../dataset/clm_test --generated_group_dir experiment/generated/cp --dest_path experiment/result/cp.csv
python compute_metric.py --test_dir ../dataset/clm_test --generated_group_dir experiment/generated/mt --dest_path experiment/result/mt.csv

python compute_metric.py --test_dir ../dataset/infilling_test --generated_dir experiment/generated/vli/infilling --dest_path experiment/result/vli.csv --force_filename
```

### 6 Deploy
```bash
python production/convert_checkpoint.py --checkpoint_path experiment/final/model/ngram-multi-single_60/finetune_clm/lightning_logs/version_0/checkpoints/epoch=9.ckpt --config_path experiment/final/model/ngram-multi-single_60/finetune_clm/lightning_logs/version_0/config.yaml --output_path experiment/production/melodyglm_base_completion.ckpt
python production/convert_checkpoint.py --checkpoint_path experiment/final/model/ngram-multi-single_60/finetune_inpainting/lightning_logs/version_0/checkpoints/epoch=9.ckpt --config_path experiment/final/model/ngram-multi-single_60/finetune_inpainting/lightning_logs/version_0/config.yaml --output_path experiment/production/melodyglm_base_inpainting.ckpt
```

## Dependencies

- `torch`
- `lightning`
- `jsonargparse[signatures]`
- `miditoolkit`
