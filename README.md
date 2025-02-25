# MelodyGLM: Multi-task Pre-training for Symbolic Melody Generation

[Paper](https://arxiv.org/abs/2309.10738)  
[Presentation Slides](asset/MelodyGLM.pdf)

## Abstract
Pre-trained language models have achieved impressive results in various music understanding and generation tasks. However, existing pre-training methods for symbolic melody generation struggle to capture multi-scale, multi-dimensional structural information in note sequences, due to the domain knowledge discrepancy between text and music. Moreover, the lack of available large-scale symbolic melody datasets limits the pre-training improvement.

In this paper, we propose MelodyGLM, a multi-task pre-training framework for generating melodies with long-term structure. We design the melodic n-gram and long span sampling strategies to create local and global blank infilling tasks for modeling the local and global structures in melodies.

Specifically, we incorporate pitch n-grams, rhythm n-grams, and their combined n-grams into the melodic n-gram blank infilling tasks for modeling the multi-dimensional structures in melodies. To this end, we have constructed a large-scale symbolic melody dataset, MelodyNet, containing more than 0.4 million melody pieces. MelodyNet is utilized for large-scale pre-training and domain-specific n-gram lexicon construction.

Both subjective and objective evaluations demonstrate that MelodyGLM surpasses the standard and previous pre-training methods. In particular, subjective evaluations show that, on the melody continuation task, MelodyGLM gains average improvements of 0.82, 0.87, 0.78, and 0.94 in consistency, rhythmicity, structure, and overall quality, respectively. Notably, MelodyGLM nearly matches the quality of human-composed melodies on the melody inpainting task.

## Demo
Melody completion on *Ode to Joy* (a little bit swing):

https://github.com/user-attachments/assets/48bca977-e17c-4376-9c33-1fa279d65dc2

Melody inpainting on *Jasmine* for the 4 bars in the middle:

https://github.com/user-attachments/assets/c46bbfb6-c2d7-4080-9c6c-bde439eebe34

## Structure
- `main.py`: Starting point for training, testing and inference, defining Lightning CLI.
- `lexicon.py`: Script to extract N-grams, build lexicon and annotate the MIDI files.
- `prepare_data.py`: Script to compile NumPy formatted dataset from MIDI files and their extracted N-grams.
- `compute_metric.py`: Script to compute metrics from generated pieces and ground truth pieces.
- `melody_pretrain`
  - `dataset.py`: Datasets and dataloaders with various custom data collators to implement different input/target format and masking methods for ablation experiments.
  - `model.py`: The core model, with several subclasses for different purposes, such as pre-training, testing, completion and infilling.
  - `module.py`: Custom modules such as compound token fuser and positional encoding.
  - `ngram.py`: N-gram-related codes to extract N-grams, calculate frequency, score, ranking, build the lexicon and annotate the MIDI files.
  - `task.py`: Use the task abstraction to setup both input/target format and masking method combination for given experiment config.
  - `tokenizer.py`: Different flavors of MIDI file tokenizers, including `MIDITokenizer` [(Huang et al., 2019)](https://research.google/pubs/music-transformer-generating-music-with-long-term-structure/), `RemiTokenizer` [(Huang and Yang, 2020)](https://arxiv.org/abs/2002.00212), `CPTokenizer` [(Hsiao et al., 2021)](https://arxiv.org/abs/2101.02402), and `OctupleTokenizer` [(Zeng et al., 2021)](https://arxiv.org/abs/2106.05630).
- `metric`: Codes related to evaluation.
- `config`: All model config files for different experiment groups.
- `script`: Scripts for pre-training, fine-tuning and generation for experiments.
- `production`: Contains script to convert Lightning checkpoint to deployable PyTorch checkpoint, the model codes and example codes for inference.
- And other codes for drawing figures, statistics, etc.

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
