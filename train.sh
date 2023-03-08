python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_advanced/ngram.yaml --trainer.default_root_dir experiment/model/pretrain_ngram_multi

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/ngram.yaml --trainer.default_root_dir experiment/model/pretrain_ngram

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/bar.yaml --trainer.default_root_dir experiment/model/pretrain_bar

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/span.yaml --trainer.default_root_dir experiment/model/pretrain_span

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/single.yaml --trainer.default_root_dir experiment/model/pretrain_single

# span
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_span --load_from_checkpoint experiment/model/pretrain_span/lightning_logs/version_0/checkpoints/step=5000.ckpt

# bar
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_bar --load_from_checkpoint experiment/model/pretrain_bar/lightning_logs/version_0/checkpoints/step=5000.ckpt

# ngram
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram --load_from_checkpoint experiment/model/pretrain_ngram/lightning_logs/version_0/checkpoints/step=5000.ckpt

# single span
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_single --load_from_checkpoint experiment/model/pretrain_single/lightning_logs/version_0/checkpoints/step=5000.ckpt

# ngram with multi-target
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_multi --load_from_checkpoint experiment/model/pretrain_ngram_multi/lightning_logs/version_0/checkpoints/step=5000.ckpt

# ngram with multi-target plus
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model/finetune_clm_ngram_plus --load_from_checkpoint experiment/model/pretrain_ngram_plus/lightning_logs/version_0/checkpoints/step=5000.ckpt

# span
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_span --load_from_checkpoint experiment/model/pretrain_span/lightning_logs/version_0/checkpoints/step=5000.ckpt

# bar
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_bar --load_from_checkpoint experiment/model/pretrain_bar/lightning_logs/version_0/checkpoints/step=5000.ckpt

# ngram
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram --load_from_checkpoint experiment/model/pretrain_ngram/lightning_logs/version_0/checkpoints/step=5000.ckpt

# single span
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_single --load_from_checkpoint experiment/model/pretrain_single/lightning_logs/version_0/checkpoints/step=5000.ckpt

# ngram with multi-target
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram_multi --load_from_checkpoint experiment/model/pretrain_ngram_multi/lightning_logs/version_0/checkpoints/step=5000.ckpt

# ngram with multi-target plus
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model/finetune_infilling_ngram_plus --load_from_checkpoint experiment/model/pretrain_ngram_plus/lightning_logs/version_0/checkpoints/step=5000.ckpt
