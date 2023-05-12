# gpt
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/ablation_other/model/gpt/finetune_clm --load_from_checkpoint experiment/ablation_other/model/gpt/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/ablation_other/model/gpt/finetune_infilling --load_from_checkpoint experiment/ablation_other/model/gpt/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt

# from-scratch-clm
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/clm.yaml --config config/ablation_other/from-scratch.yaml --trainer.default_root_dir experiment/ablation_other/model/from-scratch-clm/finetune_clm

# from-scratch-infilling
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/infilling.yaml --config config/ablation_other/from-scratch.yaml --trainer.default_root_dir experiment/ablation_other/model/from-scratch-infilling/finetune_infilling