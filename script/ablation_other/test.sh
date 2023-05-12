# gpt
# python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_clm.yaml --trainer.default_root_dir experiment/ablation_other/model/gpt/finetune_clm --ckpt_path experiment/ablation_other/model/gpt/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt
# python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_infilling.yaml --trainer.default_root_dir experiment/ablation_other/model/gpt/finetune_infilling --ckpt_path experiment/ablation_other/model/gpt/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt

# from-scratch-clm
python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_clm.yaml --trainer.default_root_dir experiment/ablation_other/model/from-scratch-clm/finetune_clm --ckpt_path experiment/ablation_other/model/from-scratch-clm/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt

# from-scratch-infilling
python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_infilling.yaml --trainer.default_root_dir experiment/ablation_other/model/from-scratch-infilling/finetune_infilling --ckpt_path experiment/ablation_other/model/from-scratch-infilling/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt