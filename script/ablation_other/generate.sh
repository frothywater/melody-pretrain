# gpt
# python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_other/model/gpt/finetune_clm --ckpt_path experiment/ablation_other/model/gpt/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_other/generated/clm/gpt-best
# python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_other/model/gpt/finetune_infilling --ckpt_path experiment/ablation_other/model/gpt/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_other/generated/infilling/gpt-best
# from-scratch-clm
python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_other/model/from-scratch-clm/finetune_clm --ckpt_path experiment/ablation_other/model/from-scratch-clm/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_other/generated/clm/from-scratch-clm-best
# from-scratch-infilling
python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_other/model/from-scratch-infilling/finetune_infilling --ckpt_path experiment/ablation_other/model/from-scratch-infilling/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_other/generated/infilling/from-scratch-infilling-best

# gpt
# python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_other/model/gpt/finetune_clm --ckpt_path experiment/ablation_other/model/gpt/finetune_clm/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_other/generated/clm/gpt
# python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_other/model/gpt/finetune_infilling --ckpt_path experiment/ablation_other/model/gpt/finetune_infilling/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_other/generated/infilling/gpt
# from-scratch-clm
python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_other/model/from-scratch-clm/finetune_clm --ckpt_path experiment/ablation_other/model/from-scratch-clm/finetune_clm/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_other/generated/clm/from-scratch-clm
# from-scratch-infilling
python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_other/model/from-scratch-infilling/finetune_infilling --ckpt_path experiment/ablation_other/model/from-scratch-infilling/finetune_infilling/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_other/generated/infilling/from-scratch-infilling

python compute_metric.py --test_dir ../dataset/clm_test --generated_group_dir experiment/ablation_other/generated/clm --dest_path experiment/ablation_other/result/clm.csv
python compute_metric.py --test_dir ../dataset/infilling_test --generated_group_dir experiment/ablation_other/generated/infilling --dest_path experiment/ablation_other/result/infilling.csv --force_filename