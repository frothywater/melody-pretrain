# ngram-mixed
python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-mixed/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-mixed/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt
python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-mixed/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-mixed/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt

# ngram-8
python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-8/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-8/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt
python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-8/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-8/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt

# ngram-4
python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-4/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-4/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt
python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-4/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-4/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt

# ngram-pitch
python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-pitch/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-pitch/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt
python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-pitch/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-pitch/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt

# # ngram-rhythm
# python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-rhythm/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-rhythm/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt
# python main.py test --config config/trainer.yaml --config config/model/small.yaml --config config/predict/test_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-rhythm/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-rhythm/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt