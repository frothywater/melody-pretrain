# ngram-mixed
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-mixed/finetune_clm --load_from_checkpoint experiment/ablation_ngram/model/ngram-mixed/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-mixed/finetune_infilling --load_from_checkpoint experiment/ablation_ngram/model/ngram-mixed/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt

# ngram-8
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-8/finetune_clm --load_from_checkpoint experiment/ablation_ngram/model/ngram-8/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-8/finetune_infilling --load_from_checkpoint experiment/ablation_ngram/model/ngram-8/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt

# ngram-4
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-4/finetune_clm --load_from_checkpoint experiment/ablation_ngram/model/ngram-4/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-4/finetune_infilling --load_from_checkpoint experiment/ablation_ngram/model/ngram-4/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt

# ngram-pitch
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-pitch/finetune_clm --load_from_checkpoint experiment/ablation_ngram/model/ngram-pitch/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-pitch/finetune_infilling --load_from_checkpoint experiment/ablation_ngram/model/ngram-pitch/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt

# # ngram-rhythm
# python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-rhythm/finetune_clm --load_from_checkpoint experiment/ablation_ngram/model/ngram-rhythm/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt
# python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-rhythm/finetune_infilling --load_from_checkpoint experiment/ablation_ngram/model/ngram-rhythm/pretrain/lightning_logs/version_0/checkpoints/step=100000.ckpt