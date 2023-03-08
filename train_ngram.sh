# 40
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/ngram.yaml --trainer.default_root_dir experiment/model-ngram/pretrain_ngram40 --model.dataset_dir experiment/dataset/pretrain_base_ngram40 --data.dataset_dir experiment/dataset/pretrain_base_ngram40

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model-ngram/finetune_clm_ngram40 --load_from_checkpoint experiment/model-ngram/pretrain_ngram40/lightning_logs/version_0/checkpoints/step=3000.ckpt

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-ngram/finetune_infilling_ngram40 --load_from_checkpoint experiment/model-ngram/pretrain_ngram40/lightning_logs/version_0/checkpoints/step=3000.ckpt

# 30
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/ngram.yaml --trainer.default_root_dir experiment/model-ngram/pretrain_ngram30 --model.dataset_dir experiment/dataset/pretrain_base_ngram30 --data.dataset_dir experiment/dataset/pretrain_base_ngram30

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model-ngram/finetune_clm_ngram30 --load_from_checkpoint experiment/model-ngram/pretrain_ngram30/lightning_logs/version_0/checkpoints/step=3000.ckpt

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-ngram/finetune_infilling_ngram30 --load_from_checkpoint experiment/model-ngram/pretrain_ngram30/lightning_logs/version_0/checkpoints/step=3000.ckpt

# 20
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_masking/ngram.yaml --trainer.default_root_dir experiment/model-ngram/pretrain_ngram20 --model.dataset_dir experiment/dataset/pretrain_base_ngram20 --data.dataset_dir experiment/dataset/pretrain_base_ngram20

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model-ngram/finetune_clm_ngram20 --load_from_checkpoint experiment/model-ngram/pretrain_ngram20/lightning_logs/version_0/checkpoints/step=3000.ckpt

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-ngram/finetune_infilling_ngram20 --load_from_checkpoint experiment/model-ngram/pretrain_ngram20/lightning_logs/version_0/checkpoints/step=3000.ckpt
