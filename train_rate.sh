# 30
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_rate/span_30.yaml --trainer.default_root_dir experiment/model-rate/pretrain_span_30

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model-rate/finetune_clm_span_30 --load_from_checkpoint experiment/model-rate/pretrain_span_30/lightning_logs/version_0/checkpoints/step=3000.ckpt

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-rate/finetune_infilling_span_30 --load_from_checkpoint experiment/model-rate/pretrain_span_30/lightning_logs/version_0/checkpoints/step=3000.ckpt

# 40
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_rate/span_40.yaml --trainer.default_root_dir experiment/model-rate/pretrain_span_40

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model-rate/finetune_clm_span_40 --load_from_checkpoint experiment/model-rate/pretrain_span_40/lightning_logs/version_0/checkpoints/step=3000.ckpt

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-rate/finetune_infilling_span_40 --load_from_checkpoint experiment/model-rate/pretrain_span_40/lightning_logs/version_0/checkpoints/step=3000.ckpt

# 50
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_rate/span_50.yaml --trainer.default_root_dir experiment/model-rate/pretrain_span_50

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model-rate/finetune_clm_span_50 --load_from_checkpoint experiment/model-rate/pretrain_span_50/lightning_logs/version_0/checkpoints/step=3000.ckpt

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-rate/finetune_infilling_span_50 --load_from_checkpoint experiment/model-rate/pretrain_span_50/lightning_logs/version_0/checkpoints/step=3000.ckpt

# 60
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_rate/span_60.yaml --trainer.default_root_dir experiment/model-rate/pretrain_span_60

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model-rate/finetune_clm_span_60 --load_from_checkpoint experiment/model-rate/pretrain_span_60/lightning_logs/version_0/checkpoints/step=3000.ckpt

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-rate/finetune_infilling_span_60 --load_from_checkpoint experiment/model-rate/pretrain_span_60/lightning_logs/version_0/checkpoints/step=3000.ckpt

# 70
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_rate/span_70.yaml --trainer.default_root_dir experiment/model-rate/pretrain_span_70

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model-rate/finetune_clm_span_70 --load_from_checkpoint experiment/model-rate/pretrain_span_70/lightning_logs/version_0/checkpoints/step=3000.ckpt

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-rate/finetune_infilling_span_70 --load_from_checkpoint experiment/model-rate/pretrain_span_70/lightning_logs/version_0/checkpoints/step=3000.ckpt

# 80
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/pretrain_rate/span_80.yaml --trainer.default_root_dir experiment/model-rate/pretrain_span_80

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/clm.yaml --trainer.default_root_dir experiment/model-rate/finetune_clm_span_80 --load_from_checkpoint experiment/model-rate/pretrain_span_80/lightning_logs/version_0/checkpoints/step=3000.ckpt

python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/finetune/infilling.yaml --trainer.default_root_dir experiment/model-rate/finetune_infilling_span_80 --load_from_checkpoint experiment/model-rate/pretrain_span_80/lightning_logs/version_0/checkpoints/step=3000.ckpt
