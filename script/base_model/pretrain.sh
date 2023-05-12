# seg
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/base_model/seg.yaml --trainer.default_root_dir experiment/base_model/model/seg/pretrain

# noseg
python main.py fit --config config/trainer.yaml --config config/model/base.yaml --config config/base_model/noseg.yaml --trainer.default_root_dir experiment/base_model/model/noseg/pretrain