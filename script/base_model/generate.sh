# seg
python main.py predict --config config/model/base.yaml \
--config config/predict/generate_clm.yaml \
--trainer.default_root_dir experiment/base_model/model/seg/finetune_clm \
--ckpt_path experiment/base_model/model/seg/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt \
--trainer.callbacks.output_dir experiment/base_model/generated/clm/seg_best

python main.py predict --config config/model/base.yaml \
--config config/predict/generate_clm.yaml \
--trainer.default_root_dir experiment/base_model/model/seg/finetune_clm \
--ckpt_path experiment/base_model/model/seg/finetune_clm/lightning_logs/version_0/checkpoints/last.ckpt \
--trainer.callbacks.output_dir experiment/base_model/generated/clm/seg

python main.py predict --config config/model/base.yaml \
--config config/predict/generate_infilling.yaml \
--trainer.default_root_dir experiment/base_model/model/seg/finetune_infilling \
--ckpt_path experiment/base_model/model/seg/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt \
--trainer.callbacks.output_dir experiment/base_model/generated/infilling/seg_best

python main.py predict --config config/model/base.yaml \
--config config/predict/generate_infilling.yaml \
--trainer.default_root_dir experiment/base_model/model/seg/finetune_infilling \
--ckpt_path experiment/base_model/model/seg/finetune_infilling/lightning_logs/version_0/checkpoints/last.ckpt \
--trainer.callbacks.output_dir experiment/base_model/generated/infilling/seg

# noseg
python main.py predict --config config/model/base.yaml \
--config config/predict/generate_clm.yaml \
--config config/base_model/noseg_data.yaml \
--trainer.default_root_dir experiment/base_model/model/noseg/finetune_clm \
--ckpt_path experiment/base_model/model/noseg/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt \
--trainer.callbacks.output_dir experiment/base_model/generated/clm/noseg_best

python main.py predict --config config/model/base.yaml \
--config config/predict/generate_clm.yaml \
--config config/base_model/noseg_data.yaml \
--trainer.default_root_dir experiment/base_model/model/noseg/finetune_clm \
--ckpt_path experiment/base_model/model/noseg/finetune_clm/lightning_logs/version_0/checkpoints/last.ckpt \
--trainer.callbacks.output_dir experiment/base_model/generated/clm/noseg

python main.py predict --config config/model/base.yaml \
--config config/predict/generate_infilling.yaml \
--config config/base_model/noseg_data.yaml \
--trainer.default_root_dir experiment/base_model/model/noseg/finetune_infilling \
--ckpt_path experiment/base_model/model/noseg/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt \
--trainer.callbacks.output_dir experiment/base_model/generated/infilling/noseg_best \
--data.dataset_dir experiment/dataset/infilling_noseg \
--model.dataset_dir experiment/dataset/infilling_noseg

python main.py predict --config config/model/base.yaml \
--config config/predict/generate_infilling.yaml \
--config config/base_model/noseg_data.yaml \
--trainer.default_root_dir experiment/base_model/model/noseg/finetune_infilling \
--ckpt_path experiment/base_model/model/noseg/finetune_infilling/lightning_logs/version_0/checkpoints/last.ckpt \
--trainer.callbacks.output_dir experiment/base_model/generated/infilling/noseg \
--data.dataset_dir experiment/dataset/infilling_noseg \
--model.dataset_dir experiment/dataset/infilling_noseg

python compute_metric.py --test_dir ../dataset/clm_test --generated_group_dir experiment/base_model/generated/clm --dest_path experiment/base_model/result/clm.csv
python compute_metric.py --test_dir ../dataset/infilling_test --generated_group_dir experiment/base_model/generated/infilling --dest_path experiment/base_model/result/infilling.csv --force_filename