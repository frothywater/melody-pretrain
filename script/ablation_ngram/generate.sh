# ngram-mixed
python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-mixed/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-mixed/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/clm/ngram-mixed-best
python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-mixed/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-mixed/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/infilling/ngram-mixed-best
# ngram-8
python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-8/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-8/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/clm/ngram-8-best
python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-8/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-8/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/infilling/ngram-8-best
# ngram-4
python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-4/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-4/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/clm/ngram-4-best
python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-4/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-4/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/infilling/ngram-4-best
# ngram-pitch
python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-pitch/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-pitch/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/clm/ngram-pitch-best
python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-pitch/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-pitch/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/infilling/ngram-pitch-best
# # ngram-rhythm
# python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-rhythm/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-rhythm/finetune_clm/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/clm/ngram-rhythm-best
# python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-rhythm/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-rhythm/finetune_infilling/lightning_logs/version_0/checkpoints/best.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/infilling/ngram-rhythm-best

# ngram-mixed
python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-mixed/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-mixed/finetune_clm/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/clm/ngram-mixed
python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-mixed/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-mixed/finetune_infilling/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/infilling/ngram-mixed
# ngram-8
python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-8/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-8/finetune_clm/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/clm/ngram-8
python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-8/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-8/finetune_infilling/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/infilling/ngram-8
# ngram-4
python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-4/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-4/finetune_clm/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/clm/ngram-4
python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-4/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-4/finetune_infilling/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/infilling/ngram-4
# ngram-pitch
python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-pitch/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-pitch/finetune_clm/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/clm/ngram-pitch
python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-pitch/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-pitch/finetune_infilling/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/infilling/ngram-pitch
# # ngram-rhythm
# python main.py predict --config config/model/small.yaml --config config/predict/generate_clm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-rhythm/finetune_clm --ckpt_path experiment/ablation_ngram/model/ngram-rhythm/finetune_clm/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/clm/ngram-rhythm
# python main.py predict --config config/model/small.yaml --config config/predict/generate_infilling.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-rhythm/finetune_infilling --ckpt_path experiment/ablation_ngram/model/ngram-rhythm/finetune_infilling/lightning_logs/version_0/checkpoints/last.ckpt --trainer.callbacks.output_dir experiment/ablation_ngram/generated/infilling/ngram-rhythm


python compute_metric.py --test_dir ../dataset/clm_test --generated_group_dir experiment/ablation_ngram/generated/clm --dest_path experiment/ablation_ngram/result/clm.csv
python compute_metric.py --test_dir ../dataset/infilling_test --generated_group_dir experiment/ablation_ngram/generated/infilling --dest_path experiment/ablation_ngram/result/infilling.csv --force_filename