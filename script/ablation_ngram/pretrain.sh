# ngram-mixed
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/ablation_ngram/ngram-mixed.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-mixed/pretrain
# ngram-8
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/ablation_ngram/ngram-8.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-8/pretrain
# ngram-4
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/ablation_ngram/ngram-4.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-4/pretrain
# ngram-pitch
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/ablation_ngram/ngram-pitch.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-pitch/pretrain
# ngram-rhythm
python main.py fit --config config/trainer.yaml --config config/model/small.yaml --config config/ablation_ngram/ngram-rhythm.yaml --trainer.default_root_dir experiment/ablation_ngram/model/ngram-rhythm/pretrain