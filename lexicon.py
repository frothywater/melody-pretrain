import argparse
import os
from glob import glob

from melody_pretrain.ngram import BarOnsetNgram, MixedNgram, NgramExtractor, PitchClassNgram

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand", type=str, choices=["extract", "build", "prepare"])
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--midi_dir", type=str)
    parser.add_argument("--length", type=int)
    parser.add_argument("--ngram_kind", type=str, default="mixed", choices=["pitch_class", "bar_onset", "mixed"])
    args = parser.parse_args()

    ngram_config_path = os.path.join(args.dataset_dir, "ngram", "config.json")
    if args.subcommand == "extract":
        assert args.midi_dir is not None
        midi_files = glob(args.midi_dir + "/**/*.mid", recursive=True)
        ngram_dir = os.path.join(args.dataset_dir, "ngram", "data")
        os.makedirs(ngram_dir, exist_ok=True)

        assert args.length is not None
        if args.ngram_kind == "pitch_class":
            ngram_type = PitchClassNgram
        elif args.ngram_kind == "bar_onset":
            ngram_type = BarOnsetNgram
        elif args.ngram_kind == "mixed":
            ngram_type = MixedNgram
        extractor = NgramExtractor(n_range=(3, args.length), ngram_type=ngram_type)
        extractor.extract_ngrams(midi_files, ngram_dir)
        extractor.save_config(ngram_config_path)

    elif args.subcommand == "build":
        lexicon_path = os.path.join(args.dataset_dir, "ngram", "lexicon.pkl")
        ngram_files = glob(os.path.join(args.dataset_dir, "ngram", "data", "*.pkl"))

        extractor = NgramExtractor.from_config(ngram_config_path)
        extractor.build_lexicon(ngram_files, lexicon_path)

    elif args.subcommand == "prepare":
        lexicon_path = os.path.join(args.dataset_dir, "ngram", "lexicon.pkl")
        ngram_files = glob(os.path.join(args.dataset_dir, "ngram", "data", "*.pkl"))
        label_dir = os.path.join(args.dataset_dir, "ngram", "label")
        os.makedirs(label_dir, exist_ok=True)

        extractor = NgramExtractor.from_config(ngram_config_path)
        extractor.prepare_ngram_labels(ngram_files, lexicon_path, label_dir)
