import argparse
import os
from glob import glob

from melody_pretrain.ngram import BarOnsetNgram, MixedNgram, NgramExtractor, PitchClassNgram

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand", type=str, choices=["extract", "build", "prepare"])
    parser.add_argument("--midi_dir", type=str)
    parser.add_argument("--ngram_dir", type=str)
    parser.add_argument("--lexicon_path", type=str)
    parser.add_argument("--label_dir", type=str)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--ngram_kind", type=str, default="mixed", choices=["pitch_class", "bar_onset", "mixed"])
    args = parser.parse_args()

    if args.ngram_kind == "pitch_class":
        ngram_type = PitchClassNgram
    elif args.ngram_kind == "bar_onset":
        ngram_type = BarOnsetNgram
    elif args.ngram_kind == "mixed":
        ngram_type = MixedNgram

    if args.subcommand == "extract":
        assert args.midi_dir is not None and args.ngram_dir is not None
        midi_files = glob(args.midi_dir + "/**/*.mid", recursive=True)
        os.makedirs(args.ngram_dir, exist_ok=True)

        extractor = NgramExtractor(n_range=(3, args.length), ngram_type=ngram_type)
        extractor.extract_ngrams(midi_files, args.ngram_dir)

    elif args.subcommand == "build":
        assert args.ngram_dir is not None and args.lexicon_path is not None
        ngram_files = glob(os.path.join(args.ngram_dir, "*.pkl"))

        extractor = NgramExtractor(n_range=(3, args.length), ngram_type=ngram_type)
        extractor.build_lexicon(ngram_files, args.lexicon_path)

    elif args.subcommand == "prepare":
        assert args.ngram_dir is not None and args.label_dir is not None and args.lexicon_path is not None
        ngram_files = glob(os.path.join(args.ngram_dir, "*.pkl"))
        os.makedirs(args.label_dir, exist_ok=True)

        extractor = NgramExtractor(n_range=(3, args.length), ngram_type=ngram_type)
        extractor.prepare_ngram_labels(ngram_files, args.label_dir, args.lexicon_path)
