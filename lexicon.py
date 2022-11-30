import argparse
import os

from melody_pretrain.dataset.ngram import extract, prepare_lexicon, render_midi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand", choices=["extract", "prepare", "render"])
    parser.add_argument("--midi_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--top_p", type=float, default=0.1)
    args = parser.parse_args()

    if args.subcommand == "extract":
        assert args.midi_dir is not None, "midi_dir is required"
        assert args.dataset_dir is not None, "dataset_dir is required"
        data_dir = os.path.join(args.dataset_dir, "ngram_data")
        ngram_range = range(1, args.length + 1)
        extract(args.midi_dir, data_dir, ngram_range)
    elif args.subcommand == "prepare":
        assert args.dataset_dir is not None, "dataset_dir is required"
        data_dir = os.path.join(args.dataset_dir, "ngram_data")
        # check if ngram data exists
        if not os.path.exists(os.path.join(data_dir, "ngram_pitch.pkl")) or not os.path.exists(
            os.path.join(data_dir, "ngram_rhythm.pkl")
        ):
            print("ngram data not found, extracting...")
            assert args.midi_dir is not None, "midi_dir is required"
            ngram_range = range(1, args.length + 1)
            extract(args.midi_dir, data_dir, ngram_range)
        prepare_lexicon(data_dir, top_p=args.top_p)
    elif args.subcommand == "render":
        assert args.dataset_dir is not None, "dataset_dir is required"
        data_dir = os.path.join(args.dataset_dir, "ngram_data")
        rendered_dir = os.path.join(data_dir, "rendered")
        render_midi(data_dir, rendered_dir)
