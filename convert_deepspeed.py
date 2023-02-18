from argparse import ArgumentParser
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    convert_zero_checkpoint_to_fp32_state_dict(args.checkpoint_path, args.output_path)
