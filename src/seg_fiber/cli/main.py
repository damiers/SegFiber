import argparse

from .infer import configure_parser as configure_infer_parser
from .merge import configure_parser as configure_merge_parser
from .train import configure_parser as configure_train_parser


def build_parser():
    parser = argparse.ArgumentParser(prog="segfiber")
    commands = parser.add_subparsers(dest="command", required=True)
    configure_train_parser(commands.add_parser("train", help="train a model"))
    configure_infer_parser(commands.add_parser("infer", help="segment a volume"))
    configure_merge_parser(
        commands.add_parser("merge", help="merge z-slab databases")
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    main()
