import pprint

from ..model.config import (
    apply_overrides,
    load_config,
    module_config,
    parse_devices,
    parse_set_overrides,
    to_plain_data,
)


def configure_parser(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--name")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--runtime", choices=("standalone", "ddp"))
    parser.add_argument("--devices")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.set_defaults(handler=run)
    return parser


def run(args):
    from ..model.train import train

    config = load_config(args.config)
    apply_overrides(
        config,
        [
            ("experiment.name", args.name),
            ("experiment.output_dir", args.output_dir),
            ("experiment.seed", args.seed),
            ("runtime.name", args.runtime),
            (
                "runtime.params.devices",
                parse_devices(args.devices) if args.devices is not None else None,
            ),
            ("trainer.params.epochs", args.epochs),
        ],
    )
    apply_overrides(config, parse_set_overrides(args.set))
    _, runtime_params = module_config(config, "runtime")
    apply_overrides(
        config,
        [("runtime.params.devices", parse_devices(runtime_params["devices"]))],
    )
    pprint.pprint(to_plain_data(config), sort_dicts=False)
    return train(config, reset=args.reset, slurm=args.slurm)
