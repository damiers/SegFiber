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
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--runtime", choices=("standalone", "ddp"))
    parser.add_argument("--devices")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--keep-branch", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.set_defaults(handler=run)
    return parser


def run(args):
    from ..model.infer import infer

    config = load_config(args.config)
    apply_overrides(
        config,
        [
            ("dataset.infer.params.image_path", args.input),
            ("inference.params.output_path", args.output),
            ("inference.params.checkpoint", args.checkpoint),
            ("inference.params.keep_branch", True if args.keep_branch else None),
            ("runtime.name", args.runtime),
            (
                "runtime.params.devices",
                parse_devices(args.devices) if args.devices is not None else None,
            ),
        ],
    )
    apply_overrides(config, parse_set_overrides(args.set))
    _, runtime_params = module_config(config, "runtime")
    apply_overrides(
        config,
        [("runtime.params.devices", parse_devices(runtime_params["devices"]))],
    )
    pprint.pprint(to_plain_data(config), sort_dicts=False)
    return infer(to_plain_data(config), reset=args.reset)
