def configure_parser(parser):
    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="directory containing z-slab .db files",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DB",
        help="path of the merged database",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="replace an existing output database",
    )
    parser.set_defaults(handler=run)
    return parser


def run(args):
    from ..model.utils.neurodb_merge import merge_z_slab_databases

    output = merge_z_slab_databases(args.input_dir, args.output, reset=args.reset)
    print(f"Merged database: {output}")
    return output
