import argparse
import subprocess
import time
from pathlib import Path

from ..config import load_config, prepare_run_paths, resolve_run_paths, save_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Submit a SegFiber Slurm job")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-cpus", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--partition")
    parser.add_argument("--node")
    parser.add_argument("--load-env", default="")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args(argv)


def build_sbatch_script(args, paths):
    template = Path(__file__).with_name("slurm_template.sh").read_text(
        encoding="utf-8"
    )
    job_name = f"{args.job_name}_{time.strftime('%m%d-%H%M%S')}"
    replacements = {
        "${JOB_NAME}": job_name,
        "${LOG_PATH}": str(paths.slurm / job_name),
        "${NUM_NODES}": str(args.num_nodes),
        "${NUM_CPUS_PER_TASK}": str(args.num_cpus),
        "${NUM_GPUS_PER_NODE}": str(args.num_gpus),
        "${PARTITION_OPTION}": (
            f"#SBATCH --partition={args.partition}" if args.partition else ""
        ),
        "${NODE_OPTION}": f"#SBATCH --nodelist={args.node}" if args.node else "",
        "${LOAD_ENV}": args.load_env,
        "${COMMAND}": args.command,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    script_path = paths.slurm / f"{job_name}.sh"
    script_path.write_text(template, encoding="utf-8")
    return script_path


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    paths = resolve_run_paths(config)
    prepare_run_paths(paths, reset=args.reset)
    save_config(config, paths.run / "config.yaml")
    script_path = build_sbatch_script(args, paths)
    subprocess.run(["sbatch", str(script_path)], check=True)
    print(f"Submitted {script_path}")


if __name__ == "__main__":
    main()
