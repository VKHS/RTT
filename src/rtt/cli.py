"""Command line for the separate dataset experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from .configuration import resolve_config
from .datasets import DATASETS, DatasetError
from .experiment import Experiment
from .runtime import configure, json_safe, write_json


COMMANDS = ("inspect", "prepare", "train", "calibrate", "select", "audit", "report", "run", "predict", "evaluate")


def build_parser(default_config=None):
    parser = argparse.ArgumentParser(description="Reference-Tail Trust dataset experiments")
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.add_argument("--config", default=str(default_config) if default_config else None,
                        help="Dataset-specific JSON configuration")
    parser.add_argument("--run-dir", default=None, help="One frozen experiment directory")
    parser.add_argument("--profile", choices=("full", "pilot", "smoke"), default="full")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--download", action="store_true", help="Allow official data/pretrained-weight downloads")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--output", default=None, help="New directory for predict/evaluate outputs")
    parser.add_argument("--pool", default="audit", choices=("fit", "calibration", "selection", "audit", "reserved", "transfer"))
    parser.add_argument("--call-index", type=int, default=0)
    parser.add_argument("--member", default=None, help="Explicit exploratory calibrated member for prediction")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=None, help="Population-evaluation call size")
    parser.add_argument("--list-datasets", action="store_true")
    return parser


def main(argv=None, default_config=None):
    parser = build_parser(default_config)
    args = parser.parse_args(argv)
    if args.list_datasets:
        print("\n".join(DATASETS))
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    if not args.config:
        parser.error("--config is required; each experiments/<dataset>/run.py supplies its own default")
    if args.threads < 1 or args.call_index < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("threads and batch-size must be positive; call-index must be nonnegative")
    if args.command != "inspect" and not args.run_dir:
        parser.error("--run-dir is required for this command")
    try:
        config = resolve_config(args.config, profile=args.profile, data_root=args.data_root, download=args.download)
        if args.command == "inspect":
            print(json.dumps(json_safe(config), indent=2))
            return 0
        configure(config["protocol"]["seed"], args.threads)
        experiment = Experiment(config, args.run_dir, args.device)
        print(f"[RTT] dataset={config['dataset']['name']} stage={args.command} profile={args.profile} run={experiment.directory}", flush=True)
        if args.command == "run":
            experiment.prepare()
            for command, artifact in (("train", "training.json"), ("calibrate", "calibration.json"), ("select", "selection.json")):
                if not (experiment.directory / artifact).exists():
                    print(f"[RTT] stage={command}", flush=True)
                    getattr(experiment, command)()
                else:
                    experiment.load()
            if not list((experiment.directory / "records").glob("audit_shard_*.meta.json")):
                experiment.audit()
            result = experiment.report()
        elif args.command == "audit":
            result = experiment.audit(args.shard_index, args.shards)
        elif args.command == "predict":
            output = args.output or str(experiment.directory / "predictions" / f"{args.pool}_{args.call_index:06d}")
            result = experiment.predict(output, pool=args.pool, index=args.call_index, member_name=args.member)
        elif args.command == "evaluate":
            output = args.output or str(experiment.directory / "population_evaluation" / args.split)
            result = experiment.evaluate_population(output, split=args.split, batch_size=args.batch_size)
        else:
            result = getattr(experiment, args.command)()
        display = {key: value for key, value in result.items() if key not in {"history", "members", "incumbent", "adaptation", "metadata"}}
        print(json.dumps(json_safe(display), indent=2, allow_nan=False))
        return 0
    except Exception as error:
        message = str(error)
        infrastructure = isinstance(error, (OSError, MemoryError)) or any(token in message.lower() for token in ("out of memory", "cuda was requested but is unavailable", "cuda error", "connection refused", "temporary failure"))
        code = 75 if infrastructure else 2 if isinstance(error, (ValueError, DatasetError, FileNotFoundError)) else 1
        if args.run_dir:
            write_json(Path(args.run_dir) / "failures" / f"{args.command}_{args.shard_index:04d}.json",
                       {"status": "FAILED", "stage": args.command, "error_type": type(error).__name__,
                        "message": message, "retriable": infrastructure, "traceback": traceback.format_exc()})
        print(f"[RTT ERROR] {type(error).__name__}: {message}", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
