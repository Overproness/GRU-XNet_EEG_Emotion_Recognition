"""Prepare complete trial features, then extend the frozen trainer to other targets."""
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gruxnet.transfer_extension import prepare_features, run

parser = ArgumentParser(description=__doc__)
commands = parser.add_subparsers(dest="command", required=True)
prepare = commands.add_parser("prepare")
prepare.add_argument("--common-cache", type=Path, required=True)
prepare.add_argument("--previous-pack", type=Path, required=True)
prepare.add_argument("--output", type=Path, required=True)
train = commands.add_parser("run")
train.add_argument("--feature-cache", type=Path, required=True)
train.add_argument("--common-cache", type=Path, required=True)
train.add_argument("--target", choices=["DEAP", "GAMEEMO"], required=True)
train.add_argument("--plan", type=Path, required=True)
train.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.command == "prepare":
    prepare_features(args.common_cache, args.previous_pack, args.output)
else:
    run(args.feature_cache, args.common_cache, args.target, args.plan, args.output)
