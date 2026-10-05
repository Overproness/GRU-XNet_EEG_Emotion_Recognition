from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(description="Audited GRU-XNet publication pipeline (historical scripts are separate)")
    commands = parser.add_subparsers(dest="command", required=True)
    audit_cmd = commands.add_parser("audit", help="Verify all sources and recover real valence ratings")
    audit_cmd.add_argument("--data-root", type=Path, required=True)
    audit_cmd.add_argument("--output", type=Path, required=True)
    audit_cmd.add_argument("--provenance", type=Path, help="Snapshot and hash a declared source catalog")
    compare_cmd = commands.add_parser("compare-deap", help="Compare every local DEAP signal and rating with a supplied reference release")
    compare_cmd.add_argument("--official", type=Path, required=True, help="Author-downloaded Python ZIP or directory containing s01.dat through s32.dat")
    compare_cmd.add_argument("--data-root", type=Path, required=True)
    compare_cmd.add_argument("--output", type=Path, required=True, help="Fresh JSON output file; neither data copy is modified")
    prep = commands.add_parser("prepare", help="Build immutable disk-backed windows from originals")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--provenance", type=Path, help="Bind a declared source catalog into this new cache's fingerprint")
    prep.add_argument("--montage", choices=["common14", "canonical62"], default="common14")
    prep.add_argument("--seconds", type=int, default=4)
    run = commands.add_parser("train")
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--protocol", choices=["subject", "lodo", "loso"], default="subject")
    run.add_argument("--target", choices=["DEAP", "GAMEEMO", "SEEDIV"])
    run.add_argument("--test-subject", help="Dataset-qualified ID, e.g. DEAP:S01")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--accumulation", type=int, default=4)
    run.add_argument("--patience", type=int, default=8)
    run.add_argument("--max-windows-per-trial", type=int, help="A capped run is labeled pilot in every artifact")
    run.add_argument("--recurrent", choices=["gru", "lstm", "none"], default="gru")
    run.add_argument("--frequency-pooling", choices=["flatten", "mean"], default="flatten",
                     help="Preserve frequency identity (v2), or reproduce the exploratory averaging model (v1)")
    run.add_argument("--no-attention", action="store_true")
    run.add_argument("--no-augment", action="store_true")
    run.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    run.add_argument("--verify-cache", action="store_true", help="Rehash cached EEG before training")
    verify = commands.add_parser("verify", help="Re-evaluate best checkpoint and compare every saved metric")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--cache", type=Path)
    review = commands.add_parser("review-sam", help="Render all SAM valence selections for visual QA")
    review.add_argument("--audit-dir", type=Path, required=True)
    diagnostic = commands.add_parser("overfit", help="Training-only small-batch capacity diagnostic")
    diagnostic.add_argument("--cache", type=Path, required=True)
    diagnostic.add_argument("--split", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    diagnostic.add_argument("--steps", type=int, default=300)
    baseline_cmd = commands.add_parser("baseline", help="Training-fitted log-bandpower logistic regression control")
    baseline_cmd.add_argument("--cache", type=Path, required=True)
    baseline_cmd.add_argument("--split", type=Path, required=True)
    baseline_cmd.add_argument("--output", type=Path, required=True)
    deap_prepare = commands.add_parser("prepare-deap", help="Build a fresh canonical 32-electrode DEAP-only cache")
    deap_prepare.add_argument("--data-root", type=Path, required=True)
    deap_prepare.add_argument("--output", type=Path, required=True)
    deap_prepare.add_argument("--provenance", type=Path)
    deap_train = commands.add_parser("train-deap-control", help="Raw EEGNet control with independent validation participants")
    deap_train.add_argument("--cache", type=Path, required=True)
    deap_train.add_argument("--output", type=Path, required=True)
    deap_train.add_argument("--seed", type=int, default=42)
    deap_train.add_argument("--epochs", type=int, default=100)
    deap_train.add_argument("--minimum-epochs", type=int, default=25)
    deap_train.add_argument("--patience", type=int, default=15)
    deap_train.add_argument("--batch-size", type=int, default=32)
    deap_train.add_argument("--learning-rate", type=float, default=.001)
    deap_train.add_argument("--normalization", choices=["train-channel", "window"], default="train-channel")
    deap_train.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    deap_verify = commands.add_parser("verify-deap-control", help="Reconstruct training-only normalization, checkpoint choice, and all test metrics")
    deap_verify.add_argument("--run", type=Path, required=True)
    deap_verify.add_argument("--cache", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "audit":
        from .data import audit
        audit(args["data_root"], args["output"], provenance=args["provenance"])
    elif command == "compare-deap":
        from .provenance import compare_deap
        compare_deap(**args)
    elif command == "prepare":
        from .prepare import prepare
        prepare(args["data_root"], args["output"], args["montage"], args["seconds"], provenance=args["provenance"])
    elif command == "train":
        from .train import train
        args["attention"] = not args.pop("no_attention")
        args["augment"] = not args.pop("no_augment")
        args["device_name"] = args.pop("device")
        train(**args)
    elif command == "verify":
        from .train import verify_run
        verify_run(**args)
    elif command == "review-sam":
        from .review import render_sam_review
        render_sam_review(args["audit_dir"])
    elif command == "overfit":
        from .diagnostics import overfit
        overfit(**args)
    elif command == "prepare-deap":
        from .deap_control import prepare_deap
        prepare_deap(**args)
    elif command == "train-deap-control":
        from .deap_control import train_control
        args["device_name"] = args.pop("device")
        train_control(**args)
    elif command == "verify-deap-control":
        from .deap_control import verify_control
        verify_control(**args)
    else:
        from .baseline import baseline
        baseline(**args)


if __name__ == "__main__":
    main()
