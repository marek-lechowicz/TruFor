"""Upload best.pth.tar checkpoints to their corresponding W&B runs as artifacts.

For every subdirectory of `weights/` we resume the W&B run whose id equals the
subdirectory name (matching how train.py initializes wandb with
`id=args.experiment, resume='allow'`) and log `best.pth.tar` as a model artifact.
"""

import argparse
import os
import sys
from pathlib import Path

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "weights",
        help="Root directory containing per-experiment weight subdirs.",
    )
    parser.add_argument("--project", default="fake-flickr", help="W&B project.")
    parser.add_argument("--entity", default="budalema", help="W&B entity.")
    parser.add_argument(
        "--checkpoint-name",
        default="best.pth.tar",
        help="Checkpoint filename to upload from each subdir.",
    )
    parser.add_argument(
        "--artifact-alias",
        default="best",
        help="Alias to attach to the logged artifact (in addition to 'latest').",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional list of experiment names (subdir names) to restrict to.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without touching W&B.",
    )
    return parser.parse_args()


def upload_for_experiment(
    experiment: str,
    checkpoint_path: Path,
    project: str,
    entity: str,
    alias: str,
    dry_run: bool,
) -> None:
    print(f"\n=== {experiment} ===")
    print(f"checkpoint: {checkpoint_path} ({checkpoint_path.stat().st_size} bytes)")
    print(f"target run: {entity}/{project}/{experiment}")

    if dry_run:
        print("dry-run: skipping upload")
        return

    run = wandb.init(
        project=project,
        entity=entity,
        id=experiment,
        resume="must",
        job_type="upload-checkpoint",
    )

    artifact = wandb.Artifact(
        name=f"{experiment}-best",
        type="model",
        metadata={
            "experiment": experiment,
            "source_path": str(checkpoint_path),
            "filename": checkpoint_path.name,
        },
    )
    artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    run.log_artifact(artifact, aliases=[alias, "latest"])
    artifact.wait()
    print(f"logged artifact: {artifact.name} (aliases: {alias}, latest)")
    run.finish()


def main() -> int:
    args = parse_args()

    if not args.weights_dir.is_dir():
        print(f"weights dir not found: {args.weights_dir}", file=sys.stderr)
        return 1

    subdirs = sorted(p for p in args.weights_dir.iterdir() if p.is_dir())
    if args.only:
        wanted = set(args.only)
        subdirs = [p for p in subdirs if p.name in wanted]
        missing = wanted - {p.name for p in subdirs}
        if missing:
            print(f"warning: requested experiments not found: {sorted(missing)}", file=sys.stderr)

    if not subdirs:
        print("no experiment subdirs to process", file=sys.stderr)
        return 1

    failures = []
    for sub in subdirs:
        ckpt = sub / args.checkpoint_name
        if not ckpt.is_file():
            print(f"skip {sub.name}: missing {args.checkpoint_name}", file=sys.stderr)
            failures.append(sub.name)
            continue
        try:
            upload_for_experiment(
                experiment=sub.name,
                checkpoint_path=ckpt,
                project=args.project,
                entity=args.entity,
                alias=args.artifact_alias,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001 — surface any per-run failure and continue
            print(f"failed {sub.name}: {exc}", file=sys.stderr)
            failures.append(sub.name)
            if wandb.run is not None:
                wandb.finish(exit_code=1)

    if failures:
        print(f"\nfinished with failures: {failures}", file=sys.stderr)
        return 2
    print("\nall uploads complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
