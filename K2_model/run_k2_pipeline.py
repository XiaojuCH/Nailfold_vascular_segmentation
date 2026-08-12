"""Cross-platform entry point for the complete K2 training protocol."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
REQUIRED_RUN_FILES = ("best_model.pth", "config.json", "val_per_image.csv")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run F0 -> F3 -> soft targets -> K0(optional) -> K2.")
    parser.add_argument("--data_dir", required=True, help="Dataset root containing train/val/test.")
    parser.add_argument("--output_root", default="outputs", help="Relative paths are under K2_model.")
    parser.add_argument("--pretrained", default="reference_weights/R50+ViT-B_16.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--f0_epochs", type=int, default=50)
    parser.add_argument("--f3_epochs", type=int, default=50)
    parser.add_argument("--k2_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--k2_patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--f0_f3_lr", type=float, default=1e-4)
    parser.add_argument("--k2_lr", type=float, default=3e-5)
    parser.add_argument("--include_k0_control", action="store_true")
    parser.add_argument("--evaluate_test", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Accept both recommended "outputs/run" and a common accidental
    # "K2_model/outputs/run" when the command is launched from project root.
    parts = path.parts
    if parts and parts[0].lower() == ROOT.name.lower():
        path = Path(*parts[1:])
    return (ROOT / path).resolve()


def is_complete_run(path):
    return all((path / name).is_file() for name in REQUIRED_RUN_FILES)


def target_complete(target_root, data_dir, f0_weight, f3_weight):
    metadata_path = target_root / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if Path(metadata.get("f0_weight", "")).resolve() != f0_weight.resolve():
        return False
    if Path(metadata.get("f3_weight", "")).resolve() != f3_weight.resolve():
        return False
    records = {item.get("split"): item for item in metadata.get("splits", [])}
    for split in ("train", "val"):
        stems = sorted(
            path.stem for path in (data_dir / split / "images").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if records.get(split, {}).get("images") != len(stems):
            return False
        for stem in stems:
            if not (target_root / split / "ensemble_probabilities" / f"{stem}.npy").is_file():
                return False
            if not (target_root / split / "disagreement" / f"{stem}.npy").is_file():
                return False
    return metadata.get("f3_variant") == "directional_multiscale"


def run_step(name, command, logs_dir, dry_run):
    command = [str(value) for value in command]
    print("\n" + "=" * 68 + f"\n[START] {name}\n[CMD] {subprocess.list2cmdline(command)}\n" + "=" * 68)
    if dry_run:
        return
    log_path = logs_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
        log.write(f"\n[Exit code] {code}\n")
    if code != 0:
        raise RuntimeError(f"{name} failed (exit code {code}). See: {log_path}")


def run_training(name, run_dir, command, logs_dir, skip_existing, dry_run):
    if run_dir.exists():
        if is_complete_run(run_dir) and skip_existing:
            print(f"[SKIP] {name}: {run_dir}")
            return
        if is_complete_run(run_dir):
            raise FileExistsError(f"Completed run exists: {run_dir}. Add --skip_existing or use a new output root.")
        raise RuntimeError(f"Incomplete run exists and will not be overwritten: {run_dir}")
    run_step(name, command, logs_dir, dry_run)
    if not dry_run and not is_complete_run(run_dir):
        raise RuntimeError(f"{name} finished but expected outputs are incomplete: {run_dir}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = resolve_path(args.output_root)
    pretrained = resolve_path(args.pretrained)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_dir}")
    if not pretrained.is_file():
        raise FileNotFoundError(f"ImageNet initialization not found: {pretrained}")
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    python = sys.executable
    f0_dir, f3_dir = output_root / f"F0_seed{args.seed}", output_root / f"F3_seed{args.seed}"
    k0_dir, k2_dir = output_root / f"K0_seed{args.seed}", output_root / f"K2_seed{args.seed}"
    target_root = output_root / "dual_teacher_targets"
    f0_weight, f3_weight, k2_weight = f0_dir / "best_model.pth", f3_dir / "best_model.pth", k2_dir / "best_model.pth"

    run_step("00_dataset_audit", [python, CODE / "audit_dataset.py", "--data_dir", data_dir, "--out", output_root / "dataset_audit.json"], logs_dir, args.dry_run)
    common = ["--data_dir", data_dir, "--output_dir", output_root, "--seed", args.seed, "--batch_size", args.batch_size]
    run_training("01_train_F0_rgb_teacher", f0_dir, [python, CODE / "train_k2.py", "--stage", "f0", *common, "--exp_name", f"F0_seed{args.seed}", "--pretrained", pretrained, "--epochs", args.f0_epochs, "--patience", args.patience, "--lr", args.f0_f3_lr, "--intensity_aug", "on"], logs_dir, args.skip_existing, args.dry_run)
    run_training("02_train_F3_green_morphology_teacher", f3_dir, [python, CODE / "train_k2.py", "--stage", "f3", *common, "--exp_name", f"F3_seed{args.seed}", "--f3_variant", "directional_multiscale", "--epochs", args.f3_epochs, "--patience", args.patience, "--lr", args.f0_f3_lr, "--intensity_aug", "on"], logs_dir, args.skip_existing, args.dry_run)
    if target_root.exists():
        if target_complete(target_root, data_dir, f0_weight, f3_weight) and args.skip_existing:
            print(f"[SKIP] 03_generate_dual_teacher_soft_targets: {target_root}")
        elif target_complete(target_root, data_dir, f0_weight, f3_weight):
            raise FileExistsError(f"Completed targets exist: {target_root}. Add --skip_existing or use a new output root.")
        else:
            raise RuntimeError(f"Incomplete target directory will not be overwritten: {target_root}")
    else:
        run_step("03_generate_dual_teacher_soft_targets", [python, CODE / "generate_dual_teacher_targets.py", "--data_dir", data_dir, "--splits", "train,val", "--f0_weight", f0_weight, "--f3_weight", f3_weight, "--f3_variant", "directional_multiscale", "--out_dir", target_root, "--batch_size", args.batch_size, "--img_size", "256", "--threshold", "0.5"], logs_dir, args.dry_run)
    # K2 不需要 --pretrained：它先以 F0 权重初始化，ImageNet npz 会被完整覆盖，省去无效加载。
    k2_common = [python, CODE / "train_k2.py", "--stage", "k2", *common, "--init_weight", f0_weight, "--soft_target_dir", target_root / "train" / "ensemble_probabilities", "--epochs", args.k2_epochs, "--patience", args.k2_patience, "--lr", args.k2_lr, "--intensity_aug", "on"]
    if args.include_k0_control:
        run_training("04_train_K0_finetune_control", k0_dir, [*k2_common, "--exp_name", f"K0_seed{args.seed}", "--lambda_kd", "0"], logs_dir, args.skip_existing, args.dry_run)
    run_training("05_train_K2_dual_teacher_student", k2_dir, [*k2_common, "--exp_name", f"K2_seed{args.seed}", "--lambda_kd", "1.0"], logs_dir, args.skip_existing, args.dry_run)
    split = "test" if args.evaluate_test else "val"
    run_step(f"06_evaluate_K2_{split}", [python, CODE / "evaluate_k2.py", "--data_dir", data_dir, "--weight", k2_weight, "--split", split, "--out_dir", output_root / "evaluation", "--name", f"K2_seed{args.seed}", "--img_size", "256", "--batch_size", args.batch_size, "--threshold", "0.5"], logs_dir, args.dry_run)
    print(f"\n[ALL DONE]\nF0: {f0_weight}\nF3: {f3_weight}\nK2: {k2_weight}\nTargets: {target_root}")


if __name__ == "__main__":
    main()
