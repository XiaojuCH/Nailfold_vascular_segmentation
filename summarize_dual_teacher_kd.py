"""Summarize development-test deltas and choose the next KD branch."""

import argparse
import csv
import json
import os


METRICS = ("dice", "iou", "sensitivity", "precision", "specificity", "accuracy", "hd95", "cldice", "boundary_f1")


def get_args():
    parser = argparse.ArgumentParser(description="Summarize dual-teacher KD results.")
    parser.add_argument("--aggregate_csv", required=True)
    parser.add_argument("--teacher_metrics_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--control_name", required=True)
    parser.add_argument("--candidate_names", required=True, help="Comma-separated K-series experiment names.")
    parser.add_argument(
        "--summary_name",
        default="metrics_summary.csv",
        help="Output CSV filename within --out_dir.",
    )
    parser.add_argument(
        "--decision_name",
        default="first_night_decision.json",
        help="Output decision JSON filename within --out_dir.",
    )
    parser.add_argument("--strong_dice", type=float, default=0.7630)
    parser.add_argument("--continue_dice", type=float, default=0.7615)
    parser.add_argument("--min_delta_vs_control", type=float, default=0.001)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def to_float(row, key):
    return float(row[key])


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = get_args()
    aggregate_rows = read_csv(args.aggregate_csv)
    teacher_rows = read_csv(args.teacher_metrics_csv)
    os.makedirs(args.out_dir, exist_ok=True)

    teacher_test = {row["model"]: row for row in teacher_rows if row["split"] == "test"}
    if "F0" not in teacher_test or "ensemble" not in teacher_test:
        raise ValueError("Teacher metrics must include test rows for F0 and ensemble")
    original_f0 = teacher_test["F0"]
    ensemble = teacher_test["ensemble"]
    by_name = {row["experiment"]: row for row in aggregate_rows}
    if args.control_name not in by_name:
        raise ValueError(f"Control is absent from aggregate CSV: {args.control_name}")
    control = by_name[args.control_name]
    candidates = [item.strip() for item in args.candidate_names.split(",") if item.strip()]
    missing = [name for name in candidates if name not in by_name]
    if missing:
        raise ValueError(f"Candidates are absent from aggregate CSV: {missing}")

    output_rows = []
    for row in aggregate_rows:
        summary = {"experiment": row["experiment"], "weight": row["weight"]}
        for metric in METRICS:
            value = to_float(row, metric)
            summary[metric] = value
            summary[f"delta_vs_original_f0_{metric}"] = value - to_float(original_f0, metric)
            summary[f"delta_vs_control_{metric}"] = value - to_float(control, metric)
            summary[f"delta_vs_ensemble_{metric}"] = value - to_float(ensemble, metric)
        output_rows.append(summary)

    candidates_by_dice = sorted((by_name[name] for name in candidates), key=lambda row: to_float(row, "dice"), reverse=True)
    best = candidates_by_dice[0]
    best_name = best["experiment"]
    best_dice = to_float(best, "dice")
    delta_vs_control = best_dice - to_float(control, "dice")
    continue_multiseed = best_dice >= args.continue_dice and delta_vs_control >= args.min_delta_vs_control
    strong_positive = best_dice >= args.strong_dice and to_float(best, "hd95") <= to_float(control, "hd95") and to_float(best, "boundary_f1") >= to_float(control, "boundary_f1")
    decision = {
        "control_name": args.control_name,
        "best_candidate": best_name,
        "best_candidate_dice": best_dice,
        "best_candidate_delta_vs_control_dice": delta_vs_control,
        "strong_positive": strong_positive,
        "continue_multiseed": continue_multiseed,
        "next_phase": "multiseed" if continue_multiseed else "fallback_k3_k4",
        "criteria": {
            "continue_dice": args.continue_dice,
            "min_delta_vs_control": args.min_delta_vs_control,
            "strong_dice": args.strong_dice,
        },
        "source": {
            "aggregate_csv": os.path.abspath(args.aggregate_csv),
            "teacher_metrics_csv": os.path.abspath(args.teacher_metrics_csv),
        },
    }
    fields = ["experiment", "weight", *METRICS]
    for baseline in ("original_f0", "control", "ensemble"):
        fields.extend(f"delta_vs_{baseline}_{metric}" for metric in METRICS)
    summary_path = os.path.join(args.out_dir, args.summary_name)
    decision_path = os.path.join(args.out_dir, args.decision_name)
    write_csv(summary_path, output_rows, fields)
    with open(decision_path, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)

    print("[KD summary complete]")
    print(f"Best candidate: {best_name}")
    print(f"Dice: {best_dice:.4f}; delta vs K0: {delta_vs_control:+.4f}")
    print(f"Next phase: {decision['next_phase']}")


if __name__ == "__main__":
    main()
