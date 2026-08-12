"""Summarize intensity-on CGMA prior and structure-auxiliary ablations."""

import argparse
import csv
import json
import os


METRICS = (
    "dice",
    "iou",
    "sensitivity",
    "precision",
    "specificity",
    "accuracy",
    "hd95",
    "cldice",
    "boundary_f1",
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Summarize intensity-on structure auxiliary ablations."
    )
    parser.add_argument("--run_summary", required=True)
    parser.add_argument("--val_aggregate", required=True)
    parser.add_argument("--development_test_aggregate", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_decision", required=True)
    return parser.parse_args()


def read_by_experiment(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return {row["experiment"]: row for row in csv.DictReader(handle)}


def main():
    args = get_args()
    run_rows = read_by_experiment(args.run_summary)
    val_rows = read_by_experiment(args.val_aggregate)
    test_rows = read_by_experiment(args.development_test_aggregate)
    controls = [
        row for row in run_rows.values() if row.get("role") == "control"
    ]
    if len(controls) != 1:
        raise ValueError(f"Expected exactly one control row, got {len(controls)}")
    control_name = controls[0]["experiment"]
    if control_name not in val_rows or control_name not in test_rows:
        raise ValueError(f"Control is missing from aggregate evaluation: {control_name}")

    val_control = val_rows[control_name]
    test_control = test_rows[control_name]
    summaries = []
    for name, run in run_rows.items():
        if name not in val_rows or name not in test_rows:
            raise ValueError(f"Missing evaluated experiment: {name}")
        val = val_rows[name]
        test = test_rows[name]
        summary = {
            "experiment": name,
            "role": run["role"],
            "variant": run["variant"],
            "weight": run["weight"],
            "best_val_epoch": run["best_val_epoch"],
            "boundary_weight": run["boundary_weight"],
            "centerline_weight": run["centerline_weight"],
        }
        for metric in METRICS:
            summary[f"val_{metric}"] = float(val[metric])
            summary[f"delta_val_vs_control_{metric}"] = (
                float(val[metric]) - float(val_control[metric])
            )
            summary[f"development_test_{metric}"] = float(test[metric])
            summary[f"delta_development_test_vs_control_{metric}"] = (
                float(test[metric]) - float(test_control[metric])
            )
        summaries.append(summary)

    candidates = [row for row in summaries if row["role"] != "control"]
    best_by_val = max(candidates, key=lambda row: row["val_dice"])
    candidate_for_multiseed = (
        best_by_val["delta_val_vs_control_dice"] >= 0.001
        and best_by_val["delta_val_vs_control_boundary_f1"] >= 0.0
        and best_by_val["delta_val_vs_control_hd95"] <= 0.0
    )
    decision = {
        "control": control_name,
        "best_by_val_dice": best_by_val["experiment"],
        "best_by_val_dice_delta": best_by_val["delta_val_vs_control_dice"],
        "candidate_for_multiseed": candidate_for_multiseed,
        "recommended_next_step": (
            "run_control_and_best_seed43_44"
            if candidate_for_multiseed
            else "stop_structure_auxiliary_line_and_prepare_group_meeting"
        ),
        "rule": (
            "Best candidate must improve val Dice by at least +0.001 over control "
            "without worsening val BoundaryF1 or HD95."
        ),
    }

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with open(args.out_decision, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)

    print("[Structure auxiliary summary complete]")
    print(
        f"Best val candidate: {best_by_val['experiment']}; "
        f"Dice delta={best_by_val['delta_val_vs_control_dice']:+.4f}; "
        f"next={decision['recommended_next_step']}"
    )


if __name__ == "__main__":
    main()
