"""Merge the CGMA 2 x 2 validation and development-test results."""

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
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return {row["experiment"]: row for row in csv.DictReader(handle)}


def get_args():
    parser = argparse.ArgumentParser(description="Summarize the CGMA 2 x 2 probe.")
    parser.add_argument("--run_summary", required=True)
    parser.add_argument("--val_aggregate", required=True)
    parser.add_argument("--development_test_aggregate", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_decision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    run_rows = read_csv(args.run_summary)
    val_rows = read_csv(args.val_aggregate)
    test_rows = read_csv(args.development_test_aggregate)
    experiments = (
        f"M0_transunet_no_prior_no_aux_seed{args.seed}",
        f"M1_fixed_prior_seed{args.seed}",
        f"M2_structure_aux_seed{args.seed}",
        f"M3_cgma_full_seed{args.seed}",
    )
    missing = [
        name
        for name in experiments
        if name not in run_rows or name not in val_rows or name not in test_rows
    ]
    if missing:
        raise ValueError(f"Missing CGMA rows: {missing}")

    val_m0 = val_rows[experiments[0]]
    test_m0 = test_rows[experiments[0]]
    summaries = []
    for name in experiments:
        run = run_rows[name]
        val = val_rows[name]
        test = test_rows[name]
        summary = {
            "experiment": name,
            "weight": run["weight"],
            "best_val_epoch": run["best_val_epoch"],
            "train_variant": run["variant"],
        }
        for metric in METRICS:
            summary[f"val_{metric}"] = float(val[metric])
            summary[f"delta_val_vs_m0_{metric}"] = float(val[metric]) - float(val_m0[metric])
            summary[f"development_test_{metric}"] = float(test[metric])
            summary[f"delta_development_test_vs_m0_{metric}"] = float(test[metric]) - float(test_m0[metric])
        summaries.append(summary)

    by_name = {row["experiment"]: row for row in summaries}
    m1 = by_name[experiments[1]]
    m2 = by_name[experiments[2]]
    m3 = by_name[experiments[3]]
    m1_signal = m1["delta_val_vs_m0_dice"] >= 0.002 or (
        m1["delta_val_vs_m0_dice"] >= 0.0
        and m1["delta_val_vs_m0_cldice"] > 0.0
        and m1["delta_val_vs_m0_boundary_f1"] > 0.0
    )
    m2_signal = (
        m2["delta_val_vs_m0_dice"] >= -0.001
        and (
            m2["delta_val_vs_m0_cldice"] >= 0.005
            or m2["delta_val_vs_m0_boundary_f1"] >= 0.005
        )
    )
    m3_multiseed = (
        m3["delta_val_vs_m0_dice"] >= 0.004
        and m3["val_dice"] > max(m1["val_dice"], m2["val_dice"])
        and (
            m3["delta_val_vs_m0_cldice"] > 0.0
            or m3["delta_val_vs_m0_boundary_f1"] > 0.0
        )
    )
    decision = {
        "m1_prior_signal": m1_signal,
        "m2_auxiliary_signal": m2_signal,
        "m3_multiseed_candidate": m3_multiseed,
        "recommended_next_step": "run_m0_m3_seed43_44" if m3_multiseed else "stop_cgma_and_review_factors",
    }

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with open(args.out_decision, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    print("[CGMA summary complete]")
    print(
        f"M3 val Dice={m3['val_dice']:.4f}; "
        f"delta vs M0={m3['delta_val_vs_m0_dice']:+.4f}; "
        f"next={decision['recommended_next_step']}"
    )


if __name__ == "__main__":
    main()
