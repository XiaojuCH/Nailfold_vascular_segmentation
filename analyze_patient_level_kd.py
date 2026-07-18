"""Aggregate paired per-image segmentation metrics to patient-level bootstrap CIs."""

import argparse
import csv
import json
import os
import re

import numpy as np


METRICS = ("dice", "iou", "sensitivity", "precision", "specificity", "accuracy", "hd95", "cldice", "boundary_f1")


def get_args():
    parser = argparse.ArgumentParser(description="Patient-level paired bootstrap for KD experiments.")
    parser.add_argument("--control_csv", required=True)
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--rename_mapping", default="third_party/ANFC_OURS_All_dataset/backup_original_names/rename_mapping.txt")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def read_metrics(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def load_patient_lookup(path):
    lookup = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"(ANFC_\d+\.png)\s+->\s+(.+)", line.strip())
            if not match:
                continue
            filename, original = match.groups()
            parts = os.path.splitext(os.path.basename(original))[0].split("_")
            if len(parts) < 2:
                raise ValueError(f"Cannot parse patient ID from original filename: {original}")
            lookup[filename] = "_".join(parts[:2])
    return lookup


def percentile_interval(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main():
    args = get_args()
    control = read_metrics(args.control_csv)
    candidate = read_metrics(args.candidate_csv)
    if set(control) != set(candidate):
        raise ValueError("Control/candidate per-image CSVs do not contain identical filenames")
    patients = load_patient_lookup(args.rename_mapping)
    missing = sorted(set(control) - set(patients))
    if missing:
        raise ValueError(f"Missing patient mapping for {len(missing)} files, first: {missing[:5]}")

    by_patient = {}
    for filename in sorted(control):
        by_patient.setdefault(patients[filename], []).append(filename)
    patient_ids = sorted(by_patient)
    rng = np.random.default_rng(args.seed)
    rows = []

    for metric in METRICS:
        patient_delta = np.asarray([
            np.mean([float(candidate[name][metric]) - float(control[name][metric]) for name in by_patient[patient]])
            for patient in patient_ids
        ])
        bootstrap = np.asarray([
            patient_delta[rng.integers(0, len(patient_delta), size=len(patient_delta))].mean()
            for _ in range(args.iterations)
        ])
        ci_low, ci_high = percentile_interval(bootstrap)
        rows.append({
            "metric": metric,
            "n_patients": len(patient_ids),
            "patient_mean_delta": float(patient_delta.mean()),
            "patient_std_delta": float(patient_delta.std(ddof=1)),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "improved_patients": int((patient_delta > 0).sum()) if metric != "hd95" else int((patient_delta < 0).sum()),
            "worsened_patients": int((patient_delta < 0).sum()) if metric != "hd95" else int((patient_delta > 0).sum()),
        })

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Patient bootstrap complete] {args.out_csv}")


if __name__ == "__main__":
    main()
