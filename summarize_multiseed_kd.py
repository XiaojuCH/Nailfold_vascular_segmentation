"""Combine seed42 and follow-up KD runs with patient-level paired statistics."""

import argparse
import csv
import json
import os
import re

import numpy as np


METRICS = ("dice", "iou", "sensitivity", "precision", "specificity", "accuracy", "hd95", "cldice", "boundary_f1")


def get_args():
    parser = argparse.ArgumentParser(description="Summarize three-seed dual-teacher KD experiments.")
    parser.add_argument("--seed42_aggregate", required=True)
    parser.add_argument("--multiseed_aggregate", required=True)
    parser.add_argument("--decision_json", required=True)
    parser.add_argument("--rename_mapping", default="third_party/ANFC_OURS_All_dataset/backup_original_names/rename_mapping.txt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_patient_lookup(path):
    lookup = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"(ANFC_\d+\.png)\s+->\s+(.+)", line.strip())
            if match is None:
                continue
            filename, original = match.groups()
            parts = os.path.splitext(os.path.basename(original))[0].split("_")
            lookup[filename] = "_".join(parts[:2])
    return lookup


def extract_seed(name):
    match = re.search(r"_seed(\d+)$", name)
    if match is None:
        raise ValueError(f"Experiment name lacks trailing seed: {name}")
    return int(match.group(1))


def per_image_by_name(path):
    return {row["filename"]: row for row in read_csv(path)}


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def patient_deltas(control_csv, candidate_csv, patient_lookup, metric):
    control = per_image_by_name(control_csv)
    candidate = per_image_by_name(candidate_csv)
    if set(control) != set(candidate):
        raise ValueError("Control/candidate per-image files differ")
    grouped = {}
    for filename in control:
        patient = patient_lookup.get(filename)
        if patient is None:
            raise ValueError(f"Missing patient mapping for {filename}")
        grouped.setdefault(patient, []).append(float(candidate[filename][metric]) - float(control[filename][metric]))
    return {patient: float(np.mean(values)) for patient, values in grouped.items()}


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.decision_json, encoding="utf-8") as handle:
        decision = json.load(handle)
    winning_name = decision["best_candidate"]
    winning_base = re.sub(r"_seed\d+$", "", winning_name)
    control_base = re.sub(r"_seed\d+$", "", decision["control_name"])

    combined_rows = read_csv(args.seed42_aggregate) + read_csv(args.multiseed_aggregate)
    selected = [
        row for row in combined_rows
        if re.sub(r"_seed\d+$", "", row["experiment"]) in (control_base, winning_base)
    ]
    controls = {extract_seed(row["experiment"]): row for row in selected if re.sub(r"_seed\d+$", "", row["experiment"]) == control_base}
    candidates = {extract_seed(row["experiment"]): row for row in selected if re.sub(r"_seed\d+$", "", row["experiment"]) == winning_base}
    seeds = sorted(set(controls) & set(candidates))
    if len(seeds) < 2:
        raise ValueError(f"Need at least two paired seeds, found: {seeds}")

    summary_rows = []
    for label, rows in (("control", [controls[seed] for seed in seeds]), ("candidate", [candidates[seed] for seed in seeds])):
        summary = {"group": label, "n_seeds": len(seeds), "seeds": ",".join(map(str, seeds))}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in rows])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    delta_summary = {"group": "candidate_minus_control", "n_seeds": len(seeds), "seeds": ",".join(map(str, seeds))}
    for metric in METRICS:
        deltas = np.asarray([float(candidates[seed][metric]) - float(controls[seed][metric]) for seed in seeds])
        delta_summary[f"{metric}_mean"] = float(deltas.mean())
        delta_summary[f"{metric}_std"] = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    summary_rows.append(delta_summary)
    summary_fields = ["group", "n_seeds", "seeds", *[f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]]
    write_csv(os.path.join(args.out_dir, "multiseed_metric_summary.csv"), summary_rows, summary_fields)

    patient_lookup = load_patient_lookup(args.rename_mapping)
    rng = np.random.default_rng(args.seed)
    patient_rows = []
    for metric in METRICS:
        per_seed = [
            patient_deltas(controls[seed]["per_image_csv"], candidates[seed]["per_image_csv"], patient_lookup, metric)
            for seed in seeds
        ]
        patient_ids = sorted(per_seed[0])
        if any(sorted(seed_values) != patient_ids for seed_values in per_seed[1:]):
            raise ValueError("Patient identities differ between seeds")
        averaged_delta = np.asarray([np.mean([seed_values[patient] for seed_values in per_seed]) for patient in patient_ids])
        bootstrap = np.asarray([
            averaged_delta[rng.integers(0, len(averaged_delta), len(averaged_delta))].mean()
            for _ in range(args.iterations)
        ])
        patient_rows.append({
            "metric": metric,
            "n_patients": len(patient_ids),
            "mean_delta": float(averaged_delta.mean()),
            "std_delta": float(averaged_delta.std(ddof=1)),
            "ci95_low": float(np.percentile(bootstrap, 2.5)),
            "ci95_high": float(np.percentile(bootstrap, 97.5)),
            "improved_patients": int((averaged_delta < 0).sum()) if metric == "hd95" else int((averaged_delta > 0).sum()),
            "worsened_patients": int((averaged_delta > 0).sum()) if metric == "hd95" else int((averaged_delta < 0).sum()),
        })
    write_csv(
        os.path.join(args.out_dir, "patient_level_bootstrap.csv"),
        patient_rows,
        ["metric", "n_patients", "mean_delta", "std_delta", "ci95_low", "ci95_high", "improved_patients", "worsened_patients"],
    )
    print(f"[Multiseed KD summary complete] {args.out_dir}")


if __name__ == "__main__":
    main()
