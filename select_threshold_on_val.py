import argparse
import csv
import json
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_vessel import VesselDataset
from evaluate_all import DATASETS, build_model, forward_logits, load_manifest, load_state_dict, safe_name, write_csv, try_write_xlsx
from utils.metrics import METRIC_KEYS, average_metric_rows, per_image_metrics_from_logits


def get_args():
    parser = argparse.ArgumentParser(
        description="Select threshold on val split and evaluate once on test split."
    )
    parser.add_argument("--manifest", default="", help="JSON manifest list or {'experiments': [...]} file.")
    parser.add_argument("--use_default_manifest", action="store_true", help="Use built-in all_filtered manifest.")
    parser.add_argument("--name", default="", help="Single-run experiment name.")
    parser.add_argument("--model_type", choices=["unet", "unet++", "transunet", "ours"], default="")
    parser.add_argument("--weight", default="", help="Single-run best_model.pth path.")
    parser.add_argument("--dataset", default="all_filtered", choices=list(DATASETS.keys()))
    parser.add_argument("--data_dir", default="", help="Custom dataset root.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--boundary_tolerance", type=int, default=2)
    parser.add_argument("--thresholds", default="0.30:0.70:0.02", help="start:end:step or comma-separated values.")
    parser.add_argument("--selection_metric", default="dice", choices=["dice", "iou", "cldice", "boundary_f1", "structure_combo"])
    parser.add_argument("--enhancer", default="basic", choices=["basic", "multiscale", "anisotropic"])
    parser.add_argument("--enhancer_norm", default="bn", choices=["bn", "none"])
    parser.add_argument("--joint_model", default="v1", choices=["v1", "v2", "gated", "boundary_refine", "decoder_distill", "dual_fusion"])
    parser.add_argument("--attention_mode", default="normal", choices=["normal", "inverse"])
    parser.add_argument("--teacher_mode", default="")
    parser.add_argument("--loss_weighting", default="fixed")
    parser.add_argument("--lambda_mse", type=float, default=10.0)
    parser.add_argument("--lambda_grad", type=float, default=30.0)
    parser.add_argument("--seg_loss", default="bce_dice")
    parser.add_argument("--cldice_weight", type=float, default=0.5)
    parser.add_argument("--boundary_weight", type=float, default=0.5)
    parser.add_argument("--focal_alpha", type=float, default=0.3)
    parser.add_argument("--focal_beta", type=float, default=0.7)
    parser.add_argument("--focal_gamma", type=float, default=0.75)
    parser.add_argument("--out_dir", default="results/threshold_selection")
    return parser.parse_args()


def parse_thresholds(text):
    text = text.strip()
    if ":" in text:
        start, end, step = [float(part) for part in text.split(":")]
        values = []
        current = start
        while current <= end + 1e-9:
            values.append(round(current, 6))
            current += step
        return values
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def make_dataset(data_dir, split, img_size):
    return VesselDataset(
        image_dir=os.path.join(data_dir, split, "images"),
        mask_dir=os.path.join(data_dir, split, "masks"),
        teacher_dir=None,
        img_size=img_size,
        augment=False,
    )


def collect_logits(model, model_type, dataset, loader, device):
    batches = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collect logits"):
            images = batch["image"].to(device)
            masks = batch["mask"].cpu()
            logits = forward_logits(model, images, model_type).detach().cpu()
            batches.append((logits, masks))
    return batches


def evaluate_cached(cached_batches, threshold, boundary_tolerance):
    rows = []
    for logits, masks in cached_batches:
        rows.extend(
            per_image_metrics_from_logits(
                logits,
                masks,
                threshold=threshold,
                boundary_tolerance=boundary_tolerance,
            )
        )
    return rows, average_metric_rows(rows)


def metric_value(avg, metric_name):
    if metric_name == "structure_combo":
        return avg["dice"] + 0.5 * avg["cldice"] + 0.5 * avg["boundary_f1"]
    return avg[metric_name]


def metadata_from_exp(exp, args, data_dir):
    return {
        "model_type": exp["model_type"],
        "weight": exp["weight"],
        "dataset": data_dir,
        "img_size": args.img_size,
        "teacher_mode": exp.get("teacher_mode", ""),
        "enhancer": exp.get("enhancer", ""),
        "enhancer_norm": exp.get("enhancer_norm", ""),
        "joint_model": exp.get("joint_model", ""),
        "attention_mode": exp.get("attention_mode", ""),
        "loss_weighting": exp.get("loss_weighting", ""),
        "lambda_mse": exp.get("lambda_mse", ""),
        "lambda_grad": exp.get("lambda_grad", ""),
        "seg_loss": exp.get("seg_loss", ""),
        "cldice_weight": exp.get("cldice_weight", ""),
        "boundary_weight": exp.get("boundary_weight", ""),
        "focal_alpha": exp.get("focal_alpha", ""),
        "focal_beta": exp.get("focal_beta", ""),
        "focal_gamma": exp.get("focal_gamma", ""),
    }


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = args.data_dir if args.data_dir else DATASETS[args.dataset]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Reuse evaluate_all manifest normalization by giving it an args-like object.
    experiments = load_manifest(args)
    thresholds = parse_thresholds(args.thresholds)

    val_dataset = make_dataset(data_dir, "val", args.img_size)
    test_dataset = make_dataset(data_dir, "test", args.img_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    aggregate_rows = []
    threshold_rows = []
    per_image_fields = [
        "experiment",
        "model_type",
        "split",
        "threshold",
        "filename",
        *METRIC_KEYS,
        "tp",
        "tn",
        "fp",
        "fn",
    ]

    for exp in experiments:
        if not os.path.exists(exp["weight"]):
            raise FileNotFoundError(f"Missing weight for {exp['name']}: {exp['weight']}")

        print(f"\n[Threshold Select] {exp['name']}")
        print(f"       model={exp['model_type']} weight={exp['weight']}")
        model = build_model(exp, args.img_size, device)
        model.load_state_dict(load_state_dict(exp["weight"], device))
        model.to(device)

        val_cached = collect_logits(model, exp["model_type"], val_dataset, val_loader, device)
        best = None
        for threshold in thresholds:
            val_rows, val_avg = evaluate_cached(val_cached, threshold, args.boundary_tolerance)
            score = metric_value(val_avg, args.selection_metric)
            row = {
                "experiment": exp["name"],
                "split": "val",
                "threshold": threshold,
                "selection_metric": args.selection_metric,
                "selection_score": score,
                "n_images": len(val_rows),
                **val_avg,
            }
            threshold_rows.append(row)
            if best is None or score > best["selection_score"]:
                best = row

        test_cached = collect_logits(model, exp["model_type"], test_dataset, test_loader, device)
        test_rows, test_avg = evaluate_cached(test_cached, best["threshold"], args.boundary_tolerance)

        per_image_rows = []
        for filename, metrics in zip(test_dataset.filenames, test_rows):
            per_image_rows.append(
                {
                    "experiment": exp["name"],
                    "model_type": exp["model_type"],
                    "split": "test",
                    "threshold": best["threshold"],
                    "filename": filename,
                    **{key: metrics[key] for key in METRIC_KEYS},
                    "tp": metrics["tp"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                }
            )

        per_image_path = os.path.join(run_dir, f"{safe_name(exp['name'])}_test_per_image.csv")
        write_csv(per_image_path, per_image_rows, per_image_fields)

        summary = {
            "experiment": exp["name"],
            **metadata_from_exp(exp, args, data_dir),
            "selection_split": "val",
            "test_split": "test",
            "selection_metric": args.selection_metric,
            "selected_threshold": best["threshold"],
            "val_selection_score": best["selection_score"],
            "val_dice_at_selected_threshold": best["dice"],
            "n_val_images": best["n_images"],
            "n_test_images": len(test_rows),
            **test_avg,
            "per_image_csv": per_image_path,
        }
        aggregate_rows.append(summary)
        print(
            f"[Selected] threshold={best['threshold']:.3f} val_{args.selection_metric}={best['selection_score']:.4f} "
            f"test Dice={test_avg['dice']:.4f}, clDice={test_avg['cldice']:.4f}, BoundaryF1={test_avg['boundary_f1']:.4f}"
        )

    threshold_fields = [
        "experiment",
        "split",
        "threshold",
        "selection_metric",
        "selection_score",
        "n_images",
        *METRIC_KEYS,
    ]
    aggregate_fields = [
        "experiment",
        "model_type",
        "weight",
        "dataset",
        "img_size",
        "teacher_mode",
        "enhancer",
        "enhancer_norm",
        "joint_model",
        "attention_mode",
        "loss_weighting",
        "lambda_mse",
        "lambda_grad",
        "seg_loss",
        "cldice_weight",
        "boundary_weight",
        "focal_alpha",
        "focal_beta",
        "focal_gamma",
        "selection_split",
        "test_split",
        "selection_metric",
        "selected_threshold",
        "val_selection_score",
        "val_dice_at_selected_threshold",
        "n_val_images",
        "n_test_images",
        *METRIC_KEYS,
        "per_image_csv",
    ]

    threshold_csv = os.path.join(run_dir, "val_threshold_sweep.csv")
    aggregate_csv = os.path.join(run_dir, "selected_threshold_test_results.csv")
    write_csv(threshold_csv, threshold_rows, threshold_fields)
    write_csv(aggregate_csv, aggregate_rows, aggregate_fields)
    aggregate_xlsx = os.path.join(run_dir, "selected_threshold_test_results.xlsx")
    wrote_xlsx = try_write_xlsx(aggregate_xlsx, aggregate_rows, aggregate_fields)

    print("\n[Threshold Selection Complete]")
    print(f"Val sweep CSV: {threshold_csv}")
    print(f"Test CSV:      {aggregate_csv}")
    if wrote_xlsx:
        print(f"Test XLSX:     {aggregate_xlsx}")


if __name__ == "__main__":
    main()
