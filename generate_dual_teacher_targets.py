"""Generate float16 ensemble probability targets from two sequential teachers."""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_vessel import VesselDataset
from utils.metrics import METRIC_KEYS, average_metric_rows, binary_metrics_from_masks
from models.green_prior_fusion import GreenPriorFusionModel, PRIOR_FUSION_VARIANTS
from models.transunet_official import TransUNetOfficial


def get_args():
    parser = argparse.ArgumentParser(description="Generate sequential dual-teacher soft targets.")
    parser.add_argument("--data_dir", default="./dataset_all_filtered")
    parser.add_argument("--splits", default="train,val", help="Comma-separated dataset splits.")
    parser.add_argument("--f0_weight", required=True)
    parser.add_argument("--f3_weight", required=True)
    parser.add_argument("--f3_variant", default="directional_multiscale", choices=PRIOR_FUSION_VARIANTS)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_f0(weight, img_size, device):
    model = TransUNetOfficial(n_channels=3, n_classes=1, img_size=img_size).to(device)
    model.load_state_dict(load_state_dict(weight, device))
    model.eval()
    return model


def build_f3(weight, variant, img_size, device):
    segmentor = TransUNetOfficial(n_channels=3, n_classes=1, img_size=img_size)
    model = GreenPriorFusionModel(segmentor, variant=variant).to(device)
    model.load_state_dict(load_state_dict(weight, device))
    model.eval()
    return model


def ensure_probability(array, label, filenames):
    if not torch.isfinite(array).all():
        raise ValueError(f"{label} generated NaN/Inf for {filenames}")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < -1e-6 or maximum > 1.000001:
        raise ValueError(f"{label} outside [0, 1] for {filenames}: [{minimum}, {maximum}]")


def save_array(path, array, overwrite, dtype=np.float16):
    if os.path.exists(path) and not overwrite:
        existing = np.load(path, allow_pickle=False)
        if existing.shape != array.shape:
            raise ValueError(f"Existing target has wrong shape: {path}: {existing.shape} != {array.shape}")
        return
    np.save(path, array.astype(dtype, copy=False), allow_pickle=False)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_metric_rows(rows, filenames, predictions, targets):
    for filename, prediction, target in zip(filenames, predictions, targets):
        metrics = binary_metrics_from_masks(prediction, target)
        rows.append({
            "filename": filename,
            **{key: metrics[key] for key in METRIC_KEYS},
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        })


def generate_f0_cache(args, split, f0, device):
    dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, split, "images"),
        mask_dir=os.path.join(args.data_dir, split, "masks"),
        img_size=args.img_size,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    cache_dir = os.path.join(args.out_dir, split, "_f0_probabilities")
    os.makedirs(cache_dir, exist_ok=True)
    rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"Generate {split} F0 cache")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            filenames = dataset.filenames[batch_index * args.batch_size : batch_index * args.batch_size + images.shape[0]]
            probs = torch.sigmoid(f0(images))
            ensure_probability(probs, "F0 probability", filenames)
            probability_np = probs[:, 0].cpu().numpy()
            for filename, probability in zip(filenames, probability_np):
                save_array(
                    os.path.join(cache_dir, os.path.splitext(filename)[0] + ".npy"),
                    probability,
                    args.overwrite,
                    dtype=np.float32,
                )
            predictions = probability_np > args.threshold
            targets = (masks > 0.5).cpu().numpy().astype(bool)
            append_metric_rows(rows, filenames, predictions, targets[:, 0])
    return dataset, loader, cache_dir, rows


def generate_split_targets(args, split, f3, device):
    dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, split, "images"),
        mask_dir=os.path.join(args.data_dir, split, "masks"),
        img_size=args.img_size,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    split_root = os.path.join(args.out_dir, split)
    f0_cache_dir = os.path.join(split_root, "_f0_probabilities")
    probability_dir = os.path.join(split_root, "ensemble_probabilities")
    disagreement_dir = os.path.join(split_root, "disagreement")
    if not os.path.isdir(f0_cache_dir):
        raise FileNotFoundError(f"Missing F0 cache for {split}: {f0_cache_dir}")
    os.makedirs(probability_dir, exist_ok=True)
    os.makedirs(disagreement_dir, exist_ok=True)

    f3_rows = []
    ensemble_rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"Generate {split} targets")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            filenames = dataset.filenames[batch_index * args.batch_size : batch_index * args.batch_size + images.shape[0]]
            p_f3 = torch.sigmoid(f3(images))
            f0_arrays = [
                np.load(os.path.join(f0_cache_dir, os.path.splitext(filename)[0] + ".npy"), allow_pickle=False)
                for filename in filenames
            ]
            p_f0 = torch.from_numpy(np.stack(f0_arrays).astype(np.float32, copy=False)).to(device).unsqueeze(1)
            ensemble = 0.5 * (p_f0 + p_f3)
            disagreement = torch.abs(p_f0 - p_f3)
            ensure_probability(p_f3, "F3 probability", filenames)
            ensure_probability(ensemble, "Ensemble probability", filenames)
            ensure_probability(disagreement, "Teacher disagreement", filenames)

            for index, filename in enumerate(filenames):
                stem = os.path.splitext(filename)[0]
                probability = ensemble[index, 0].detach().cpu().numpy()
                difference = disagreement[index, 0].detach().cpu().numpy()
                if probability.shape != (args.img_size, args.img_size):
                    raise ValueError(f"Unexpected target shape for {filename}: {probability.shape}")
                save_array(os.path.join(probability_dir, stem + ".npy"), probability, args.overwrite)
                save_array(os.path.join(disagreement_dir, stem + ".npy"), difference, args.overwrite)

            targets = (masks > 0.5).cpu().numpy().astype(bool)[:, 0]
            append_metric_rows(f3_rows, filenames, p_f3[:, 0].cpu().numpy() > args.threshold, targets)
            append_metric_rows(ensemble_rows, filenames, ensemble[:, 0].cpu().numpy() > args.threshold, targets)

    expected = len(dataset)
    actual_probability = len([name for name in os.listdir(probability_dir) if name.endswith(".npy")])
    actual_disagreement = len([name for name in os.listdir(disagreement_dir) if name.endswith(".npy")])
    if actual_probability != expected or actual_disagreement != expected:
        raise RuntimeError(
            f"{split} target count mismatch: expected={expected}, probability={actual_probability}, disagreement={actual_disagreement}"
        )
    return {
        "split": split,
        "images": expected,
        "probability_dir": probability_dir,
        "disagreement_dir": disagreement_dir,
    }, f3_rows, ensemble_rows, f0_cache_dir


def main():
    args = get_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    for path in (args.f0_weight, args.f3_weight):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing teacher weight: {path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits or any(split not in ("train", "val", "test") for split in splits):
        raise ValueError("splits must be a non-empty subset of train,val,test")
    os.makedirs(args.out_dir, exist_ok=True)

    # Cache F0 outputs first, then free it before F3 is put on GPU.
    f0 = build_f0(args.f0_weight, args.img_size, device)
    f0_results = {}
    for split in splits:
        _, _, _, f0_results[split] = generate_f0_cache(args, split, f0, device)
    del f0
    if device == "cuda":
        torch.cuda.empty_cache()

    f3 = build_f3(args.f3_weight, args.f3_variant, args.img_size, device)
    records = []
    f3_results = {}
    ensemble_results = {}
    cache_dirs = []
    for split in splits:
        record, f3_results[split], ensemble_results[split], cache_dir = generate_split_targets(args, split, f3, device)
        records.append(record)
        cache_dirs.append(cache_dir)
    del f3
    if device == "cuda":
        torch.cuda.empty_cache()

    # F0 caches are float32 intermediates; only float16 ensemble targets are needed for KD.
    for cache_dir in cache_dirs:
        shutil.rmtree(cache_dir)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "data_dir": os.path.abspath(args.data_dir),
        "f0_weight": os.path.abspath(args.f0_weight),
        "f3_weight": os.path.abspath(args.f3_weight),
        "f3_variant": args.f3_variant,
        "ensemble": "0.5 * sigmoid(F0) + 0.5 * sigmoid(F3)",
        "dtype": "float16 .npy",
        "img_size": args.img_size,
        "splits": records,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    evaluation_rows = []
    for split in splits:
        split_dir = os.path.join(args.out_dir, split)
        for model_name, rows in (("F0", f0_results[split]), ("F3", f3_results[split]), ("ensemble", ensemble_results[split])):
            write_csv(
                os.path.join(split_dir, f"{model_name.lower()}_per_image.csv"),
                rows,
                ["filename", *METRIC_KEYS, "tp", "tn", "fp", "fn"],
            )
            evaluation_rows.append({
                "split": split,
                "model": model_name,
                "n_images": len(rows),
                **average_metric_rows(rows),
            })
    write_csv(os.path.join(args.out_dir, "teacher_ensemble_metrics.csv"), evaluation_rows, ["split", "model", "n_images", *METRIC_KEYS])

    print("[Dual-teacher targets complete]")
    print(f"Output: {args.out_dir}")
    for record in records:
        print(f"{record['split']}: {record['images']} images")


if __name__ == "__main__":
    main()
