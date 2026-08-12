"""Generate a full-dataset visual error review for segmentation checkpoints.

The script intentionally reuses evaluate_all.py model construction so that visual
review and unified evaluation load exactly the same architectures and weights.
"""

import argparse
import csv
import html
import json
import os
import re
import shutil
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import label
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_vessel import VesselDataset
from evaluate_all import build_model, forward_logits, load_state_dict, safe_name
from utils.metrics import METRIC_KEYS, average_metric_rows, binary_metrics_from_masks


def parse_args():
    parser = argparse.ArgumentParser(description="Visual error analysis for nailfold segmentation models.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data_dir", default="./dataset_all_filtered")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out_dir", default="results/prediction_error_review_20260730")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary_tolerance", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--panel_size", type=int, default=192)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--reuse_predictions", action="store_true")
    parser.add_argument("--skip_all_case_panels", action="store_true")
    parser.add_argument("--patient_mapping", default="third_party/ANFC_OURS_All_dataset/backup_original_names/rename_mapping.txt")
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_patient_lookup(path):
    if not path or not os.path.isfile(path):
        return {}
    lookup = {}
    with open(path, encoding="utf-8") as handle:
        for line_text in handle:
            match = re.match(r"(ANFC_\d+\.png)\s+->\s+(.+)", line_text.strip())
            if not match:
                continue
            filename, original = match.groups()
            parts = os.path.splitext(os.path.basename(original))[0].split("_")
            if len(parts) >= 2:
                lookup[filename] = "_".join(parts[:2])
    return lookup


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def normalize_experiment(exp):
    item = dict(exp)
    item.setdefault("display_name", item["name"])
    item.setdefault("panel", True)
    item.setdefault("enhancer", "basic")
    item.setdefault("enhancer_norm", "bn")
    item.setdefault("joint_model", "v1")
    item.setdefault("attention_mode", "normal")
    item.setdefault("prior_fusion_variant", "plain_single")
    item.setdefault("cgma_prior", "on")
    item.setdefault("cgma_auxiliary", "on")
    return item


def cache_path(out_dir, name):
    return os.path.join(out_dir, "probability_cache", safe_name(name) + ".npz")


def save_prediction_pngs(out_dir, exp, filenames, probabilities, threshold):
    model_dir = os.path.join(out_dir, "predictions", safe_name(exp["name"]))
    os.makedirs(model_dir, exist_ok=True)
    for filename, probability in zip(filenames, probabilities):
        prediction = (probability >= threshold).astype(np.uint8) * 255
        if not cv2.imwrite(os.path.join(model_dir, filename), prediction):
            raise RuntimeError(f"Failed to save prediction: {filename}")


def load_cached_probabilities(path, filenames, img_size):
    if not os.path.isfile(path):
        return None
    payload = np.load(path, allow_pickle=False)
    cached_names = [str(item) for item in payload["filenames"]]
    probabilities = payload["probabilities"]
    if cached_names != list(filenames):
        return None
    if probabilities.shape != (len(filenames), img_size, img_size):
        return None
    return probabilities.astype(np.float32)


def infer_experiment(exp, loader, filenames, args, device):
    path = cache_path(args.out_dir, exp["name"])
    if args.reuse_predictions:
        cached = load_cached_probabilities(path, filenames, args.img_size)
        if cached is not None:
            print(f"[Cache] {exp['name']}")
            return cached

    weight = exp.get("weight", "")
    if not os.path.isfile(weight):
        raise FileNotFoundError(f"Missing weight for {exp['name']}: {weight}")

    print(f"\n[Predict] {exp['name']}\n          {weight}")
    model = build_model(exp, args.img_size, device)
    model.load_state_dict(load_state_dict(weight, device))
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=exp["display_name"]):
            images = batch["image"].to(device)
            logits = forward_logits(model, images, exp["model_type"])
            probability = torch.sigmoid(logits)[:, 0].cpu().numpy()
            chunks.append(probability.astype(np.float16))
    probabilities = np.concatenate(chunks, axis=0)
    if probabilities.shape != (len(filenames), args.img_size, args.img_size):
        raise ValueError(f"Unexpected probability shape for {exp['name']}: {probabilities.shape}")
    if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError(f"Invalid probabilities generated by {exp['name']}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, filenames=np.asarray(filenames), probabilities=probabilities)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities.astype(np.float32)


def build_ensemble(exp, predictions):
    members = exp.get("members", [])
    if not members:
        raise ValueError(f"Ensemble has no members: {exp['name']}")
    total_weight = sum(float(member.get("weight", 1.0)) for member in members)
    if total_weight <= 0:
        raise ValueError(f"Ensemble weights must sum to a positive number: {exp['name']}")
    probability = None
    for member in members:
        member_name = member["name"]
        if member_name not in predictions:
            raise KeyError(f"Ensemble member has not been predicted: {member_name}")
        contribution = predictions[member_name] * (float(member.get("weight", 1.0)) / total_weight)
        probability = contribution if probability is None else probability + contribution
    return probability.astype(np.float32)


def read_images_and_masks(data_dir, split, filenames, img_size):
    images = []
    masks = []
    image_dir = os.path.join(data_dir, split, "images")
    mask_dir = os.path.join(data_dir, split, "masks")
    for filename in tqdm(filenames, desc="Load images/masks"):
        image = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Cannot load image or mask: {filename}")
        image = cv2.cvtColor(cv2.resize(image, (img_size, img_size)), cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST) > 127
        images.append(image)
        masks.append(mask)
    return np.stack(images), np.stack(masks)


def image_diagnostics(image, mask):
    green = image[:, :, 1].astype(np.float32) / 255.0
    local_dark = np.maximum(cv2.GaussianBlur(green, (0, 0), 9) - green, 0.0)
    components = int(label(mask)[1])
    return {
        "gt_area_fraction": float(mask.mean()),
        "gt_components": components,
        "mean_green": float(green.mean()),
        "green_std": float(green.std()),
        "green_local_contrast_mean": float(local_dark.mean()),
        "green_local_contrast_on_vessel": float(local_dark[mask].mean()) if mask.any() else 0.0,
    }


def metric_rows(experiments, filenames, images, masks, predictions, args, patient_lookup):
    rows = []
    diagnostics = {filename: image_diagnostics(image, mask) for filename, image, mask in zip(filenames, images, masks)}
    for exp in experiments:
        probabilities = predictions[exp["name"]]
        for filename, target, probability in tqdm(
            zip(filenames, masks, probabilities), total=len(filenames), desc=f"Metrics {exp['display_name']}"
        ):
            prediction = probability >= args.threshold
            metrics = binary_metrics_from_masks(
                prediction, target, boundary_tolerance=args.boundary_tolerance
            )
            pred_components = int(label(prediction)[1])
            gt_pixels = metrics["tp"] + metrics["fn"]
            rows.append({
                "experiment": exp["name"],
                "display_name": exp["display_name"],
                "model_type": exp["model_type"],
                "filename": filename,
                "patient_id": patient_lookup.get(filename, "unknown"),
                **{key: metrics[key] for key in METRIC_KEYS},
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "pred_area_fraction": float(prediction.mean()),
                "pred_to_gt_area_ratio": float(prediction.sum() / max(gt_pixels, 1.0)),
                "fn_fraction_of_gt": float(metrics["fn"] / max(gt_pixels, 1.0)),
                "fp_fraction_of_gt": float(metrics["fp"] / max(gt_pixels, 1.0)),
                "pred_components": pred_components,
                "component_delta": pred_components - diagnostics[filename]["gt_components"],
                **diagnostics[filename],
            })
    return rows


def aggregate_rows(experiments, rows):
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["experiment"]].append(row)
    output = []
    for exp in experiments:
        model_rows = by_model[exp["name"]]
        metrics = average_metric_rows(model_rows)
        output.append({
            "experiment": exp["name"],
            "display_name": exp["display_name"],
            "model_type": exp["model_type"],
            "n_images": len(model_rows),
            **metrics,
            "mean_pred_to_gt_area_ratio": float(np.mean([row["pred_to_gt_area_ratio"] for row in model_rows])),
            "mean_fn_fraction_of_gt": float(np.mean([row["fn_fraction_of_gt"] for row in model_rows])),
            "mean_fp_fraction_of_gt": float(np.mean([row["fp_fraction_of_gt"] for row in model_rows])),
        })
    return output


def comparison_rows(comparisons, rows):
    by_key = {(row["experiment"], row["filename"]): row for row in rows}
    output = []
    deltas = {}
    for comparison in comparisons:
        control = comparison["control"]
        candidate = comparison["candidate"]
        name = comparison.get("name", f"{candidate}_vs_{control}")
        model_deltas = []
        for filename in sorted({row["filename"] for row in rows if row["experiment"] == control}):
            control_row = by_key[(control, filename)]
            candidate_row = by_key[(candidate, filename)]
            item = {
                "comparison": name,
                "control": control,
                "candidate": candidate,
                "filename": filename,
                "patient_id": control_row["patient_id"],
            }
            for metric in METRIC_KEYS:
                item[f"delta_{metric}"] = candidate_row[metric] - control_row[metric]
            model_deltas.append(item)
        deltas[name] = model_deltas
        dice_delta = np.asarray([item["delta_dice"] for item in model_deltas])
        output.append({
            "comparison": name,
            "control": control,
            "candidate": candidate,
            "mean_delta_dice": float(dice_delta.mean()),
            "median_delta_dice": float(np.median(dice_delta)),
            "improved_images": int((dice_delta > 0).sum()),
            "worsened_images": int((dice_delta < 0).sum()),
            "strong_improvements_delta_gt_0p05": int((dice_delta >= 0.05).sum()),
            "strong_regressions_delta_lt_minus_0p05": int((dice_delta <= -0.05).sum()),
            **{
                f"mean_delta_{metric}": float(np.mean([item[f"delta_{metric}"] for item in model_deltas]))
                for metric in METRIC_KEYS if metric != "dice"
            },
        })
    return output, deltas


def patient_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["experiment"], row["display_name"], row["patient_id"])].append(row)
    output = []
    for (experiment, display_name, patient_id), group in sorted(groups.items()):
        output.append({
            "experiment": experiment,
            "display_name": display_name,
            "patient_id": patient_id,
            "n_images": len(group),
            **{metric: float(np.mean([row[metric] for row in group])) for metric in METRIC_KEYS},
        })
    return output


def overlay_error(image, prediction, target):
    base = (image.astype(np.float32) * 0.52).astype(np.uint8)
    overlay = base.copy()
    tp = prediction & target
    fp = prediction & ~target
    fn = ~prediction & target
    overlay[tp] = (35, 220, 90)
    overlay[fp] = (245, 70, 65)
    overlay[fn] = (55, 145, 255)
    return cv2.addWeighted(base, 0.35, overlay, 0.65, 0)


def gt_overlay(image, target):
    output = image.copy()
    contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (30, 235, 235), 2)
    return output


def tile(image, title, subtitle, size):
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    canvas = np.full((size + 46, size, 3), 247, dtype=np.uint8)
    canvas[46:] = resized
    cv2.putText(canvas, title[:25], (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 28, 38), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle[:31], (7, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (65, 72, 82), 1, cv2.LINE_AA)
    return canvas


def render_case_panel(path, filename, image, target, panel_experiments, predictions, metric_lookup, size):
    panels = [
        tile(image, "RGB input", filename, size),
        tile(gt_overlay(image, target), "Ground truth", "cyan contour", size),
    ]
    for exp in panel_experiments:
        probability = predictions[exp["name"]][metric_lookup["index"]]
        prediction = probability >= metric_lookup["threshold"]
        row = metric_lookup[(exp["name"], filename)]
        subtitle = f"D {row['dice']:.3f} P {row['precision']:.3f} R {row['sensitivity']:.3f}"
        panels.append(tile(overlay_error(image, prediction, target), exp["display_name"], subtitle, size))
    canvas = np.concatenate(panels, axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def rank_cases(manifest, experiments, rows, deltas, predictions, filenames, top_k):
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["experiment"]].append(row)
    categories = []
    primary = manifest["primary_baseline"]
    primary_rows = by_model[primary]
    categories.extend([
        ("baseline_best", "Baseline best", sorted(primary_rows, key=lambda row: row["dice"], reverse=True)[:top_k]),
        ("baseline_worst", "Baseline worst", sorted(primary_rows, key=lambda row: row["dice"])[:top_k]),
        ("baseline_high_fn", "Baseline highest FN", sorted(primary_rows, key=lambda row: row["fn_fraction_of_gt"], reverse=True)[:top_k]),
        ("baseline_high_fp", "Baseline highest FP", sorted(primary_rows, key=lambda row: row["fp_fraction_of_gt"], reverse=True)[:top_k]),
        ("baseline_high_hd95", "Baseline largest HD95", sorted(primary_rows, key=lambda row: row["hd95"], reverse=True)[:top_k]),
    ])
    for comparison in manifest.get("comparisons", []):
        name = comparison.get("name", f"{comparison['candidate']}_vs_{comparison['control']}")
        comparison_deltas = deltas[name]
        candidates = {row["filename"]: row for row in by_model[comparison["candidate"]]}
        gains = sorted(comparison_deltas, key=lambda row: row["delta_dice"], reverse=True)[:top_k]
        losses = sorted(comparison_deltas, key=lambda row: row["delta_dice"])[:top_k]
        categories.append((safe_name(name) + "_wins", name + " largest wins", [candidates[item["filename"]] for item in gains]))
        categories.append((safe_name(name) + "_losses", name + " largest losses", [candidates[item["filename"]] for item in losses]))

    panel_names = [exp["name"] for exp in experiments if exp.get("panel", True)]
    stack = np.stack([predictions[name] for name in panel_names], axis=0)
    disagreement = np.mean(np.std(stack, axis=0), axis=(1, 2))
    disagreement_rows = [dict(primary_rows[index], disagreement=float(disagreement[index])) for index in range(len(filenames))]
    categories.append(("model_disagreement", "Largest model disagreement", sorted(disagreement_rows, key=lambda row: row["disagreement"], reverse=True)[:top_k]))
    return categories, disagreement


def plot_metric_distributions(out_dir, experiments, rows):
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["experiment"]].append(row)
    labels = [exp["display_name"] for exp in experiments]
    values = [[row["dice"] for row in by_model[exp["name"]]] for exp in experiments]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.25), 5.5))
    box = ax.boxplot(values, labels=labels, patch_artist=True, showfliers=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(box["boxes"])))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Per-image Dice")
    ax.set_title("Development-test Dice distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "dice_distribution.png"), dpi=180)
    plt.close(fig)


def plot_comparisons(out_dir, comparison_summaries, deltas):
    for summary in comparison_summaries:
        name = summary["comparison"]
        values = np.asarray([row["delta_dice"] for row in deltas[name]])
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        ax.hist(values, bins=35, color="#2878b5", alpha=0.86)
        ax.axvline(0, color="#20242a", linewidth=1.2)
        ax.axvline(values.mean(), color="#d94841", linewidth=1.5, linestyle="--", label=f"mean {values.mean():+.4f}")
        ax.set_xlabel("Candidate - control per-image Dice")
        ax.set_ylabel("Images")
        ax.set_title(name)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", safe_name(name) + "_delta_hist.png"), dpi=180)
        plt.close(fig)


def plot_difficulty(out_dir, primary_name, rows):
    primary = [row for row in rows if row["experiment"] == primary_name]
    x_area = np.asarray([row["gt_area_fraction"] for row in primary])
    x_contrast = np.asarray([row["green_local_contrast_on_vessel"] for row in primary])
    y = np.asarray([row["dice"] for row in primary])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].scatter(x_area, y, s=14, alpha=0.55, color="#2878b5")
    axes[0].set_xlabel("GT vessel area fraction")
    axes[0].set_ylabel("Baseline Dice")
    axes[1].scatter(x_contrast, y, s=14, alpha=0.55, color="#d97706")
    axes[1].set_xlabel("Green local contrast on vessel")
    axes[1].set_ylabel("Baseline Dice")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Image difficulty diagnostics")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "difficulty_diagnostics.png"), dpi=180)
    plt.close(fig)


def html_table(rows, fields, precision=4):
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in fields)
    body = []
    for row in rows:
        cells = []
        for key, _ in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.{precision}f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html_report(out_dir, manifest, aggregate, comparisons, categories, rows):
    aggregate_table = html_table(aggregate, [
        ("display_name", "Model"), ("dice", "Dice"), ("iou", "IoU"),
        ("sensitivity", "Recall"), ("precision", "Precision"), ("hd95", "HD95"),
        ("cldice", "clDice"), ("boundary_f1", "Boundary F1"),
    ])
    comparison_table = html_table(comparisons, [
        ("comparison", "Comparison"), ("mean_delta_dice", "Mean delta Dice"),
        ("median_delta_dice", "Median delta"), ("improved_images", "Improved"),
        ("worsened_images", "Worsened"), ("mean_delta_hd95", "Delta HD95"),
        ("mean_delta_cldice", "Delta clDice"), ("mean_delta_boundary_f1", "Delta Boundary F1"),
    ])
    galleries = []
    for key, title, category_rows in categories:
        cards = []
        for row in category_rows:
            filename = row["filename"]
            stem = os.path.splitext(filename)[0]
            path = f"all_cases/{stem}.jpg"
            extra = f"Dice {row['dice']:.3f} | HD95 {row['hd95']:.1f}"
            cards.append(f"<a class='card' href='{path}'><img loading='lazy' src='{path}'><span>{html.escape(filename)}<br>{extra}</span></a>")
        galleries.append(f"<section><h2>{html.escape(title)}</h2><div class='gallery'>{''.join(cards)}</div></section>")

    primary = manifest["primary_baseline"]
    primary_rows = sorted([row for row in rows if row["experiment"] == primary], key=lambda row: row["dice"])
    patient_groups = defaultdict(list)
    for row in primary_rows:
        patient_groups[row["patient_id"]].append(row["dice"])
    patient_means = {
        patient_id: float(np.mean(values))
        for patient_id, values in patient_groups.items()
    }
    case_rows = []
    for row in primary_rows:
        filename = row["filename"]
        stem = os.path.splitext(filename)[0]
        case_rows.append({**row, "filename_link": f"<a href='all_cases/{stem}.jpg'>{html.escape(filename)}</a>"})
    case_head = "".join(f"<th>{label}</th>" for label in ("Filename", "Patient", "Dice", "Recall", "Precision", "HD95", "GT area"))
    case_body = "".join(
        "<tr>"
        f"<td>{row['filename_link']}</td><td>{html.escape(str(row['patient_id']))}</td>"
        f"<td>{row['dice']:.4f}</td><td>{row['sensitivity']:.4f}</td><td>{row['precision']:.4f}</td>"
        f"<td>{row['hd95']:.2f}</td><td>{row['gt_area_fraction']:.4f}</td></tr>"
        for row in case_rows
    )
    page = f"""<!doctype html>
<html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>甲襞分割预测误差复盘</title>
<style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f4f6f8;color:#18212b}}
main{{max-width:1480px;margin:auto;padding:30px}} h1{{margin-bottom:8px}} .note{{background:#fff3cd;border-left:4px solid #d49b00;padding:12px 16px}}
table{{border-collapse:collapse;width:100%;background:white;margin:12px 0 28px;font-size:14px}} th,td{{border:1px solid #d9dee5;padding:7px 9px;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#eaf0f6}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(285px,1fr));gap:12px}} .card{{background:white;color:#18212b;text-decoration:none;border:1px solid #dce2e8;padding:7px}}
.card img{{width:100%;display:block;margin-bottom:6px}} .figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}} .figures img{{width:100%;background:white}}
.legend span{{display:inline-block;margin-right:18px}} .dot{{width:12px;height:12px;display:inline-block;margin-right:5px}}
</style></head><body><main>
<h1>甲襞毛细血管分割：全测试集预测与失败模式复盘</h1>
<p>数据：{html.escape(manifest.get('dataset_note','dataset_all_filtered/test'))}；阈值固定 0.5。当前 test 已参与研发决策，因此本报告称其为 development-test。</p>
<p class='note'>误差图颜色：<span class='legend'><span><i class='dot' style='background:#23dc5a'></i>TP</span><span><i class='dot' style='background:#f54641'></i>FP</span><span><i class='dot' style='background:#3791ff'></i>FN</span></span>。病例图用于诊断，不替代独立测试或患者级外层验证。</p>
<h2>整体指标</h2>{aggregate_table}
<h2>相对基线的逐图变化</h2>{comparison_table}
<h2>总体图表</h2><div class='figures'><img src='figures/dice_distribution.png'><img src='figures/difficulty_diagnostics.png'></div>
{''.join(galleries)}
<section><h2>全部 436 张图（按主基线 Dice 从低到高）</h2><table><thead><tr>{case_head}</tr></thead><tbody>{case_body}</tbody></table></section>
</main></body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as handle:
        handle.write(page)


def write_markdown_summary(out_dir, manifest, aggregate, comparisons, categories, rows):
    by_name = {row["experiment"]: row for row in aggregate}
    primary = manifest["primary_baseline"]
    primary_rows = [row for row in rows if row["experiment"] == primary]
    patient_groups = defaultdict(list)
    for row in primary_rows:
        patient_groups[row["patient_id"]].append(row["dice"])
    patient_means = {
        patient_id: float(np.mean(values))
        for patient_id, values in patient_groups.items()
    }
    area = np.asarray([row["gt_area_fraction"] for row in primary_rows])
    contrast = np.asarray([row["green_local_contrast_on_vessel"] for row in primary_rows])
    dice = np.asarray([row["dice"] for row in primary_rows])
    area_corr = float(np.corrcoef(area, dice)[0, 1])
    contrast_corr = float(np.corrcoef(contrast, dice)[0, 1])
    lines = [
        "# 预测可视化复盘摘要",
        "",
        "> 注意：`dataset_all_filtered/test` 已参与方法研发，本文档统一称为 development-test。",
        "",
        "## 整体结果",
        "",
        "| 模型 | Dice | Recall | Precision | HD95 | clDice | Boundary F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in manifest["experiments"]:
        row = by_name[exp["name"]]
        lines.append(
            f"| {exp.get('display_name', exp['name'])} | {row['dice']:.4f} | {row['sensitivity']:.4f} | "
            f"{row['precision']:.4f} | {row['hd95']:.2f} | {row['cldice']:.4f} | {row['boundary_f1']:.4f} |"
        )
    lines.extend(["", "## 相对变化", "", "| 比较 | mean Dice delta | 改善图数 | 退化图数 | HD95 delta | clDice delta | Boundary F1 delta |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in comparisons:
        lines.append(
            f"| {row['comparison']} | {row['mean_delta_dice']:+.4f} | {row['improved_images']} | {row['worsened_images']} | "
            f"{row['mean_delta_hd95']:+.2f} | {row['mean_delta_cldice']:+.4f} | {row['mean_delta_boundary_f1']:+.4f} |"
        )
    worst = sorted(primary_rows, key=lambda row: row["dice"])[:10]
    patient_dice = sorted(patient_means.items(), key=lambda item: item[1])
    lines.extend([
        "",
        "## 初步诊断",
        "",
        f"- 主基线逐图 Dice 与 GT 血管面积占比的 Pearson 相关系数为 `{area_corr:+.3f}`；与血管区域 green local contrast 的相关系数为 `{contrast_corr:+.3f}`。",
        f"- 主基线平均 Recall 为 `{by_name[primary]['sensitivity']:.4f}`，Precision 为 `{by_name[primary]['precision']:.4f}`。结合高 FN/FP 排名图判断主要错误是漏检、过分割还是两者并存。",
        "- 不能只展示 Ours 获胜病例；`*_wins.csv` 与 `*_losses.csv` 同时保留，便于判断提升是否集中于某种图像条件。",
        "- HTML 报告中的全部病例按主基线 Dice 排序，可逐张打开大图查看 TP/FP/FN。",
        "",
        "## 主基线最差 10 张",
        "",
        "| 文件 | 患者 | Dice | Recall | Precision | HD95 | GT area |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in worst:
        lines.append(
            f"| {row['filename']} | {row['patient_id']} | {row['dice']:.4f} | {row['sensitivity']:.4f} | "
            f"{row['precision']:.4f} | {row['hd95']:.2f} | {row['gt_area_fraction']:.4f} |"
        )
    lines.extend([
        "",
        "## 主基线最难患者",
        "",
        "| 患者 | 图像数 | 平均 Dice |",
        "|---|---:|---:|",
    ])
    for patient_id, mean_dice in patient_dice[:8]:
        lines.append(f"| {patient_id} | {len(patient_groups[patient_id])} | {mean_dice:.4f} |")
    lines.extend([
        "",
        "## 输出说明",
        "",
        "- `index.html`：可视化总览与全部病例索引。",
        "- `predictions/<model>/`：每个模型对全部图像的二值预测。",
        "- `all_cases/`：每张图的 RGB、GT 和各代表模型 TP/FP/FN 对照。",
        "- `rankings/`：最好、最差、误检、漏检、模型改善/退化和分歧样本清单。",
        "- `per_image_metrics.csv`：后续统计与失败模式聚类的主表。",
    ])
    with open(os.path.join(out_dir, "analysis_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    manifest = read_json(args.manifest)
    experiments = [normalize_experiment(exp) for exp in manifest["experiments"]]
    names = [exp["name"] for exp in experiments]
    if len(names) != len(set(names)):
        raise ValueError("Experiment names must be unique")
    if manifest["primary_baseline"] not in names:
        raise ValueError("primary_baseline is not present in experiments")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "rankings"), exist_ok=True)

    dataset = VesselDataset(
        image_dir=os.path.join(args.data_dir, args.split, "images"),
        mask_dir=os.path.join(args.data_dir, args.split, "masks"),
        img_size=args.img_size,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    filenames = dataset.filenames
    images, masks = read_images_and_masks(args.data_dir, args.split, filenames, args.img_size)
    device = resolve_device(args.device)
    print(f"[Setup] device={device} images={len(filenames)} models={len(experiments)}")

    predictions = {}
    for exp in experiments:
        if exp["model_type"] == "ensemble":
            continue
        probabilities = infer_experiment(exp, loader, filenames, args, device)
        predictions[exp["name"]] = probabilities
        save_prediction_pngs(args.out_dir, exp, filenames, probabilities, args.threshold)
    for exp in experiments:
        if exp["model_type"] != "ensemble":
            continue
        probabilities = build_ensemble(exp, predictions)
        predictions[exp["name"]] = probabilities
        path = cache_path(args.out_dir, exp["name"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, filenames=np.asarray(filenames), probabilities=probabilities.astype(np.float16))
        save_prediction_pngs(args.out_dir, exp, filenames, probabilities, args.threshold)

    patient_lookup = load_patient_lookup(args.patient_mapping)
    rows = metric_rows(experiments, filenames, images, masks, predictions, args, patient_lookup)
    aggregate = aggregate_rows(experiments, rows)
    comparisons, deltas = comparison_rows(manifest.get("comparisons", []), rows)
    patients = patient_rows(rows)
    categories, disagreement = rank_cases(
        manifest, experiments, rows, deltas, predictions, filenames, args.top_k
    )

    write_csv(os.path.join(args.out_dir, "per_image_metrics.csv"), rows)
    write_csv(os.path.join(args.out_dir, "aggregate_metrics.csv"), aggregate)
    write_csv(os.path.join(args.out_dir, "comparison_summary.csv"), comparisons)
    write_csv(os.path.join(args.out_dir, "patient_metrics.csv"), patients)
    for key, _, category_rows in categories:
        write_csv(os.path.join(args.out_dir, "rankings", key + ".csv"), category_rows)
    for name, model_deltas in deltas.items():
        write_csv(os.path.join(args.out_dir, "rankings", safe_name(name) + "_all_deltas.csv"), model_deltas)

    metric_lookup = {(row["experiment"], row["filename"]): row for row in rows}
    metric_lookup["threshold"] = args.threshold
    panel_experiments = [exp for exp in experiments if exp.get("panel", True)]
    all_case_dir = os.path.join(args.out_dir, "all_cases")
    if not args.skip_all_case_panels:
        for index, (filename, image, mask) in enumerate(tqdm(zip(filenames, images, masks), total=len(filenames), desc="Render all cases")):
            metric_lookup["index"] = index
            render_case_panel(
                os.path.join(all_case_dir, os.path.splitext(filename)[0] + ".jpg"),
                filename, image, mask, panel_experiments, predictions, metric_lookup, args.panel_size,
            )

    plot_metric_distributions(args.out_dir, experiments, rows)
    plot_comparisons(args.out_dir, comparisons, deltas)
    plot_difficulty(args.out_dir, manifest["primary_baseline"], rows)
    write_html_report(args.out_dir, manifest, aggregate, comparisons, categories, rows)
    write_markdown_summary(args.out_dir, manifest, aggregate, comparisons, categories, rows)
    shutil.copy2(args.manifest, os.path.join(args.out_dir, "manifest.json"))
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump({**vars(args), "device_used": str(device), "n_images": len(filenames)}, handle, ensure_ascii=False, indent=2)

    print("\n[Prediction error review complete]")
    print(f"HTML: {os.path.join(args.out_dir, 'index.html')}")
    print(f"Summary: {os.path.join(args.out_dir, 'analysis_summary.md')}")
    print(f"Predictions: {os.path.join(args.out_dir, 'predictions')}")


if __name__ == "__main__":
    main()
