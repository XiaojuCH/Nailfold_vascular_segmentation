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
from utils.metrics import METRIC_KEYS, average_metric_rows, per_image_metrics_from_logits
from models.joint_framework import (
    AnisotropicEnhancer,
    Enhancer,
    JointModel,
    JointModel_BoundaryRefine,
    JointModel_DecoderDistill,
    JointModel_DualFusion,
    JointModel_Gated,
    JointModel_V2,
    MultiScaleEnhancer,
)
from models.transunet_official import TransUNetOfficial
from models.unet_baseline import UNet
from models.unet_plus_plus import UNetPlusPlus


DATASETS = {
    "jiabi": "./dataset_raw_split",
    "anfc256": "./dataset_anfc256_split",
    "all": "./dataset_all_split",
    "all_filtered": "./dataset_all_filtered",
    "all_filtered_VT_Turn": "./dataset_all_filtered_VT_Turn",
}


DEFAULT_MANIFEST = [
    {
        "name": "UNet",
        "model_type": "unet",
        "weight": "results/experiments/all_filtered/unet/best_model.pth",
    },
    {
        "name": "UNet++",
        "model_type": "unet++",
        "weight": "results/experiments/all_filtered/unet++/best_model.pth",
    },
    {
        "name": "TransUNet",
        "model_type": "transunet",
        "weight": "results/experiments/all_filtered/baseline_retrain_20260619/0619_0232/best_model.pth",
    },
    {
        "name": "Ours green+CLAHE",
        "model_type": "ours",
        "weight": "results/experiments/all_filtered/ours/best_model.pth",
        "teacher_mode": "green+clahe",
        "enhancer": "basic",
        "joint_model": "v1",
        "lambda_mse": 10.0,
        "lambda_grad": 30.0,
    },
    {
        "name": "Ours CLAHE only",
        "model_type": "ours",
        "weight": "results/experiments/all_filtered/ours_clahe_only/best_model.pth",
        "teacher_mode": "clahe_only",
        "enhancer": "basic",
        "joint_model": "v1",
        "lambda_mse": 10.0,
        "lambda_grad": 30.0,
    },
    {
        "name": "Ours green only",
        "model_type": "ours",
        "weight": "results/experiments/all_filtered/ours_green_only/best_model.pth",
        "teacher_mode": "green_only",
        "enhancer": "basic",
        "joint_model": "v1",
        "lambda_mse": 10.0,
        "lambda_grad": 30.0,
    },
    {
        "name": "Ours green only multiscale",
        "model_type": "ours",
        "weight": "results/experiments/all_filtered/ours_green_only_multiscale/0529_1358/best_model.pth",
        "teacher_mode": "green_only",
        "enhancer": "multiscale",
        "joint_model": "v1",
        "lambda_mse": 10.0,
        "lambda_grad": 30.0,
    },
    {
        "name": "Ours green only gated",
        "model_type": "ours",
        "weight": "results/experiments/all_filtered/ours_green_only_gated/0529_1755/best_model.pth",
        "teacher_mode": "green_only",
        "enhancer": "basic",
        "joint_model": "gated",
        "lambda_mse": 10.0,
        "lambda_grad": 30.0,
    },
    {
        "name": "Ours green only inverse attention",
        "model_type": "ours",
        "weight": "results/experiments/all_filtered/ours_green_only_v2_inverse_attention/0529_1619/best_model.pth",
        "teacher_mode": "green_only",
        "enhancer": "basic",
        "joint_model": "v2",
        "attention_mode": "inverse",
        "lambda_mse": 10.0,
        "lambda_grad": 30.0,
    },
    {
        "name": "Ours green only learnable loss",
        "model_type": "ours",
        "weight": "results/experiments/all_filtered/ours_green_only_v1_learnable/0529_1133/best_model.pth",
        "teacher_mode": "green_only",
        "enhancer": "basic",
        "joint_model": "v1",
        "loss_weighting": "learnable",
        "lambda_mse": 10.0,
        "lambda_grad": 30.0,
    },
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Unified sequential evaluator for nailfold capillary segmentation models."
    )
    parser.add_argument("--manifest", default="", help="JSON manifest list or {'experiments': [...]} file.")
    parser.add_argument("--use_default_manifest", action="store_true", help="Evaluate built-in all_filtered manifest.")
    parser.add_argument("--name", default="", help="Single-run experiment name.")
    parser.add_argument("--model_type", choices=["unet", "unet++", "transunet", "ours"], default="")
    parser.add_argument("--weight", default="", help="Single-run best_model.pth path.")
    parser.add_argument("--dataset", default="all_filtered", choices=list(DATASETS.keys()))
    parser.add_argument("--data_dir", default="", help="Custom dataset root.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary_tolerance", type=int, default=2)
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
    parser.add_argument("--cbdice_weight", type=float, default=0.5)
    parser.add_argument("--boundary_aux_weight", type=float, default=0.3)
    parser.add_argument("--lambda_decoder_distill", type=float, default=1.0)
    parser.add_argument("--decoder_distill_layers", default="2,3")
    parser.add_argument("--focal_alpha", type=float, default=0.3)
    parser.add_argument("--focal_beta", type=float, default=0.7)
    parser.add_argument("--focal_gamma", type=float, default=0.75)
    parser.add_argument("--out_dir", default="results/unified_eval")
    return parser.parse_args()


def load_manifest(args):
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        experiments = data.get("experiments", data) if isinstance(data, dict) else data
    elif args.use_default_manifest:
        experiments = DEFAULT_MANIFEST
    else:
        if not args.model_type or not args.weight:
            raise ValueError("Provide --model_type and --weight, or pass --manifest/--use_default_manifest.")
        experiments = [
            {
                "name": args.name or args.model_type,
                "model_type": args.model_type,
                "weight": args.weight,
                "teacher_mode": args.teacher_mode,
                "enhancer": args.enhancer,
                "enhancer_norm": args.enhancer_norm,
                "joint_model": args.joint_model,
                "attention_mode": args.attention_mode,
                "loss_weighting": args.loss_weighting,
                "lambda_mse": args.lambda_mse,
                "lambda_grad": args.lambda_grad,
                "seg_loss": args.seg_loss,
                "cldice_weight": args.cldice_weight,
                "boundary_weight": args.boundary_weight,
                "cbdice_weight": args.cbdice_weight,
                "boundary_aux_weight": args.boundary_aux_weight,
                "lambda_decoder_distill": args.lambda_decoder_distill,
                "decoder_distill_layers": args.decoder_distill_layers,
                "focal_alpha": args.focal_alpha,
                "focal_beta": args.focal_beta,
                "focal_gamma": args.focal_gamma,
            }
        ]

    normalized = []
    for exp in experiments:
        item = dict(exp)
        item.setdefault("name", item.get("model_type", "experiment"))
        item.setdefault("enhancer", args.enhancer)
        item.setdefault("enhancer_norm", args.enhancer_norm)
        item.setdefault("joint_model", args.joint_model)
        item.setdefault("attention_mode", args.attention_mode)
        item.setdefault("teacher_mode", "")
        item.setdefault("loss_weighting", "fixed")
        item.setdefault("lambda_mse", "")
        item.setdefault("lambda_grad", "")
        item.setdefault("seg_loss", "")
        item.setdefault("cldice_weight", "")
        item.setdefault("boundary_weight", "")
        item.setdefault("cbdice_weight", "")
        item.setdefault("boundary_aux_weight", "")
        item.setdefault("lambda_decoder_distill", "")
        item.setdefault("decoder_distill_layers", "")
        item.setdefault("focal_alpha", "")
        item.setdefault("focal_beta", "")
        item.setdefault("focal_gamma", "")
        normalized.append(item)
    return normalized


def build_model(exp, img_size, device):
    model_type = exp["model_type"]
    if model_type == "unet":
        model = UNet(n_channels=3, n_classes=1)
    elif model_type == "unet++":
        model = UNetPlusPlus(n_channels=3, n_classes=1)
    elif model_type == "transunet":
        model = TransUNetOfficial(n_channels=3, n_classes=1, img_size=img_size)
    elif model_type == "ours":
        segmentor = TransUNetOfficial(n_channels=3, n_classes=1, img_size=img_size)
        enhancer_norm = exp.get("enhancer_norm", "bn") or "bn"
        if exp.get("enhancer", "basic") == "multiscale":
            enhancer = MultiScaleEnhancer(in_channels=3, out_channels=3, norm_type=enhancer_norm)
        elif exp.get("enhancer", "basic") == "anisotropic":
            enhancer = AnisotropicEnhancer(in_channels=3, out_channels=3, norm_type=enhancer_norm)
        else:
            enhancer = Enhancer(in_channels=3, out_channels=3, norm_type=enhancer_norm)

        if exp.get("joint_model", "v1") == "v2":
            model = JointModel_V2(enhancer, segmentor, attention_mode=exp.get("attention_mode", "normal"))
        elif exp.get("joint_model", "v1") == "gated":
            model = JointModel_Gated(enhancer, segmentor)
        elif exp.get("joint_model", "v1") == "boundary_refine":
            model = JointModel_BoundaryRefine(enhancer, segmentor)
        elif exp.get("joint_model", "v1") == "decoder_distill":
            model = JointModel_DecoderDistill(enhancer, segmentor)
        elif exp.get("joint_model", "v1") == "dual_fusion":
            model = JointModel_DualFusion(enhancer, segmentor, norm_type=enhancer_norm)
        else:
            model = JointModel(enhancer, segmentor)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)


def forward_logits(model, images, model_type):
    outputs = model(images)
    if model_type == "ours":
        return outputs[0]
    return outputs


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_write_xlsx(path, rows, fieldnames):
    try:
        import openpyxl
    except Exception:
        return False

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "aggregate"
    sheet.append(fieldnames)
    for row in rows:
        sheet.append([row.get(field, "") for field in fieldnames])
    workbook.save(path)
    return True


def evaluate_one(exp, dataset, loader, args, device, run_dir):
    weight = exp["weight"]
    if not os.path.exists(weight):
        raise FileNotFoundError(f"Missing weight for {exp['name']}: {weight}")

    print(f"\n[Eval] {exp['name']}")
    print(f"       model={exp['model_type']} weight={weight}")

    model = build_model(exp, args.img_size, device)
    state_dict = load_state_dict(weight, device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=exp["name"]):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = forward_logits(model, images, exp["model_type"])
            batch_rows = per_image_metrics_from_logits(
                logits,
                masks,
                threshold=args.threshold,
                boundary_tolerance=args.boundary_tolerance,
            )
            start = len(rows)
            for offset, metrics in enumerate(batch_rows):
                filename = dataset.filenames[start + offset]
                row = {
                    "experiment": exp["name"],
                    "model_type": exp["model_type"],
                    "filename": filename,
                    **{key: metrics[key] for key in METRIC_KEYS},
                    "tp": metrics["tp"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                }
                rows.append(row)

    per_image_path = os.path.join(run_dir, f"{safe_name(exp['name'])}_per_image.csv")
    per_image_fields = [
        "experiment",
        "model_type",
        "filename",
        *METRIC_KEYS,
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    write_csv(per_image_path, rows, per_image_fields)

    avg = average_metric_rows(rows)
    summary = {
        "experiment": exp["name"],
        "model_type": exp["model_type"],
        "weight": weight,
        "dataset": args.data_dir or DATASETS[args.dataset],
        "split": args.split,
        "threshold": args.threshold,
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
        "cbdice_weight": exp.get("cbdice_weight", ""),
        "boundary_aux_weight": exp.get("boundary_aux_weight", ""),
        "lambda_decoder_distill": exp.get("lambda_decoder_distill", ""),
        "decoder_distill_layers": exp.get("decoder_distill_layers", ""),
        "focal_alpha": exp.get("focal_alpha", ""),
        "focal_beta": exp.get("focal_beta", ""),
        "focal_gamma": exp.get("focal_gamma", ""),
        "n_images": len(rows),
        **avg,
        "per_image_csv": per_image_path,
    }
    return summary


def safe_name(name):
    allowed = []
    for ch in name:
        if ch.isalnum():
            allowed.append(ch)
        elif ch == "+":
            allowed.append("_plus_")
        else:
            allowed.append("_")
    compact = "".join(allowed).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact or "experiment"


def format_summary_line(row):
    return (
        f"{row['experiment']}: "
        f"Dice={row['dice']:.4f}, "
        f"IoU={row['iou']:.4f}, "
        f"Recall={row['sensitivity']:.4f}, "
        f"Precision={row['precision']:.4f}, "
        f"Spec={row['specificity']:.4f}, "
        f"Acc={row['accuracy']:.4f}, "
        f"HD95={row['hd95']:.2f}, "
        f"clDice={row['cldice']:.4f}, "
        f"BoundaryF1={row['boundary_f1']:.4f}"
    )


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = args.data_dir if args.data_dir else DATASETS[args.dataset]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    dataset = VesselDataset(
        image_dir=os.path.join(data_dir, args.split, "images"),
        mask_dir=os.path.join(data_dir, args.split, "masks"),
        teacher_dir=None,
        img_size=args.img_size,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    experiments = load_manifest(args)

    summaries = []
    for exp in experiments:
        summaries.append(evaluate_one(exp, dataset, loader, args, device, run_dir))

    aggregate_fields = [
        "experiment",
        "model_type",
        "weight",
        "dataset",
        "split",
        "threshold",
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
        "cbdice_weight",
        "boundary_aux_weight",
        "lambda_decoder_distill",
        "decoder_distill_layers",
        "focal_alpha",
        "focal_beta",
        "focal_gamma",
        "n_images",
        *METRIC_KEYS,
        "per_image_csv",
    ]
    aggregate_csv = os.path.join(run_dir, "aggregate_results.csv")
    write_csv(aggregate_csv, summaries, aggregate_fields)
    aggregate_xlsx = os.path.join(run_dir, "aggregate_results.xlsx")
    wrote_xlsx = try_write_xlsx(aggregate_xlsx, summaries, aggregate_fields)

    print("\n[Unified Evaluation Complete]")
    print(f"Aggregate CSV:  {aggregate_csv}")
    if wrote_xlsx:
        print(f"Aggregate XLSX: {aggregate_xlsx}")
    else:
        print("Aggregate XLSX: skipped because openpyxl is unavailable")
    for row in summaries:
        print(format_summary_line(row))


if __name__ == "__main__":
    main()
