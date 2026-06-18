import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_vessel import VesselDataset
from utils.metrics import calculate_comprehensive_metrics
from models.joint_framework import Enhancer, JointModel, JointModel_Gated, JointModel_V2, MultiScaleEnhancer
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


def cl_score(pred, target, smooth=1e-6):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    skel_pred = skeletonize(pred)
    skel_target = skeletonize(target)
    if skel_pred.sum() == 0 or skel_target.sum() == 0:
        return 0.0

    tprec = (skel_pred * target).sum() / (skel_pred.sum() + smooth)
    tsens = (skel_target * pred).sum() / (skel_target.sum() + smooth)
    return (2.0 * tprec * tsens) / (tprec + tsens + smooth)


def boundary_f1(pred, target, tolerance=2, smooth=1e-6):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    pred_boundary = pred ^ binary_dilation(pred)
    target_boundary = target ^ binary_dilation(target)
    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return 0.0

    pred_match = distance_transform_edt(~target_boundary) <= tolerance
    target_match = distance_transform_edt(~pred_boundary) <= tolerance
    precision = (pred_boundary & pred_match).sum() / (pred_boundary.sum() + smooth)
    recall = (target_boundary & target_match).sum() / (target_boundary.sum() + smooth)
    return (2.0 * precision * recall) / (precision + recall + smooth)


def extra_structure_metrics(pred_logits, target_masks, threshold=0.5):
    pred_probs = torch.sigmoid(pred_logits)
    preds = (pred_probs > threshold).float().cpu().numpy()
    targets = target_masks.float().cpu().numpy()

    cldice_values = []
    bf1_values = []
    for i in range(preds.shape[0]):
        pred = preds[i, 0] > 0.5
        target = targets[i, 0] > 0.5
        cldice_values.append(cl_score(pred, target))
        bf1_values.append(boundary_f1(pred, target))
    return {
        "cldice": float(np.mean(cldice_values)),
        "boundary_f1": float(np.mean(bf1_values)),
    }


def build_model(args, device):
    if args.model_type == "unet":
        model = UNet(n_channels=3, n_classes=1)
    elif args.model_type == "unet++":
        model = UNetPlusPlus(n_channels=3, n_classes=1)
    elif args.model_type == "transunet":
        model = TransUNetOfficial(n_channels=3, n_classes=1, img_size=args.img_size)
    elif args.model_type == "ours":
        segmentor = TransUNetOfficial(n_channels=3, n_classes=1, img_size=args.img_size)
        enhancer = MultiScaleEnhancer(in_channels=3, out_channels=3) if args.enhancer == "multiscale" else Enhancer(in_channels=3, out_channels=3)
        if args.joint_model == "v2":
            model = JointModel_V2(enhancer, segmentor, attention_mode=args.attention_mode)
        elif args.joint_model == "gated":
            model = JointModel_Gated(enhancer, segmentor)
        else:
            model = JointModel(enhancer, segmentor)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    return model.to(device)


def forward_logits(model, images, args):
    outputs = model(images)
    if args.model_type == "ours":
        return outputs[0]
    return outputs


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate saved models with clDice and boundary F1.")
    parser.add_argument("--model_type", required=True, choices=["unet", "unet++", "transunet", "ours"])
    parser.add_argument("--weight", required=True, help="Path to best_model.pth")
    parser.add_argument("--dataset", default="all_filtered", choices=list(DATASETS.keys()))
    parser.add_argument("--data_dir", default="", help="Custom dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--enhancer", default="basic", choices=["basic", "multiscale"])
    parser.add_argument("--joint_model", default="v1", choices=["v1", "v2", "gated"])
    parser.add_argument("--attention_mode", default="normal", choices=["normal", "inverse"])
    return parser.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = args.data_dir if args.data_dir else DATASETS[args.dataset]

    dataset = VesselDataset(
        image_dir=os.path.join(data_dir, args.split, "images"),
        mask_dir=os.path.join(data_dir, args.split, "masks"),
        teacher_dir=None,
        img_size=args.img_size,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args, device)
    state_dict = torch.load(args.weight, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    metrics_sum = {
        "dice": 0.0,
        "iou": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "hd95": 0.0,
        "cldice": 0.0,
        "boundary_f1": 0.0,
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {args.split}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = forward_logits(model, images, args)

            base_metrics = calculate_comprehensive_metrics(logits, masks)
            struct_metrics = extra_structure_metrics(logits, masks, threshold=args.threshold)
            for key, value in {**base_metrics, **struct_metrics}.items():
                metrics_sum[key] += value

    avg_metrics = {key: value / len(loader) for key, value in metrics_sum.items()}
    print("\n[FINAL EVAL RESULTS]")
    print(f"Model:     {args.model_type}")
    print(f"Weight:    {args.weight}")
    print(f"Dataset:   {data_dir} / {args.split}")
    print(f"Threshold: {args.threshold}")
    print("-" * 50)
    print(f"Dice:        {avg_metrics['dice']:.4f}")
    print(f"IoU:         {avg_metrics['iou']:.4f}")
    print(f"Recall:      {avg_metrics['sensitivity']:.4f}")
    print(f"Precision:   {avg_metrics['precision']:.4f}")
    print(f"HD95:        {avg_metrics['hd95']:.2f}")
    print(f"clDice:      {avg_metrics['cldice']:.4f}")
    print(f"Boundary F1: {avg_metrics['boundary_f1']:.4f}")
    print(f"Spec:        {avg_metrics['specificity']:.4f}")
    print(f"Acc:         {avg_metrics['accuracy']:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
