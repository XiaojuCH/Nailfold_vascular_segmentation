import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_vessel import VesselDataset
from utils.metrics import calculate_comprehensive_metrics, per_image_metrics_from_logits, average_metric_rows
from models.unet_baseline import UNet
from models.unet_plus_plus import UNetPlusPlus
from models.joint_framework import Enhancer, JointModel, JointModel_Gated, JointModel_V2, MultiScaleEnhancer
from models.transunet_official import TransUNetOfficial


DATASETS = {
    "jiabi": "./dataset_raw_split",
    "anfc256": "./dataset_anfc256_split",
    "all": "./dataset_all_split",
    "all_filtered": "./dataset_all_filtered",
    "all_filtered_VT_Turn": "./dataset_all_filtered_VT_Turn",
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


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


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
    state_dict = load_state_dict(args.weight, device)
    model.load_state_dict(state_dict)
    model.eval()

    metric_rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {args.split}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = forward_logits(model, images, args)

            metric_rows.extend(per_image_metrics_from_logits(logits, masks, threshold=args.threshold))

    avg_metrics = average_metric_rows(metric_rows)
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
