"""Official MMSegmentation DeepLabV3+ reproduction on the fixed JiaBi split.

This config reproduces the supplied MobileNetV2 + ASPP protocol, but uses an
MMSeg-compatible 0/1 label view and explicitly defines train/val/test splits.
"""

import os


data_root = os.environ.get("JIABI_MMSEG_DATA_ROOT", "./dataset_all_filtered_mmseg")
crop_size = (256, 256)
max_iters = 10000
val_interval = 1000

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    size_divisor=None,
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="MobileNetV2",
        widen_factor=1.0,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 4, 6),
        norm_cfg=dict(type="BN", requires_grad=True),
    ),
    decode_head=dict(
        type="DepthwiseSeparableASPPHead",
        in_channels=320,
        in_index=2,
        channels=512,
        c1_in_channels=24,
        c1_channels=12,
        dilations=(1, 12, 24, 36),
        num_classes=2,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True),
            dict(type="DiceLoss", loss_weight=2.0, ignore_index=255),
        ],
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=crop_size, keep_ratio=False),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="RandomRotate", prob=0.5, degree=(-45, 45), pad_val=0, seg_pad_val=255),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=crop_size, keep_ratio=False),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

dataset_meta = dict(classes=("background", "vessel"), palette=[[0, 0, 0], [255, 255, 255]])
dataset_common = dict(
    type="BaseSegDataset",
    data_root=data_root,
    img_suffix=".png",
    seg_map_suffix=".png",
    metainfo=dataset_meta,
)

train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        **dataset_common,
        data_prefix=dict(img_path="train/images", seg_map_path="train/masks"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        **dataset_common,
        data_prefix=dict(img_path="val/images", seg_map_path="val/masks"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        **dataset_common,
        data_prefix=dict(img_path="test/images", seg_map_path="test/masks"),
        pipeline=test_pipeline,
    ),
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=1e-3, weight_decay=1e-4))
param_scheduler = [
    dict(type="CosineAnnealingLR", T_max=max_iters, eta_min=1e-5, by_epoch=False, begin=0, end=max_iters)
]

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice"])
test_evaluator = val_evaluator

default_scope = "mmseg"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=val_interval, save_best="mDice", rule="greater", max_keep_ckpts=2),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=False),
)
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0), dist_cfg=dict(backend="nccl"))
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False
# Keep the seed fixed, but do not force torch deterministic algorithms: the
# official MMSeg IoU evaluator uses CUDA histc, which has no deterministic
# implementation in PyTorch 2.1 on this platform.
randomness = dict(seed=42, deterministic=False)
