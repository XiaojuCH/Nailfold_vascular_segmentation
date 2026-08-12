crop_size = (
    256,
    256,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    size_divisor=None,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'D:\\Projects_\\JiaBi_new\\dataset_all_filtered_mmseg'
dataset_common = dict(
    data_root='D:\\Projects_\\JiaBi_new\\dataset_all_filtered_mmseg',
    img_suffix='.png',
    metainfo=dict(
        classes=(
            'background',
            'vessel',
        ),
        palette=[
            [
                0,
                0,
                0,
            ],
            [
                255,
                255,
                255,
            ],
        ]),
    seg_map_suffix='.png',
    type='BaseSegDataset')
dataset_meta = dict(
    classes=(
        'background',
        'vessel',
    ),
    palette=[
        [
            0,
            0,
            0,
        ],
        [
            255,
            255,
            255,
        ],
    ])
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=2,
        rule='greater',
        save_best='mDice',
        type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
max_iters = 10000
model = dict(
    auxiliary_head=None,
    backbone=dict(
        dilations=(
            1,
            1,
            1,
            2,
            2,
            4,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='BN'),
        out_indices=(
            1,
            4,
            6,
        ),
        strides=(
            1,
            2,
            2,
            1,
            1,
            1,
            1,
        ),
        type='MobileNetV2',
        widen_factor=1.0),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        size_divisor=None,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        c1_channels=12,
        c1_in_channels=24,
        channels=512,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        in_channels=320,
        in_index=2,
        loss_decode=[
            dict(
                avg_non_ignore=True,
                loss_weight=1.0,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(ignore_index=255, loss_weight=2.0, type='DiceLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        type='DepthwiseSeparableASPPHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=10000,
        begin=0,
        by_epoch=False,
        end=10000,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=True, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        data_root='D:\\Projects_\\JiaBi_new\\dataset_all_filtered_mmseg',
        img_suffix='.png',
        metainfo=dict(
            classes=(
                'background',
                'vessel',
            ),
            palette=[
                [
                    0,
                    0,
                    0,
                ],
                [
                    255,
                    255,
                    255,
                ],
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='BaseSegDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=10000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        data_root='D:\\Projects_\\JiaBi_new\\dataset_all_filtered_mmseg',
        img_suffix='.png',
        metainfo=dict(
            classes=(
                'background',
                'vessel',
            ),
            palette=[
                [
                    0,
                    0,
                    0,
                ],
                [
                    255,
                    255,
                    255,
                ],
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(direction='vertical', prob=0.5, type='RandomFlip'),
            dict(
                degree=(
                    -45,
                    45,
                ),
                pad_val=0,
                prob=0.5,
                seg_pad_val=255,
                type='RandomRotate'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='BaseSegDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(
        degree=(
            -45,
            45,
        ),
        pad_val=0,
        prob=0.5,
        seg_pad_val=255,
        type='RandomRotate'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        data_root='D:\\Projects_\\JiaBi_new\\dataset_all_filtered_mmseg',
        img_suffix='.png',
        metainfo=dict(
            classes=(
                'background',
                'vessel',
            ),
            palette=[
                [
                    0,
                    0,
                    0,
                ],
                [
                    255,
                    255,
                    255,
                ],
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='BaseSegDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
val_interval = 1000
