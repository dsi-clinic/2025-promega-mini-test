custom_imports = dict(
    imports=[
        'mmseg.models.losses',
        'mmseg.datasets.transforms'
    ],
    # allow_failed_imports=False
)

from day_datasets import Dy30Dataset

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='ImgDataPreprocessor',
    mean=[127.5],
    std=[127.5],
    bgr_to_rgb=False
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,  # You can use 18, 34, 50, 101, or 152
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # Don't freeze any stages for fine-tuning
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
decode_head=dict(
    type='UPerHead',  # More effective for biomedical segmentation    
    in_channels=[256, 512, 1024, 2048],  # These must match ResNet-50 output channels
    in_index=[0, 1, 2, 3],
    pool_scales=(1, 2, 3, 6),
    channels=128,
    dropout_ratio=0.1,
    num_classes=2,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=[
        dict(type='DiceLoss', loss_weight=2.0, use_sigmoid=True, loss_name='loss_dice'),
        dict(type='FocalLoss', loss_weight=1.0, gamma=2.0, use_sigmoid=True, loss_name='loss_focal')
    ]
),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Define your normalization and augmentation pipelines explicitly here:
img_norm_cfg = dict(
    mean=[127.5],
    std=[127.5],
    to_rgb=False)

# Import your custom transform at the top of your config
custom_imports = dict(imports=['custom_transforms'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_seg=True),
    dict(type='Resize', scale=(256, 192), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # Add these augmentations:
    # dict(type='RandomRotate', prob=0.5, degree=20),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='RandomCrop', crop_size=(192, 192)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0),
    dict(type='PackSegInputs')  # This is crucial for creating the 'inputs' key
]

# Fixed val_pipeline with necessary transformations
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_seg=True),
    dict(type='Resize', scale=(256, 192), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0),
    dict(type='PackSegInputs')  # This is crucial for creating the 'inputs' key
]

# Updated dataloaders with data_root
train_dataloader = dict(
    batch_size=10,
    num_workers=2,
    dataset=dict(
        type='Dy30Dataset',
        json_mapping_path='${JSON_MAPPING_PATH}',
        day_filter="Dy30",
        pipeline=train_pipeline,
        lazy_init=False))

val_dataloader = dict(
    batch_size=1, # maybe change to 2 or higher?
    num_workers=2,
    dataset=dict(
        type='Dy30Dataset',
        json_mapping_path='${JSON_MAPPING_PATH}',
        day_filter="Dy30",
        pipeline=val_pipeline,
        lazy_init=False))

test_dataloader = val_dataloader

# Optimizer and scheduler settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.1))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='PolyLR',
        power=0.9,
        eta_min=0.0,
        begin=100,
        end=1000,
        by_epoch=False)
]

# Training loops configuration
train_cfg = dict(type='IterBasedTrainLoop', max_iters=1000, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks for logging and checkpointing
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=500, by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=255)
test_evaluator = val_evaluator