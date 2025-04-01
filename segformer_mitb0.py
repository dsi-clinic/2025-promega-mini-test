custom_imports = dict(
    imports=[
        'mmseg.models.losses',  # Keep your existing imports
        'mmseg.datasets.transforms'  # Add this line to import transforms
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
        type='MixVisionTransformer',
        in_channels=1,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='DiceLoss', loss_weight=10.0, use_sigmoid=True, loss_name='loss_dice'),
            dict(type='FocalLoss', loss_weight=1.0, use_sigmoid=True, loss_name='loss_focal')
        ],
    ),



    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Define your normalization and augmentation pipelines explicitly here:
img_norm_cfg = dict(
    mean=[127.5],
    std=[127.5],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', scale=(256, 192), keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32, pad_val=0),
    # dict(type='PackSegInputs')  # Replace DefaultFormatBundle and Collect
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', scale=(256, 192), keep_ratio=True),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32, pad_val=0),
    # dict(type='PackSegInputs')  # Replace DefaultFormatBundle and Collect
]


# Updated dataloaders with data_root
# Updated dataloaders for custom dataset
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='Dy30Dataset',
        json_mapping_path='${JSON_MAPPING_PATH}',  # From .env
        day_filter="Dy30",  # Filter to only Dy30 images
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='Dy30Dataset',
        json_mapping_path='${JSON_MAPPING_PATH}',  # From .env
        day_filter="Dy30",  # Filter to only Dy30 images
        pipeline=val_pipeline))

test_dataloader = val_dataloader

# Optimizer and scheduler settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.1))

param_scheduler = [
    dict(
        type='PolyLR',
        power=0.9,
        eta_min=0.0,
        begin=0,
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


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

