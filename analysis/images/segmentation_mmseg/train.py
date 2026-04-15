import argparse
import datetime
import logging
import os
import os.path as osp
import numpy as np
from pathlib import Path

from PIL import Image
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from mmengine.registry import MODELS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.backbones.mit import MixVisionTransformer
from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.models.losses import DiceLoss, FocalLoss, CrossEntropyLoss
from mmseg.engine.hooks import SegVisualizationHook
from mmengine.registry import HOOKS
from mmseg.registry import DATASETS

from analysis.images.segmentation_mmseg.datasets.day_datasets import Dy30Dataset
from mmseg.registry import DATASETS

# # Make sure it's registered with the correct registry
if 'Dy30Dataset' not in DATASETS:
    DATASETS.register_module(name='Dy30Dataset', module=Dy30Dataset)

# At the top of your script, use this exact import
from mmengine.registry import DATASETS as MMENGINE_DATASETS

# Then register your dataset here as well
if 'Dy30Dataset' not in MMENGINE_DATASETS:
    MMENGINE_DATASETS.register_module(name='Dy30Dataset', module=Dy30Dataset)

HOOKS.register_module(module=SegVisualizationHook)
MODELS.register_module(module=DiceLoss)
MODELS.register_module(module=FocalLoss)
MODELS.register_module(module=CrossEntropyLoss)
MODELS.register_module(module=EncoderDecoder)
MODELS.register_module(module=MixVisionTransformer)
MODELS.register_module(module=SegformerHead)
# resnet
from mmseg.models.backbones.resnet import ResNet
MODELS.register_module(module=ResNet)
from mmseg.models.decode_heads.uper_head import UPerHead
MODELS.register_module(module=UPerHead)
from mmseg.datasets.transforms import (
    LoadAnnotations,
    Resize,
    RandomFlip,
    RandomRotate,
    RandomCrop,
    PhotoMetricDistortion
)
# These transforms may be in different modules
from mmcv.transforms import Normalize, Pad
from mmseg.datasets import PackSegInputs

# Register them properly
from mmengine.registry import TRANSFORMS
for transform in [
    LoadAnnotations, Resize, RandomFlip,
    RandomRotate, RandomCrop, PhotoMetricDistortion,
    Normalize, Pad, PackSegInputs
]:
    if transform.__name__ not in TRANSFORMS:
        TRANSFORMS.register_module(module=transform)
        MODELS.register_module(module=transform)
# MODELS.register_module(module=PackSegInputs)

# Add this import at the top of train.py
from mmseg.evaluation import IoUMetric
from mmengine.registry import METRICS

# Register IoUMetric with the metrics registry
if 'IoUMetric' not in METRICS:
    METRICS.register_module(module=IoUMetric)

from mmseg.datasets import PackSegInputs
from mmengine.registry import TRANSFORMS

# Register PackSegInputs if not already registered
if 'PackSegInputs' not in TRANSFORMS:
    TRANSFORMS.register_module(module=PackSegInputs)

from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmseg.registry import DATASETS as MMSEG_DATASETS

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    level=logging.INFO
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).resolve().parent / "segformer_mitb0.py",
        help='train config file path, e.g. segformer_mitb0.py'
    )
    parser.add_argument(
        '--splits-dir',
        type=Path,
        required=True,
        help='path to the splits directory created by test_split/resize_img_masks.py'
    )
    parser.add_argument(
        '--split',
        choices=['early', 'late'],
        required=True,
        help='Which day split to train on'
    )
    parser.add_argument(
        '--work-dir',
        type=Path,
        required=True,
        help='dir to save logs and models, e.g. work_dirs/segformer_mitb0'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, key=value, merged into config'
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher'
    )
    parser.add_argument(
        '--local_rank',
        '--local-rank',
        type=int,
        default=0,
        help='local rank for distributed training'
    )
    return parser.parse_args()


def set_env_vars(args):
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)


def _mapping_paths(splits_dir: Path, split: str):
    # match the filenames in your log exactly
    tag = "days0310" if split == "early" else "days1330"
    return (
        splits_dir / f"mapping_{tag}_train.json",
        splits_dir / f"mapping_{tag}_val.json",
        splits_dir / f"mapping_{tag}_test.json",
    )


def assert_results(runner, split: str):
    # train/val sizes should add up to expected total minus test size,
    # but easiest is just to assert total across train+val+test.
    train_ds = runner.train_dataloader.dataset
    val_ds = runner.val_dataloader.dataset
    test_ds = runner.test_dataloader.dataset

    n_train = len(train_ds.load_data_list())
    n_val = len(val_ds.load_data_list())
    n_test = len(test_ds.load_data_list())

    total = n_train + n_val + n_test
    logging.info("%s dataset sizes: train=%d val=%d test=%d total=%d", split, n_train, n_val, n_test, total)


def main():
    start_time = datetime.datetime.now()
    args = parse_args()
    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)

    set_env_vars(args)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    train_map, val_map, test_map = _mapping_paths(args.splits_dir, args.split)

    # Set mapping paths into the config
    cfg.train_dataloader.dataset.json_mapping_path = str(train_map)
    cfg.val_dataloader.dataset.json_mapping_path = str(val_map)
    cfg.test_dataloader.dataset.json_mapping_path = str(test_map)

    # Each training run lives in its own timestamped subdir so checkpoints,
    # logs, and config are co-located and reruns don't clobber prior weights.
    # On --resume, reuse the latest existing run dir (where last_checkpoint lives).
    if args.resume:
        existing = sorted(
            (p for p in args.work_dir.glob("[0-9]" * 8 + "_" + "[0-9]" * 6) if p.is_dir()),
            key=lambda p: p.name,
        )
        resumable = [p for p in existing if (p / "last_checkpoint").exists()]
        if not resumable:
            raise FileNotFoundError(
                f"--resume set but no run dir with last_checkpoint under {args.work_dir}"
            )
        run_dir = resumable[-1]
        logging.info("Resuming into existing run dir: %s", run_dir)
    else:
        run_dir = args.work_dir / start_time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(run_dir)
    # Snapshot the merged config alongside the checkpoints for reproducibility
    # and so step 9 can pick it up without diving into mmengine's inner log dir.
    cfg.dump(str(run_dir / "config.py"))

    # AMP
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP training is already enabled in your config.', logger='current', level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when optim_wrapper.type is OptimWrapper '
                f'but got {optim_wrapper}.'
            )
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    cfg.resume = args.resume

    # build runner
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # quick sanity check on first sample
    sample = runner.train_dataloader.dataset[0]
    logging.info("Sample keys: %s", sample.keys())
    logging.info("Input shape: %s", sample['inputs'].shape if 'inputs' in sample else "No inputs")

    if 'data_samples' in sample and hasattr(sample['data_samples'], 'metainfo'):
        seg_path = sample['data_samples'].metainfo.get('seg_map_path', None)
        logging.info("Mask path: %s", seg_path)
        if seg_path is not None and osp.exists(seg_path):
            mask = np.array(Image.open(seg_path))
            logging.info("Mask unique values: %s", np.unique(mask))
            logging.info("Mask shape: %s", mask.shape)

    runner.train()

    assert_results(runner, args.split)

    end_time = datetime.datetime.now()
    logging.info("Training completed in %s", end_time - start_time)


if __name__ == '__main__':
    main()
