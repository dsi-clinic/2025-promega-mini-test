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
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).resolve().parent / "segformer_mitb0.py",
        help='train config file path, e.g. segformer_mitb0.py   '
    )
    parser.add_argument(
        '--splits-dir',
        type=Path,
        help='path to the splits directory created by resize_img_masks.py'
    )
    parser.add_argument(
        '--work-dir',
        type=Path,
        help='the dir to save logs and models, e.g. work_dirs/segformer_mitb0'
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
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.'
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher'
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument(
        '--local_rank',
        '--local-rank',
        type=int,
        default=0,
        help='local rank for distributed training'
    )
    args = parser.parse_args()

    if not args.splits_dir:
        parser.error("--splits-dir is required")

    if not args.work_dir:
        parser.error("--work-dir is required")

    return args

def set_env_vars(args):
    """Set environment variables needed for training."""
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

# Then in the main() function, add this after loading the config:
def main():
    start_time = datetime.datetime.now()
    args = parse_args()
    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)

    # load config
    set_env_vars(args)  # Sets LOCAL_RANK
    cfg = Config.fromfile(args.config)

    # load config
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set paths directly in config (like work_dir) - convert to strings for proper serialization
    splits_dir = str(args.splits_dir)
    cfg.train_dataloader.dataset.json_mapping_path = f'{splits_dir}/mapping_days1330_train.json'
    cfg.val_dataloader.dataset.json_mapping_path = f'{splits_dir}/mapping_days1330_val.json'
    cfg.test_dataloader.dataset.json_mapping_path = f'{splits_dir}/mapping_days1330_test.json'

    # working directory
    cfg.work_dir = str(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # After loading and processing config - start training
    sample = runner.train_dataloader.dataset[0]    # Debug sample
    logging.info("Sample keys: %s", sample.keys())
    logging.info("Input shape: %s", sample['inputs'].shape if 'inputs' in sample else "No inputs")
    logging.info("Mask path: %s", sample['data_samples'].metainfo['seg_map_path'])

    mask = np.array(Image.open(sample['data_samples'].metainfo['seg_map_path']))
    logging.info("Mask unique values: %s", np.unique(mask))  # Should be [0, 1]
    logging.info("Mask shape: %s", mask.shape)  # Should be (H,W)
    runner.train()

    end_time = datetime.datetime.now()
    logging.info("Training completed in %s", end_time - start_time)

if __name__ == '__main__':
    main()
