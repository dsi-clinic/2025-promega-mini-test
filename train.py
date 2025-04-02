from env_config_handler import process_env_vars_in_config

import argparse
import logging
import os
import os.path as osp
import numpy as np
from PIL import Image

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from mmengine.registry import MODELS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.backbones.mit import MixVisionTransformer
from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.models.losses import DiceLoss, FocalLoss
from mmseg.engine.hooks import SegVisualizationHook
from mmengine.registry import HOOKS
from mmseg.registry import DATASETS

from datasets.day_datasets import Dy30Dataset
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
MODELS.register_module(module=EncoderDecoder)
MODELS.register_module(module=MixVisionTransformer)
MODELS.register_module(module=SegformerHead)
# MODELS.register_module(module=PackSegInputs)

# Add this import at the top of train.py
from mmseg.evaluation import IoUMetric
from mmengine.registry import METRICS

# Register IoUMetric with the metrics registry
if 'IoUMetric' not in METRICS:
    METRICS.register_module(module=IoUMetric)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


from mmseg.datasets import PackSegInputs
from mmengine.registry import TRANSFORMS

# Register PackSegInputs if not already registered
if 'PackSegInputs' not in TRANSFORMS:
    TRANSFORMS.register_module(module=PackSegInputs)

    
# Then in the main() function, add this after loading the config:
def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    
    # Process environment variables in the config
    cfg = process_env_vars_in_config(cfg)
    
    # load config
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

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
    # After loading and processing config
    # start training
    from mmengine.registry import DATASETS as MMENGINE_DATASETS
    from mmseg.registry import DATASETS as MMSEG_DATASETS

    # Debug sample
    sample = runner.train_dataloader.dataset[0]
    print("Sample keys:", sample.keys())
    print("Input shape:", sample['inputs'].shape if 'inputs' in sample else "No inputs")
    print("Mask path:", sample['data_samples'].metainfo['seg_map_path'])

    mask = np.array(Image.open(sample['data_samples'].metainfo['seg_map_path']))
    print("Mask unique values:", np.unique(mask))  # Should be [0, 1]
    print("Mask shape:", mask.shape)  # Should be (H,W)
    runner.train()


if __name__ == '__main__':
    main()
