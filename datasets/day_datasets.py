from dotenv import load_dotenv
import os
import json
import glob
import numpy as np
from pathlib import Path
# from skimage.io import imread
from mmseg.registry import DATASETS
import os
import json
import numpy as np
from PIL import Image
import torch
from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS
from mmseg.structures import SegDataSample
import torch


# Get the absolute path to the script's directory
script_dir = Path(__file__).resolve().parent

# Get the parent directory (where .env should be)
parent_dir = script_dir.parent

# Load the .env file from the parent directory
dotenv_path = parent_dir / ".env"

load_successful = load_dotenv(dotenv_path)
if not load_successful:
    raise Exception('.env file failed to load')

# Assume these paths come from your .env loading logic:
MASKS_FOLDER = Path(os.environ["MASKS_FOLDER"])
JSON_MAPPING_PATH = Path(os.environ["JSON_MAPPING_PATH"])
PREPROCESSED_FOLDER = Path(os.environ["PREPROCESSED_FOLDER"])


@DATASETS.register_module()
class Dy30Dataset(BaseDataset):
    METAINFO = {
        'classes': ['background', 'cell'],
        'palette': [[0, 0, 0], [255, 255, 255]]
    }
    """Custom dataset for segmentation using JSON mapping file.
    
    This dataset doesn't rely on a specific directory structure but instead
    uses a JSON file that maps image IDs to their file paths.
    
    Args:
        json_mapping_path (str): Path to JSON file with image mappings.
        day_filter (str, optional): Filter images by day ID. Default: 'Dy30'.
        pipeline (list[dict]): Processing pipeline.
        test_mode (bool): If True, dataset will work in test mode. Default: False.
    """
    
    def __init__(self,
                 json_mapping_path,
                 day_filter=None,
                 pipeline=None,
                 test_mode=False,
                 lazy_init=False,
                 **kwargs):
        self.json_mapping_path = json_mapping_path
        self.day_filter = day_filter
        self.test_mode = test_mode
        
        print(f"Initializing Dy30Dataset with:")
        print(f"  json_mapping_path: {self.json_mapping_path}")
        print(f"  day_filter: {self.day_filter}")
        print(f"  test_mode: {self.test_mode}")
        
        # Ensure the JSON mapping file exists
        if not os.path.exists(self.json_mapping_path):
            raise FileNotFoundError(f"JSON mapping file does not exist: {self.json_mapping_path}")
            
        # Initialize the dataset
        super().__init__(
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            **kwargs)
        
    def load_data_list(self):
        """Load annotations from the JSON mapping file.
        
        Returns:
            list[dict]: A list of annotation.
        """
        # Load the JSON mapping file
        with open(self.json_mapping_path, 'r') as f:
            image_mapping = json.load(f)
            
        print(f"Loaded {len(image_mapping)} entries from image mapping JSON")
        
        # only filter if day_filter is set
        if self.day_filter:
            image_mapping = {
                k: v for k, v in image_mapping.items()
                if v.get('dayID') == self.day_filter
            }
            print(f"Filtered to {len(image_mapping)} entries with dayID={self.day_filter}")
            
        data_list = []
        data_list = []
        for img_id, info in image_mapping.items():
            img_p  = Path(info.get('img_path', ''))
            # prefer the lowercase key if available
            msk_p  = Path(info.get('mask_path', '') or info.get('Mask Path', ''))
            if not img_p.exists() or not msk_p.exists():
                continue

            data_list.append({
                'img_path':     str(img_p),
                'seg_map_path': str(msk_p),
                'img_id':       img_id,
                'seg_fields':   ['gt_sem_seg'],
                'dayID':        info.get('dayID'),
                'BA':           info.get('BA'),
                'wellID':       info.get('wellID'),
            })

        print(f"Found {len(data_list)} valid image-mask pairs")
        if not data_list:
            print("WARNING: No valid pairs found! Check your mapping paths or filters.")

        return data_list

        
    def parse_data_info(self, data_info):
        result = super().parse_data_info(data_info)
        return result