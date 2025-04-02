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
                 day_filter='Dy30',
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
        
        # Filter entries by day if specified
        if self.day_filter:
            filtered_mapping = {
                k: v for k, v in image_mapping.items() 
                if v.get('dayID') == self.day_filter
            }
            print(f"Filtered to {len(filtered_mapping)} entries with dayID={self.day_filter}")
        else:
            filtered_mapping = image_mapping
            
        data_list = []
        for img_id, img_info in filtered_mapping.items():
            # Get the original image path from the mapping
            img_path = img_info.get('Best Z Filename')
            
            if img_path and os.path.exists(img_path):
                # Construct mask path based on your actual mask naming convention
                # You'll need to adapt this part to match how your masks are named/located
                img_dir = os.path.dirname(img_path)
                base_name = os.path.basename(img_path).split('.')[0]
                mask_path = os.path.join(MASKS_FOLDER, f"{base_name}_cellpose_mask.png")
                
                # Alternative mask path if your masks are in a central location
                # mask_path = os.path.join('/path/to/masks', f"{img_id}_mask.npy")
                
                # Only add if both image and mask exist
                if os.path.exists(mask_path):
                    data_info = {
                        'img_path': img_path,
                        'seg_map_path': mask_path,
                        'img_id': img_id,
                        'seg_fields': ['gt_sem_seg'],
                        # Add any other metadata from img_info that you need
                        'dayID': img_info.get('dayID'),
                        'BA': img_info.get('BA'),
                        'wellID': img_info.get('wellID')
                    }
                    data_list.append(data_info)
                else:
                    print(f"Warning: Mask not found for {img_id} at {mask_path}")
            else:
                print(f"Warning: Image not found for {img_id} at {img_path}")
                
        print(f"Found {len(data_list)} valid image-mask pairs")
        if len(data_list) == 0:
            print(f"WARNING: No valid image-mask pairs found! Check paths and filters.")
            
        return data_list
        
    # def parse_data_info(self, data_info):
    #     return {
    #         'img_path': data_info['img_path'],
    #         'gt_seg_map': data_info['seg_map_path'], 
    #         'seg_fields': ['gt_seg_map'], 
    #         # Keep your metadata:
    #         'img_id': data_info['img_id'],
    #         'dayID': data_info.get('dayID'),
    #         'BA': data_info.get('BA'), 
    #         'wellID': data_info.get('wellID')
    #     }