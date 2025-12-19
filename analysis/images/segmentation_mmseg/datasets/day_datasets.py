from dotenv import load_dotenv
import os
import json
import logging
from pathlib import Path

from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

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

        logging.info("Initializing Dy30Dataset with:")
        logging.info("  json_mapping_path: %s", self.json_mapping_path)
        logging.info("  day_filter: %s", self.day_filter)
        logging.info("  test_mode: %s", self.test_mode)

        # Ensure the JSON mapping file exists
        if not os.path.exists(self.json_mapping_path):
            raise FileNotFoundError("JSON mapping file does not exist: %s", self.json_mapping_path)

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

        logging.info("Loaded %d entries from image mapping JSON", len(image_mapping))

        # only filter if day_filter is set
        if self.day_filter:
            image_mapping = {
                k: v for k, v in image_mapping.items()
                if v.get('dayID') == self.day_filter
            }
            logging.info("Filtered to %d entries with dayID=%s", len(image_mapping), self.day_filter)

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

        logging.info("Found %d valid image-mask pairs", len(data_list))
        if not data_list:
            logging.warning("WARNING: No valid pairs found! Check your mapping paths or filters.")

        return data_list


    def parse_data_info(self, data_info):
        result = super().parse_data_info(data_info)
        return result