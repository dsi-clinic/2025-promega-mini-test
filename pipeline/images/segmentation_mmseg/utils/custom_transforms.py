import numpy as np
from PIL import Image
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class CustomLoadAnnotations(LoadAnnotations):
    """Load annotations and ensure proper binary mask handling."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _load_seg_map(self, results):
        """Load segmentation map with binary processing."""
        # Original implementation in parent class
        seg_map = results.get('seg_map_path', None)
        if seg_map is None:
            return results
        
        # Load the mask using PIL
        mask = np.array(Image.open(seg_map))
        
        # Normalize to binary [0, 1] for your binary segmentation
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        
        results['gt_seg_map'] = mask
        results['seg_fields'].append('gt_seg_map')
        
        return results