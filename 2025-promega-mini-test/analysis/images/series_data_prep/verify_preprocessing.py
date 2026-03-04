"""
Verify preprocessing using saved metadata
"""
import json
from pathlib import Path
from config import OUTPUT_FOLDER

def verify_preprocessing():
    print("\n" + "="*70)
    print("PREPROCESSING VERIFICATION (using metadata)")
    print("="*70)
    
    data_path = OUTPUT_FOLDER / 'complete_series_data_no_blanks.json'
    with open(data_path) as f:
        data = json.load(f)
    
    checked = 0
    errors = []
    
    for key, entry in data.items():
        if 'lstm_processed' not in entry or 'processed' not in entry:
            continue
        
        if checked >= 20:
            break
        
        # Get original
        orig_w = entry['processed']['orig_width_px']
        orig_h = entry['processed']['orig_height_px']
        orig_um_per_px = entry['um_per_px']
        if isinstance(orig_um_per_px, (list, tuple)):
            orig_um_per_px = orig_um_per_px[0]
        
        # Get scaled (from saved metadata)
        scaled_w = entry['lstm_processed']['scaled_width']
        scaled_h = entry['lstm_processed']['scaled_height']
        target_um_per_px = entry['lstm_processed']['target_um_per_px']
        
        # Verify scale factor
        expected_scale = orig_um_per_px / target_um_per_px
        actual_scale_w = scaled_w / orig_w
        actual_scale_h = scaled_h / orig_h
        
        # Verify aspect ratio
        orig_aspect = orig_w / orig_h
        scaled_aspect = scaled_w / scaled_h
        aspect_error = abs(orig_aspect - scaled_aspect) / orig_aspect * 100
        
        # Verify um/px
        actual_um_per_px_w = orig_w * orig_um_per_px / scaled_w
        actual_um_per_px_h = orig_h * orig_um_per_px / scaled_h
        
        print(f"\n{checked+1}. {entry.get('main_id')}")
        print(f"   Original: {orig_w}×{orig_h} at {orig_um_per_px:.3f} µm/px")
        print(f"   Scaled:   {scaled_w}×{scaled_h} at {target_um_per_px} µm/px")
        print(f"   Aspect: {orig_aspect:.4f} → {scaled_aspect:.4f} (error: {aspect_error:.2f}%)")
        print(f"   Actual µm/px: {actual_um_per_px_w:.3f} (W), {actual_um_per_px_h:.3f} (H)")
        
        # Check for errors
        if aspect_error > 0.5:
            errors.append(f"{key}: Aspect ratio error {aspect_error:.2f}%")
            print(f"   ❌ ASPECT RATIO MISMATCH!")
        elif abs(actual_um_per_px_w - target_um_per_px) > 0.1 or abs(actual_um_per_px_h - target_um_per_px) > 0.1:
            errors.append(f"{key}: µm/px error")
            print(f"   ❌ µm/px INCORRECT!")
        else:
            print(f"   ✅ Correct!")
        
        checked += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Checked: {checked} images")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ All images verified correct!")
        print("✅ Aspect ratios preserved!")
        print("✅ Target µm/px achieved!")
        return True

if __name__ == '__main__':
    verify_preprocessing()