The data merging script is systematically missing all `split_1` entries while successfully processing `split_2` entries, despite both types of data existing in the source JSON files.

All upstream maps to me look good, surveys and image mapper properly split indexed. Metabolites (parent wells) are fine. 

Errors -->
[MERGE] Parent BA4 96_1 Dy21 C12 has children: ['BA4 96_1 Dy21 C12 split_2', 'BA4 96_1 Dy21 C12 split_1']
[MERGE] No processed data found for split: BA4 96_1 Dy21 C12 split_1
  Similar keys in processed_map: ['BA4 96_1 Dy21 C12 split_2']
[MERGE] Added 512x384 data to split: BA4 96_1 Dy21 C12 split_2
[MERGE] Parent BA4 96_1 Dy24 C12 has children: ['BA4 96_1 Dy24 C12 split_1', 'BA4 96_1 Dy24 C12 split_2']

the data exists, images and masks are there and fine, i don't see any difference in the file structure/naming -->

    "BA4 96_1 Dy21 C12 split_1": {
    "img_path": "/net/projects2/promega/data-analysis/output/infer_resized_512x384/auto_processed/ba496_1_Dy21/BA4_96_1_Dy21_C12_split_1.png",
    "orig_width_px": 1128,
    "orig_height_px": 832,
    "orig_um_per_px_x": 2.019503546099291,
    "orig_um_per_px_y": 2.019503546099291,
    "final_um_per_px_x": 4.44921875,
    "final_um_per_px_y": 4.375591016548463,
    "mask_path": "/net/projects2/promega/data-analysis/predictions/batch4/day21/predicted_masks/BA4_96_1_Dy21_C12_split_1_predmask.png"
  },
  "BA4 96_1 Dy21 C12 split_2": {
    "img_path": "/net/projects2/promega/data-analysis/output/infer_resized_512x384/auto_processed/ba496_1_Dy21/BA4_96_1_Dy21_C12_split_2.png",
    "orig_width_px": 1128,
    "orig_height_px": 832,
    "orig_um_per_px_x": 2.019503546099291,
    "orig_um_per_px_y": 2.019503546099291,
    "final_um_per_px_x": 4.44921875,
    "final_um_per_px_y": 4.375591016548463,
    "mask_path": "/net/projects2/promega/data-analysis/predictions/batch4/day21/predicted_masks/BA4_96_1_Dy21_C12_split_2_predmask.png"
  },


### What We Know Works
- `split_2` entries are processed correctly and appear in the final merged data
- All source JSON files contain both `split_1` and `split_2` entries with valid data
- The normalization and key correction logic works for both types

### What's Failing
- `split_1` entries are being loaded from JSON files but not making it into `processed_map`
- This causes merge failures: `[MERGE] No processed data found for split: BA4 96_1 Dy17 C12 split_1`
- Pattern is 100% consistent - ALL `split_1` entries fail, ALL `split_2` entries succeed

### Example Data (Confirmed to Exist in Source)
```json
"BA4 96_1 Dy17 C12 split_1": {
  "img_path": "/net/.../BA4_96_1_Dy17_C12_split_1.png",
  "mask_path": "/net/.../BA4_96_1_Dy17_C12_split_1_predmask.png"
},
"BA4 96_1 Dy17 C12 split_2": {
  "img_path": "/net/.../BA4_96_1_Dy17_C12_split_2.png", 
  "mask_path": "/net/.../BA4_96_1_Dy17_C12_split_2_predmask.png"
}
```

What Claude says...

**UPDATE**: After reviewing the `OrganoidNormalizer` code, the `extract_resolution()` method appears correct and should work for both `split_1` and `split_2` paths. This suggests the issue is elsewhere in the processing pipeline.

The problem occurs during the "Stage 2: Processed masks and images" phase where `split_1` entries are being filtered out before reaching `processed_map`, while `split_2` entries pass through successfully.

### Code Flow Analysis
1. JSON files are loaded successfully ✓
2. Both `split_1` and `split_2` keys are found in the raw data ✓  
3. Key normalization works for both ✓
4. **Something systematically filters out `split_1` entries** ❌
5. Only `split_2` entries are added to `processed_map` ❌
6. Merge phase can't find the missing `split_1` entries ❌

### Revised Hypothesis
The issue is likely NOT in `OrganoidNormalizer.extract_resolution()` but could be:
- **Hidden exception handling**: `split_1` entries might be throwing exceptions that are caught and ignored
- **Data format differences**: The actual JSON data for `split_1` entries might be malformed 
- **Code path bug**: There may be a conditional that incorrectly skips `split_1` entries
- **Debugging artifacts**: The debug code itself might have introduced a bug that affects only `split_1` entries

## Debugging Attempts
- Added extensive debug logging to trace the issue
- Confirmed source data exists and is valid
- Verified key normalization works correctly
- Isolated the problem to the resolution extraction step
- The issue is 100% reproducible and affects all `split_1` entries

## Next Steps
1. **Immediate**: Review `OrganoidNormalizer.extract_resolution()` implementation
2. **Debug**: Add logging inside the resolution extraction method to see exact failure point
3. **Test**: Compare path processing between `split_1` and `split_2` entries character-by-character
4. **Fix**: Modify the regex/parsing logic to handle both suffixes consistently
