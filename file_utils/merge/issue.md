We want to track a well through time with multiple sources of data, a well has an organoid and sometimes the organoid can split (split_1 or split_2) but those splits are still identified as the same (i.e. the parent well)
We want a map that retains all the information for the organoid itself, I am not sure if the best way to do this is organoid or well centric. Organoid centric we would repeat the parent data for each split (dayID, BA, wellID, classification, cell line, treatment, and metabolite) but a well centric view would just have that parent info and then split entries with the organoid specific information (best Z, best z filename, um_per_px, survey data, infer_resized processed sizes and locations). I think the problem with the well centric approach is that we are pulling and mixing entries in these edge cases. 

These are each data source and an example of each case:
image_mapping.json (env ORIGINAL_MAPPING)
Can have 1 or 2 entries for a well. 
    "BA2 96_1 Dy08 H5": {
      "dayID": "Dy08",
      "BA": "BA2 96_1",
      "wellID": "H5",
      "Best Z": 2,
      "Best Z Filename": "BA2/96_1/Dy08/Ba2 96_1 Dy08 H5 Z2.tif",
      "Actual Z Value": 2,
      "Classification": "Regular",
      "um_per_px": 1.6870567375886525,
      "all_files": [
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H5 Z0.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H5 Z1.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H5 Z2.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H5 Z3.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H5 Z4.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H5 Z5.tif"
      ],
      "cellLine": "GM23279A",
      "treatment": NaN,
      "Blank": false,
      "blank_area_frac": 0.11379623413085938
    },
    "BA2 96_1 Dy08 H6 split_1": {
      "dayID": "Dy08",
      "BA": "BA2 96_1",
      "wellID": "H6",
      "split_index": 1,
      "Best Z": -1,
      "Best Z Filename": "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(1)% Z3.tif",
      "Actual Z Value": 3,
      "Classification": "Split",
      "um_per_px": 1.6870567375886525,
      "all_files": [
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(1)% Z0.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(1)% Z1.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(1)% Z2.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(1)% Z3.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(1)% Z4.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(1)% Z5.tif"
      ],
      "cellLine": "GM23279A",
      "treatment": NaN,
      "Blank": false,
      "blank_area_frac": 0.12238693237304688
    },
    "BA2 96_1 Dy08 H6 split_2": {
      "dayID": "Dy08",
      "BA": "BA2 96_1",
      "wellID": "H6",
      "split_index": 2,
      "Best Z": -1,
      "Best Z Filename": "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(2)% Z1.tif",
      "Actual Z Value": 1,
      "Classification": "Split",
      "um_per_px": 1.6870567375886525,
      "all_files": [
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(2)% Z0.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(2)% Z1.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(2)% Z2.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(2)% Z3.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(2)% Z4.tif",
        "BA2/96_1/Dy08/Ba2 96_1 Dy08 H6(2)% Z5.tif"
      ],
      "cellLine": "GM23279A",
      "treatment": NaN,
      "Blank": false,
      "blank_area_frac": 0.11614608764648438
    },

    env location (INFER_RESIZED_DIR)
      "BA2 96_1 Dy08 H5": {
    "img_path": "/net/projects2/promega/data-analysis/output/infer_resized_512x384/auto_processed/ba296_1_Dy08/BA2_96_1_Dy08_H5.png",
    "orig_width_px": 1128,
    "orig_height_px": 832,
    "orig_um_per_px_x": 1.6870567375886525,
    "orig_um_per_px_y": 1.6870567375886525,
    "final_um_per_px_x": 3.716796875,
    "final_um_per_px_y": 3.6552895981087468,
    "mask_path": "/net/projects2/promega/data-analysis/predictions/batch2_96_1/day08/predicted_masks/BA2_96_1_Dy08_H5_predmask.png"
  },
  "BA2 96_1 Dy08 H6 split_1": {
    "img_path": "/net/projects2/promega/data-analysis/output/infer_resized_512x384/auto_processed/ba296_1_Dy08/BA2_96_1_Dy08_H6_split_1.png",
    "orig_width_px": 1128,
    "orig_height_px": 832,
    "orig_um_per_px_x": 1.6870567375886525,
    "orig_um_per_px_y": 1.6870567375886525,
    "final_um_per_px_x": 3.716796875,
    "final_um_per_px_y": 3.6552895981087468,
    "mask_path": "/net/projects2/promega/data-analysis/predictions/batch2_96_1/day08/predicted_masks/BA2_96_1_Dy08_H6_split_1_predmask.png"
  },
  "BA2 96_1 Dy08 H6 split_2": {
    "img_path": "/net/projects2/promega/data-analysis/output/infer_resized_512x384/auto_processed/ba296_1_Dy08/BA2_96_1_Dy08_H6_split_2.png",
    "orig_width_px": 1128,
    "orig_height_px": 832,
    "orig_um_per_px_x": 1.6870567375886525,
    "orig_um_per_px_y": 1.6870567375886525,
    "final_um_per_px_x": 3.716796875,
    "final_um_per_px_y": 3.6552895981087468,
    "mask_path": "/net/projects2/promega/data-analysis/predictions/batch2_96_1/day08/predicted_masks/BA2_96_1_Dy08_H6_split_2_predmask.png"
  },


env location (METABOLITE_MAP_JSON)
  "BA2 96_1 Dy08 H5": {
    "GlucoseGlo": {
      "concentration_uM": 8.851,
      "initial_concentration": 17702.567,
      "is_outlier": false,
      "well_384": "P10"
    },
    "GlutamateGlo": {
      "concentration_uM": 1.772,
      "initial_concentration": 177.198,
      "is_outlier": false,
      "well_384": "O10"
    },
    "MalateGlo": {
      "concentration_uM": 0.048,
      "initial_concentration": 0.95,
      "is_outlier": false,
      "well_384": "P9"
    },
    "BCAAGlo": {
      "concentration_uM": 5.096,
      "initial_concentration": 2038.449,
      "is_outlier": false,
      "well_384": "P10"
    },
    "LactateGlo": {
      "concentration_uM": 3.164,
      "initial_concentration": 1265.512,
      "is_outlier": false,
      "well_384": "P9"
    },
    "PyruvateGlo": {
      "concentration_uM": 3.035,
      "initial_concentration": 303.527,
      "is_outlier": false,
      "well_384": "O10"
    }
  },
  "BA2 96_1 Dy08 H6": {
    "GlucoseGlo": {
      "concentration_uM": 8.729,
      "initial_concentration": 17458.39,
      "is_outlier": false,
      "well_384": "P12"
    },
    "GlutamateGlo": {
      "concentration_uM": 1.754,
      "initial_concentration": 175.423,
      "is_outlier": false,
      "well_384": "O12"
    },
    "MalateGlo": {
      "concentration_uM": 0.06,
      "initial_concentration": 1.203,
      "is_outlier": false,
      "well_384": "P11"
    },
    "BCAAGlo": {
      "concentration_uM": 5.439,
      "initial_concentration": 2175.76,
      "is_outlier": false,
      "well_384": "P12"
    },
    "LactateGlo": {
      "concentration_uM": 3.208,
      "initial_concentration": 1283.387,
      "is_outlier": false,
      "well_384": "P11"
    },
    "PyruvateGlo": {
      "concentration_uM": 2.697,
      "initial_concentration": 269.679,
      "is_outlier": false,
      "well_384": "O12"
    }
  },

  env location (SURVEY_AGGREGATED_JSON)

    "Organoid_255": {
    "evaluations": [
      {
        "image_id": "Ba2 96_2 Dy30 C8",
        "source_file": "Organoid Classification (Form B) - Part 1 of 3 - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_2",
        "dayID": "Dy30",
        "wellID": "C8",
        "split_index": 2,
        "evaluation": "Not Acceptable",
        "employee": "Stevens Rehen"
      },
      {
        "image_id": "Ba2 96_2 Dy30 C8",
        "source_file": "Organoid Classification (Form B) - Part 1 of 3 - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_2",
        "dayID": "Dy30",
        "wellID": "C8",
        "split_index": 2,
        "evaluation": "Acceptable",
        "employee": "Bruna Paulsen"
      },
      {
        "image_id": "Ba2 96_2 Dy30 C8",
        "source_file": "Organoid Classification (Form A) - Part 3 of 3 - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_2",
        "dayID": "Dy30",
        "wellID": "C8",
        "split_index": 2,
        "evaluation": "Not Acceptable",
        "employee": "Beatriz Guimaraes"
      },
      {
        "image_id": "Ba2 96_2 Dy30 C8",
        "source_file": "Organoid Classification (Form C) - Part 2 of 3 - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_2",
        "dayID": "Dy30",
        "wellID": "C8",
        "split_index": 2,
        "evaluation": "Not Acceptable",
        "employee": "Livia Goto Silva"
      },
      {
        "image_id": "Ba2 96_2 Dy30 C8",
        "source_file": "Organoid Classification (Form C) - Part 2 of 3 - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_2",
        "dayID": "Dy30",
        "wellID": "C8",
        "split_index": 2,
        "evaluation": "Not Acceptable",
        "employee": "Matheus Victor"
      }
    ],
    "quality_scores": [
      {
        "image_id": "Ba2 96_2 Dy30 C8",
        "source_file": "Image Classification Form - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_2",
        "dayID": "Dy30",
        "wellID": "C8",
        "split_index": 2,
        "quality": "Good"
      }
    ]
  },
  "Organoid_126": {
    "evaluations": [
      {
        "image_id": "Ba2 96_1 Dy30 B2",
        "source_file": "Organoid Classification (Form B) - Part 1 of 3 - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_1",
        "dayID": "Dy30",
        "wellID": "B2",
        "evaluation": "Acceptable",
        "employee": "Stevens Rehen"
      },
      {
        "image_id": "Ba2 96_1 Dy30 B2",
        "source_file": "Organoid Classification (Form B) - Part 1 of 3 - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_1",
        "dayID": "Dy30",
        "wellID": "B2",
        "evaluation": "Not Acceptable",
        "employee": "Bruna Paulsen"
      },
      {
        "image_id": "Ba2 96_1 Dy30 B2",
        "source_file": "Organoid Classification (Form A) - Part 3 of 3 - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_1",
        "dayID": "Dy30",
        "wellID": "B2",
        "evaluation": "Acceptable",
        "employee": "Beatriz Guimaraes"
      },
      {
        "image_id": "Ba2 96_1 Dy30 B2",
        "source_file": "Organoid Classification (Form C) - Part 2 of 3 - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_1",
        "dayID": "Dy30",
        "wellID": "B2",
        "evaluation": "Acceptable",
        "employee": "Livia Goto Silva"
      },
      {
        "image_id": "Ba2 96_1 Dy30 B2",
        "source_file": "Organoid Classification (Form C) - Part 2 of 3 - Excel Report(2025-06-13).xlsx",
        "BA": "BA2 96_1",
        "dayID": "Dy30",
        "wellID": "B2",
        "evaluation": "Not Acceptable",
        "employee": "Matheus Victor"
      }
    ],

    