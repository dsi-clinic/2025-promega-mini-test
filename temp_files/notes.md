                       +------------------------+
                       | original_mapping.json  |
                       +------------------------+
                                   |
                  +----------------+-----------------+
                  |                                  |
+-----------------------------+     +-----------------------------+
| resize_img_masks_total.py   |     | preprocess_images.py        |
| (images + masks)            |     | (images only)               |
| resize to 296x198           |     | resize to 296x198           |
+-----------------------------+     +-----------------------------+
      |                                   |
+-----------------------------+     +-----------------------------+
| processed_dataset_256x192/  |     | image_mapping_BATCH_Dy...    |
|  - images/                  |     |   _processed.json            |
|  - masks/                   |     | (image-only mapping)         |
| log mask+image loc in json  |     | log scale um dim in json     |
|  - mapping_processed.json   |     +-----------------------------+
+-----------------------------+                 |
      |                                         |
+-----------------------------+                 |
| split_train_val_test        |                 |
+-----------------------------+                 |
      |                                         |
+-----------------------------+                 |
| mmseg training / predict    |<----------------+
+-----------------------------+