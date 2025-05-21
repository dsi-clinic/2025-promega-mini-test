| Models tracking                                                                                                                               | Batch Size | Data Subset     | Data Size & Composition                                                                | Validation Split | Early Stopping | Performance                                                                       |
|---------------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------------|----------------------------------------------------------------------------------------|------------------|----------------|-----------------------------------------------------------------------------------|
| Pretrained Resnet50 for images and simple model for masks                                                                                       | 8          | **Day30** BA1&2 | 70 labeled organoids, image + mask + label (full agreement on class and quality (good)) | 20%              | 22 epochs      | 85% F1 on validation but F1 on training is 100% (overfitting?)                 |
| Config above + class weight + data augmentation + unfreeze 10 last layers for fine-tuning                                                                                      | 8          | **Day30** BA1&2 | 70 labeled organoids, image + mask + label (full agreement on class and quality (good)) | 20%              | 38+39 epochs      | 76% F1 on validation but F1 on training is 100% (overfitting?)                 |
| Config above                                                                                     | 8          | **Day17** BA1&2 | 70 labeled organoids, image + mask + label (full agreement on class and quality (good)) | 20%              | 42+84 epochs      | 72% F1 on validation but F1 on training is 100% (overfitting?)                 |
| Config above                                                                                     | 8          | **Day15** BA1&2 | 70 labeled organoids, image + mask + label (full agreement on class and quality (good)) | 20%              | 24+48 epochs      | 67% F1 on validation but F1 on training is 100% (overfitting?)                 |
| Config above                                                                                     | 8          | **Day06** BA1&2 | 70 labeled organoids, image + mask + label (full agreement on class and quality (good)) | 20%              | 34+35 epochs      | 57% F1 on validation but F1 on training is 100% (overfitting?)                 |



 ---

confusion matrix Day30:


 [[ 1 4] 
 
 [ 0 14]]

---

confusion matrix Day17:

[[2 3]

 [5 9]]

---


 Confusion Matrix Dy15:

[[ 1  4]

 [ 2 12]]

---

  Confusion Matrix Dy06:

[[ 1  5]

 [ 2 12]]

---