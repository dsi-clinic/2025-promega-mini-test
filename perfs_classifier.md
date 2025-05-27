| Models tracking                                                                                                                               | Batch Size | Data Subset     | Data Size & Composition                                                                | Validation Split | Epochs (Early Stopping) | Performance                                                                       |
|---------------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------------|----------------------------------------------------------------------------------------|------------------|----------------|-----------------------------------------------------------------------------------|
| Pretrained Resnet50 for images and simple model for masks + class weight + data augmentation + unfreeze 10 last layers for fine-tuning | 8 | **Day30** BA1&2 | 65+35 labeled organoids, image + mask + label (full agreement on class) | 20%              | 27+55 epochs      | 91% F1 on validation and F1 on training is 100%                |
| Usual Config but learning rate e-5 for Step 2 + unfreeze 15 last layers                             | 8          | **Day28** BA1&2 | 65+34 labeled organoids, image + mask + label (full agreement on class) | 20%              | 29+31 epochs      | 89% F1 on validation and F1 on training is 100%                |
| Usual Config but learning rate e-5 for Step 2 + unfreeze 15 last layers |8          | **Day24** BA1&2 | 67+35 labeled organoids, image + mask + label (full agreement on class) | 20%              | 24+31 epochs      | 79% F1 on validation and F1 on training is 99%                |
| Usual Config | 8          | **Day17** BA1&2 | 67+36 labeled organoids, image + mask + label (full agreement on class) | 20%              | 21+31 epochs      | 78% F1 on validation but F1 on training is 94%                 |
| Usual Config                                                                                      | 8          | **Day17** BA1&2 | 65+26 labeled organoids, image + mask + label (full agreement on class and quality (good)) | 20%              | 42+84 epochs      | 72% F1 on validation but F1 on training is 100% (overfitting?)                 |
| Usual Config                                                                                     | 8          | **Day15** BA1&2 | 67+35 labeled organoids, image + mask + label (full agreement on class) | 20%              | 24+48 epochs      | 77% F1 on validation but F1 on training is 96%                 |
| Usual Config                                                                                     | 8          | **Day15** BA1&2 | 65+26 labeled organoids, image + mask + label (full agreement on class and quality (good)) | 20%              | 24+48 epochs      | 67% F1 on validation but F1 on training is 100% (overfitting?)                 |
| Usual Config                                                                                     | 8          | **Day06** BA1&2 | 65+35 labeled organoids, image + mask + label (full agreement on class) | 20%              | 34+66 epochs      | 73% F1 on validation and 97% F1 on training                |
| Usual Config                                                                                     | 8          | **Day30** BA1&2 | 136+49 labeled organoids, image + mask + label (**Strong** agreement on class) | 20%              | 35+49 epochs      | 72% F1 on validation and 97% F1 on training                |
| Usual Config                                                                                     | 8          | **Day28** BA1&2 | 135+48 labeled organoids, image + mask + label (**Strong** agreement on class) | 20%              | 50+46 epochs      | 74% F1 on validation and 100% F1 on training                |
| Usual Config                                                                                     | 8          | **Day17** BA1&2 | 140+35 labeled organoids, image + mask + label (**Strong** agreement on class) | 20%              | 37+42 epochs      | 67% F1 on validation and 97% F1 on training                |
| Usual Config                                                                                     | 8          | **Day13** BA1&2 | 142+52 labeled organoids, image + mask + label (**Strong** agreement on class) | 20%              | 49+42 epochs      | 69% F1 on validation and 99% F1 on training                |
| Usual Config                                                                                     | 8          | **Day10** BA1&2 | 143+53 labeled organoids, image + mask + label (**Strong** agreement on class) | 20%              | 49+42 epochs      | 69% F1 on validation and 99% F1 on training                |
| Usual Config                                                                                     | 8          | **Day06** BA1&2 | 147+54 labeled organoids, image + mask + label (**Strong** agreement on class) | 20%              | 38+40 epochs      | 63% F1 on validation and 99% F1 on training                |


---

## Best models path:
* /net/projects2/promega/data-analysis/best_models

---

confusion matrix Day30 Complete Agreement on class:

[[ 6  1]

 [ 1 12]]

---

confusion matrix Day28 Complete Agreement on class:


[[ 6  1]

 [ 2 11]]

---

confusion matrix Day17 Complete Agreement on class AND quality:

[[2 3]

 [5 9]]

---


 Confusion Matrix Dy15 on class AND quality:

[[ 1  4]

 [ 2 12]]

---

  Confusion Matrix Dy06 Complete Agreement on class:

[[ 5  3]

 [ 3 11]]

---


confusion matrix Day30 Strong Agreement:

[[ 2  8]

 [ 0 27]]
 
---

confusion matrix Day06 Strong Agreement:

[[ 7  4]

 [16 14]]
 
---