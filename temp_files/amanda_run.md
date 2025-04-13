**Error Message:**
cv2.error: OpenCV(4.11.0) :-1: error: (-5:Bad argument) in function 'imdecode'
> Overload resolution failed:
>  - buf is not a numpy array, neither a scalar
>  - Expected Ptr<cv::UMat> for argument 'buf'
**Resolution:** Downgraded OpenCV to version 4.5.5.64.

**Error Message:**
RuntimeError: sigmoid_focal_loss_forward_impl: implementation for device cuda:0 not found.
**Cause:** mmcv-full is needed it seems to run FocalLoss (I can only get mmcv).
**Resolution:**
- Commented out the FocalLoss from the configuration.

**Error Message:**
RuntimeError: The detected CUDA version (12.4) mismatches the version that was used to compile PyTorch (11.3)
**Cause:** Attempted to install mmcv-full, which requires a specific CUDA version that matches the one used to compile PyTorch.
**Resolution:**
- Unable to install mmcv-full due to CUDA version mismatch.
- Opted to install the standard mmcv package instead.
- Removed dependencies on mmcv-full by adjusting the loss functions used.

**Error Message:**
  Processing sample 10/10 (ID: Ba1 96_1 Dy30 G10)...
    -> Skipped: Error during inference or processing for ID Ba1 96_1 Dy30 G10: 'ConfigDict' object has no attribute 'test_pipeline'
No samples were successfully processed. Cannot create collage.
**Cause:** The config file used for inference was generated from training and didn’t include a test_pipeline. The MMSeg inference_model() API expects this attribute to exist.
**Solution:** 
Added test_pipeline = val_pipeline into the end of the config.py 

**All runs**
- Environment: `mmcv_env`  
- Model: SegFormer (MiT-B0 backbone)  
- Dataset: Binary segmentation (background vs. cell)  
- Epochs: 1 (iteration-based loop)  
- Batch size: 2  

**Run 1 — DiceLoss Only**
20250412_190516
Loss Function: DiceLoss (weight = 10.0)
Runtime: ~3 min 45 sec
Avg. Iteration Time: ~0.32 sec
Iterations: 1000
Batch Size: 2

Metric	Values:
mIoU	75.16%
aAcc	86.40%
Cell IoU	70.45%
Background IoU	79.87%

**Run 2 — DiceLoss (10.0) + BCELoss (1.0)**
20250413_113210
Runtime: ~3 min 45 sec
Avg. Iteration Time: ~0.32 sec
Iterations: 1000
Batch Size: 2

Metric	Values:
mIoU	11.41%
aAcc	22.69%
Cell IoU	22.47%
Background IoU	0.36%

**Run 3 — DiceLoss (5.0) + BCELoss (5.0)**
20250413_115230
Runtime: ~3 min 45 sec
Avg. Iteration Time: ~0.32 sec
Iterations: 1000
Batch Size: 2

Final Validation:
mIoU plateaued at ~20.94%
Background IoU remained 0%

**Run 4 — DiceLoss (10.0) + BCELoss (0.5)**
20250413_120232
Runtime: ~3 min 45 sec
Avg. Iteration Time: ~0.32 sec
Iterations: 1000
Batch Size: 2

Metric	Values:
mIoU	20.94%
aAcc	41.87%
Cell IoU	41.87%
Background IoU	0.00%

**Run 5 — DiceLoss (10.0) + BCELoss (0.1)**
20250413_123718
Runtime: ~3 min 45 sec
Avg. Iteration Time: ~0.32 sec
Iterations: 1000
Batch Size: 2

Metric	Values:
mIoU	9.01%
aAcc	17.45%
Cell IoU	16.09%
Background IoU	1.93%

**Full env list:**
Package                Version
---------------------- -----------
addict                 2.4.0
aliyun-python-sdk-core 2.16.0
aliyun-python-sdk-kms  2.16.5
certifi                2025.1.31
cffi                   1.17.1
charset-normalizer     3.4.1
click                  8.1.8
colorama               0.4.6
contourpy              1.0.5
crcmod                 1.7
cryptography           44.0.2
cycler                 0.12.1
filelock               3.14.0
fonttools              4.57.0
ftfy                   6.3.1
idna                   3.10
importlib_metadata     8.6.1
importlib_resources    6.5.2
jmespath               0.10.0
kiwisolver             1.4.7
Markdown               3.7
markdown-it-py         3.0.0
matplotlib             3.4.3
mdurl                  0.1.2
mkl_fft                1.3.11
mkl_random             1.2.8
mkl-service            2.4.0
mmcv                   2.0.0
mmengine               0.10.7
mmsegmentation         1.2.2
model-index            0.1.11
numpy                  1.20.3
opencv-python          4.5.5.64
opendatalab            0.0.10
openmim                0.3.9
openxlab               0.1.2
ordered-set            4.1.0
oss2                   2.17.0
packaging              24.2
pandas                 2.2.3
pillow                 11.1.0
pip                    25.0.1
platformdirs           4.3.7
prettytable            3.16.0
pycparser              2.22
pycryptodome           3.22.0
Pygments               2.19.1
pyparsing              3.2.3
pytest-runner          6.0.1
python-dateutil        2.9.0.post0
python-dotenv          1.1.0
pytz                   2023.4
PyYAML                 6.0.2
regex                  2024.11.6
requests               2.28.2
rich                   13.4.2
scipy                  1.7.0
setuptools             60.2.0
six                    1.17.0
tabulate               0.9.0
termcolor              3.0.1
tomli                  2.2.1
torch                  1.10.0
torchaudio             0.10.0
torchvision            0.11.1
tqdm                   4.65.2
typing_extensions      4.12.2
tzdata                 2025.2
urllib3                1.26.20
wcwidth                0.2.13
wheel                  0.45.1
yapf                   0.43.0
zipp                   3.21.0


