# Data Description 

## Dataset Overview

All datasets span **Day 1 to Day 30**, covering **11 time points**.

- **4 experimental batches**
- **96 organoids per batch**
- Not all organoids contain complete records — some are filtered out during cleaning.


## 1. Image Data  

Microscopy images were collected for each organoid across days.

- Images were **resized** before model training.
- Two main formats:
  - **Split images** — multiple cropped views of the organoid
  - **Stitched images** — combined image from multiple views

## 2. Metabolite Data  

Metabolite data captures biochemical activity through assay readouts, including:

- `GlucoseGlo`
- `GlutamateGlo`
- `MalateGlo`
- `LactateGlo`
- `PyruvateGlo`
- `BCAAGlo` *(excluded from modeling)*

## 3. Survey Label Data  

Each organoid–day pair was evaluated by **five domain experts**.

- Each expert independently assigned a binary label:
  - **Acceptable** — healthy or viable growth
  - **Not Acceptable** — poor quality or likely failure

To construct the final label:

- Only **majority-agreement** cases were kept.

## Data Engineering and Cleaning  

### Batch Filtering  

The raw data contained **four batches** of experiments.

- **Batches 3 and 4 were removed** from analysis due to reported quality issues.
- Only the remaining batches were used in training and evaluation.

### Image Cleaning  

- Images were **resized** to a consistent resolution suitable for the image classifier.
- Split and stitched images were handled as part of the same dataset, but model performance was interpreted with these sources of noise in mind.

### Metabolite Cleaning  

Steps applied to metabolite features:

- **Drop near-constant columns** that carry little to no signal.
- **Fill missing values** with `0` as a simple and consistent placeholder.
- **Filter out unreliable features**:
  - Remove `BCAAGlo` entirely.
  - Remove early-day `MalateGlo` values (pre–Day 10).
- **Add growth-related features**, such as day-to-day changes in metabolite levels for each organoid.
