**PART I: Feedback on Current Draft**
# What looks decent now:
- **Experimental design.** Shared train/val/test splits across modalities, per-day classifiers that respect temporal information availability, and consistent evaluation metrics.
- **Metabolite results.** LightGBM hitting 0.94 balanced accuracy at Day 30 from culture medium alone is a strong result. The feature importance trajectory (lactate mid-stage, malate late-stage) has real biological content.
- **Temporal comparison.** Showing that predictive signal changes across development — images strong early/mid, metabolites dominant late — is useful and not obvious.
# Paper draft review (higher level):
Paper says what was done but doesn’t tell a story. Missing throughout: What is the question? What did we find? Why does it matter? Right now it answers “What did we try?”
# Suggestions on how to rework the paper:
## 1. Specify brain organoids and frame a real research question
- The dataset is brain organoids, current draft talks about generic “organoids.” Brain organoids have specific challenges (high heterogeneity, long maturation, morphological ambiguity early) that make non-destructive quality prediction especially relevant. Narrowing the scope would make a stronger, more publishable paper rather than just a definition on organoids.
- Current framing (“can we predict if an organoid is Acceptable?”) is procedural. The question should be about what we learn about brain organoid development from these predictions, not just whether the classifier works.
## 2. Separate what was done from what was found
Methods, results, and interpretation are mixed together throughout. Hyperparameters and training decisions sit alongside findings. Separate them:
- **Methods:** i.e backbone selection, training procedures, feature engineering, evaluation protocol.
- **Results:** What happened at each stage, how modalities compared, where prediction succeeded and failed.
- **Discussion:** Why do metabolites become informative late? Why doesn’t fusion help? What does this mean for screening workflows?
## 3. Add biological interpretation
Results are described numerically but never explained. Some questions that need answers:
- Why do metabolites gain predictive power around Day 24? Does this align with known brain organoid maturation milestones?
- What does the lactate → malate feature importance shift mean metabolically?
- Why does the combined model not outperform single modalities?
# Keep vs. Rework

| **Keep / Build On**                                                                                                                                                                                                             | **Rework Substantially**                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Experimental design and split strategy<br><br>LightGBM metabolite results + feature importance<br><br>Per-day vs. time-series comparison<br><br>Three-model comparison figures<br><br>Image quality documentation (for Methods) | Entire introduction (brain organoids + question)<br><br>Literature review (argument, not survey)<br><br>Data section (add biology, move methods out)<br><br>Results framing (add interpretation)<br><br>Label propagation (needs prominent justification)<br><br>Add a real Discussion section (doesn’t exist yet) |
**PART II: Proposed Path Forward**
# Central Claim
- Non-destructive longitudinal measurements can predict brain organoid quality before endpoint assessment, but the informative modality shifts across development.
- Brightfield morphology provides discriminative signal from early developmental stages, while culture medium metabolites become the strongest predictor at later stages — offering a scalable, low-cost alternative to repeated expert review or destructive assays.
# Proposed Paper Structure
**1. Introduction (~4 paragraphs)**
Brain organoids matter → quality is a problem → here’s the specific question. ~4 paragraphs.
- **Open with brain organoids.** Brain organoids specifically: what they recapitulate, why they’re valuable, why they’re particularly prone to heterogeneity.
- **The quality problem.** Current QC is destructive (IF, sequencing) or subjective (expert scoring). Both are expensive, endpoint-only, and don’t scale.
- **Why longitudinal and multimodal.** The question isn’t just “can we predict quality” but “when does each measurement type become informative, and which is most useful at each stage?”
- **Research question.** “We ask whether non-destructive imaging and culture medium biochemistry, collected longitudinally, can predict eventual quality — and how the predictive window differs between modalities.”

**2. Related Work (shortened, argument-driven)**
- Image-based organoid/cell screening: active field, mostly single-timepoint. Establish the gap for longitudinal work, especially brain organoid-specific.
- Metabolomics in organoid systems: biological case that culture medium chemistry reflects health. Underexplored for brain organoids as a predictive modality?
- Multimodal prediction: brief. Transition to the gap — no systematic comparison across the developmental timeline for brain organoids.

**3. Data**
- 265 (more with BA4?) brain organoids, 12 time points (Day 3–30), three modalities.
- Brightfield imaging: brief description, note challenges (shadowing, stitching) without pipeline details.
- Metabolites: introduce the five analytes biologically. What does each reflect about cellular metabolism?
- Expert labels: 5 raters, 80% consensus. Report Fleiss’ kappa. Discuss vote distribution.
- Label propagation: justify prominently. Acknowledge the assumption and its implications.
- Class distribution: 72.5/27.5. Motivate balanced accuracy.

**4. Methods**
- Image preprocessing: overlay construction, segmentation, resizing, normalization.
- Image models: EfficientNet-B0 (justify; backbone comparison to supplementary). Per-day vs. time-series. Training details.
- Metabolite models: LightGBM on concentrations + day-to-day differences. Logistic regression baseline.
- Combined model: EfficientNet features → PCA + metabolite features → LightGBM. Justify design, acknowledge limitations.
- Evaluation: shared splits, balanced accuracy, threshold analysis.

**5. Results**
- **Morphology is informative early/mid.** Image prediction provides signal from Day 6 onward, peaks mid-development. Per-day outperforms time-series. Day 17 drop.
- **Metabolites dominate late.** LightGBM strong by Day 24, peaks at Day 30. Feature importance shifts: glutamate/lactate early → lactate differences mid → malate differences late.
- **Combined model: honest negative result [could change when we run again?].** Fusion doesn’t outperform best single modality. Frame as informative negative result: correlated errors or fusion method too crude.
- **The informative modality shifts.** Images cover early/mid window, metabolites dominate late. Practical implication: staged screening workflow.

**6. Discussion**
- Biological interpretation of the temporal shift: why do metabolites gain power at Day 24? Maturation milestones?
- Feature importance: what does lactate → malate shift mean? Glycolysis to TCA transition?
- Why doesn’t fusion help? Error overlap analysis.
- Translational angle: metabolite assays as cheap, scalable QC. Proposed screening workflow.
- Limitations: dataset size, single lab, label propagation assumption, fusion simplicity.
# What We Still Need
To resolve before the next draft.

| **Item**                                                                      | **Status**                  |
| ----------------------------------------------------------------------------- | --------------------------- |
| Fleiss’ kappa on the 5-rater survey data                                      | **Needed**                  |
| Vote split distribution (5-0 through 3-2) with examples                       | **Needed**                  |
| Error overlap: do image and metabolite models misclassify the same organoids? | **Needed**                  |
| Performance stratified by vote confidence (unanimous vs. borderline)          | **Needed**                  |
| Day 17 investigation (media change? transition? imaging issue?)               | **Needed**                  |
| Biological context for metabolite feature importance                          | **Needed**                  |
| Updated lit review: brain organoid QC, morphology-based screening field       | **Needed**                  |
| Backbone comparison moved to supplementary                                    | **In draft (needs rework)** |
| Confirm Day 26 — exists in any data or phantom? Why is it mentioned?          | **Needed**                  |
| Label propagation justification written up prominently                        | **Needed**                  |
| Target journal decision                                                       | **Needed**                  |
| Three-model comparison figure                                                 | **Exists**                  |
| Feature importance figure                                                     | **Exists**                  |
| Per-day vs. time-series figure                                                | **Exists**                  |

# Open Questions:
- Combined model needs a decision. Right now there’s no rationale for why fusion should help — no analysis of whether the two modalities fail on different organoids, no hypothesis about what complementary signal looks like. The design (PCA + concatenation + LightGBM) is generic. And the result confirms it doesn’t help. Options: (a) do the error overlap analysis, and if the modalities really do fail on different subsets, redesign the fusion and keep it; (b) if they fail on the same organoids, demote to a paragraph that says “we tried it, it didn’t help, here’s why” — which actually strengthens the temporal shift story (the modalities are informative at different times, not simultaneously complementary); (c) drop it entirely.
- Combined model: keep as full result, compress to a paragraph, or drop unless results improve?
- Time-series underperformance: worth trying a better temporal architecture, or accept the negative result?
- Target journal? (Cell Systems?)
- Include a “proposed screening workflow” figure showing how this could