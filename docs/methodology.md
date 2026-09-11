# Detailed Methodology — Chest X-Ray Lung Segmentation, Explained



## 1. Preprocessing

**Pipeline: Percentile normalization → CLAHE → resize to 256×256 → Z-score normalization**

- **Percentile normalization**: raw X-ray pixel intensities can vary a lot between machines/exposures. This clips extreme outlier pixel values (e.g., the top/bottom 1-2%) before rescaling, so a single overexposed image doesn't distort the whole preprocessing pipeline.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): boosts local contrast in small tiles of the image rather than globally. Think of it like sharpening the detail *within* the rib cage and lung borders specifically, without blowing out contrast everywhere else. This helps because lung boundaries in a chest X-ray are often faint and low-contrast.
- **Resize to 256×256**: standardizes input size so every image can be batched together and fed through the same network architecture.
- **Z-score normalization**: `(pixel/255 − mean) / std` — centers pixel values around 0 with unit variance, which helps gradient-based optimizers (like Adam) converge faster and more stably.

**Critical detail — train-only statistics**: the mean/std used for Z-score are computed **only from the training set**, never from validation or test. Why this matters: if you computed these statistics over the *entire* dataset first, then split, information about the val/test distribution would leak into training indirectly (the model's inputs would already be "aware" of statistics from data it's not supposed to have seen). This is a classic and easy-to-miss form of **data leakage**.

## 2. Split

- **Stratified split**: 70% train / 15% validation / 15% test, stratified across the 4 diagnostic classes (COVID-19, Normal, Lung Opacity, Viral Pneumonia). Stratification ensures each split has roughly the same proportion of each class — so the test set isn't accidentally missing, say, Viral Pneumonia cases.
- **Reproducibility**: `random_state=42` is fixed, and the split is done in *cascade* — since scikit-learn's `train_test_split` only splits into 2 parts at a time, it's called twice (first splitting off train vs. the rest, then splitting the remainder into val/test).
- **Deduplication before splitting**: exact-hash and perceptual-hash duplicate detection is run **before** the split — not after. This is important: if you split first and dedupe later, a duplicate or near-duplicate image could end up in *both* the train and test sets, which would let the model effectively "memorize" a test image it saw during training — artificially inflating test scores. Deduping first guarantees no image (or near-identical variant of it) crosses the train/val/test boundary.

## 3. Data Augmentation

Applied **only to the training split** — never to validation or test, since those need to reflect real, un-augmented data to give a trustworthy performance estimate.

- **Geometric**: rotation, translation, scale — simulates natural variation in patient positioning during the X-ray.
- **Photometric**: brightness/contrast jitter, gamma correction — simulates variation in exposure settings across different X-ray machines.
- **Noise**: Gaussian noise, motion blur — simulates sensor noise or slight patient movement during acquisition.
## 4. Loss Function and Metrics

**Combined loss**: `dice_weight × DiceLoss + (1 − dice_weight) × CrossEntropy`

- **Dice Loss** directly optimizes for overlap between predicted and true masks — well-suited to segmentation, and robust to class imbalance (background pixels vastly outnumber lung pixels).
- **Cross-Entropy** provides per-pixel classification signal and tends to give more stable gradients early in training, when Dice-based gradients can be noisy.
- Combining both is a common practice: Dice for overlap quality, Cross-Entropy for training stability.
- `dice_weight` controls the balance between the two — found via Optuna to be **0.7** (leaning more toward Dice).

**Metrics tracked**: Dice coefficient (per class + mean), IoU (Intersection over Union), and pixel accuracy — all computed **per class** (background, right lung, left lung) rather than just as one global average, so you can catch a case where, say, the right lung is segmented well but the left lung is systematically worse.

## 5. Regularization

| Technique | Role |
|---|---|
| **BatchNorm** | Normalizes activations within each mini-batch during training, which stabilizes and speeds up convergence — also has a mild regularizing side-effect. |
| **Data augmentation** | The main defense against overfitting — exposes the model to more visual variation than the raw dataset alone provides. |
| **Weight decay (via Optuna)** | Adds an L2 penalty to the optimizer, discouraging excessively large weights, which tends to produce smoother, more generalizable models. |
| **Early Stopping** | Halts training once the validation metric stops improving — prevents continuing to train past the point of diminishing (or negative) returns. |
| **ReduceLROnPlateau** | Cuts the learning rate in steps once validation performance plateaus, allowing finer convergence in later training stages. |

## 6. Hyperparameter Optimization

- **Optuna (TPE sampler)** was used to search hyperparameters, with the objective of maximizing **Validation Soft Lung Dice** (the *soft*, probability-based Dice — before thresholding/argmax — which gives smoother, more informative gradients for the optimizer to compare across trials).
- **Parameters tuned**: learning rate, weight decay, batch size, dice weight.
- **Result**:
  - `learning_rate = 5.94e-05`
  - `weight_decay = 1.06e-07`
  - `batch_size = 4`
  - `dice_weight = 0.7`

## 7. Training

- **Optimizer**: Adam.
- **Mixed precision (AMP)**: via `torch.autocast` + `GradScaler` — runs parts of the forward/backward pass in lower precision (float16) instead of full float32, cutting memory usage and speeding up training, without materially hurting accuracy.

## 8. Explainability — Seg-Grad-CAM

**Seg-Grad-CAM** adapts the classic Grad-CAM technique (originally designed for classification) to segmentation tasks. Instead of computing gradients with respect to a single class-prediction score, it computes the gradient of the *sum of logits for a given class across the entire spatial map* — producing a heatmap over the image showing which regions most influenced the model's prediction for that class.

**Why this matters here**: it's used to verify the model is basing its lung predictions on actual lung anatomy — not shortcuts like:
- orientation letters ("L"/"R") printed on the image,
- the image's outer borders,
- external objects in frame,
- medical equipment visible in the scan.

If Seg-Grad-CAM showed the model paying attention to, say, the "L" letter in the corner rather than the lung tissue itself, that would be a red flag that the model learned a spurious shortcut rather than genuine anatomical understanding — a common and important failure mode to check for in medical imaging models.
