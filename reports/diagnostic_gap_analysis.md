# Diagnostic Gap & Bottleneck Analysis — KDEF Facial Emotion Recognition (`Hoc-May`)

**Author:** `audit-gap` (Data, Preprocessing & Gap Diagnostic Researcher)  
**Target Repository:** `f:\Hoc-May`  
**Reference Baseline:** `notebooks/CNN_KDEF_FER.ipynb` (`KDEFNet`, 1,241,191 params) on `data/official-kdef-dataset-by-ksatrajit0`  
**Baseline Metrics:** Train Macro-F1 `0.955` | Val Macro-F1 `0.8798` (Ep 81) | Test Accuracy `82.76%` (`811/980`) | Test Macro-F1 `82.75%` (`0.8275`) | Class Recall Spread `27.86%`

---

## 1. Environment & Compute Probe (Windows vs. WSL2 GPU)

### 1.1 Diagnostic Findings
| Environment | Python Executable | PyTorch / CUDA Status | Bottleneck / Verdict |
| :--- | :--- | :--- | :--- |
| **Windows Host** | `F:\anaconda3\envs\hocmay_env\python.exe` | **BLOCKED** (`OSError: [WinError 4551]`) | Windows Smart App Control / Application Control policy (WDAC) blocks loading `torch\lib\shm.dll` and `c10.dll`. **Do NOT use Windows Python for PyTorch.** |
| **WSL2 (`Ubuntu-22.04`)** | `/home/dhp01/kdef_env/bin/python` *(or `/home/dhp01/Microduck/.venv/bin/python`)* | **ACTIVE** (`torch 2.9.1+cu128`, CUDA `True`, **NVIDIA GeForce RTX 4060 Laptop GPU**) | Native CUDA 12.8 execution works at full speed. |
| **Filesystem I/O** | `/mnt/f/Hoc-May/data/official-kdef-dataset-by-ksatrajit0` vs. WSL2 ext4 (`/tmp/kdef_official`) | **9p DrvFs Overhead** (~25–40s/epoch on `/mnt/f`) vs. **WSL2 Native ext4** (<1.5s/epoch) | Reading 4,898 `.JPG` files across the Windows-WSL2 `/mnt/f` mount causes severe I/O stalls. Caching the dataset to `/tmp/kdef_official` (or RAM/tensor cache) on WSL2 ext4 speeds up data loading by **15–25x**. |

### 1.2 Fast GPU Execution Commands for `@arch-opt`, `@train-opt`, and `@core-integrator`

1. **One-time Dataset Cache into WSL2 Native `ext4` (`/tmp/kdef_official`):**
   ```powershell
   wsl -e bash -c "if [ ! -d /tmp/kdef_official ]; then cp -r /mnt/f/Hoc-May/data/official-kdef-dataset-by-ksatrajit0 /tmp/kdef_official; fi && ls -la /tmp/kdef_official"
   ```
   *(Note: Also supports in-memory caching of the 4,888 clean images in RAM since `4888 x 128 x 128` uint8 grayscale is only **80.1 MB** in memory! Pre-loading decoded/cropped images into a RAM tensor/array eliminates disk I/O completely and reduces a 60-epoch run to **~12–20 seconds** on RTX 4060.)*

2. **Running Training / Ablation Scripts on RTX 4060 GPU from PowerShell:**
   ```powershell
   wsl -e /home/dhp01/kdef_env/bin/python /mnt/f/Hoc-May/scripts/<your_script>.py
   ```
   *(Fallback virtualenv with identical `torch 2.9.1+cu128`, `torchvision`, `scipy`, `matplotlib`, `tqdm`, `Pillow`, `numpy` if needed: `/home/dhp01/Microduck/.venv/bin/python`)*

---

## 2. Gap 1: Core `src/` Pipeline vs. Notebook (`notebooks/CNN_KDEF_FER.ipynb`) Discrepancy

The core Python package (`src/`) is completely out of sync with the research findings in `reports/data_analysis.md`, `reports/EXPERIMENT_REPORT.md`, and `notebooks/CNN_KDEF_FER.ipynb`. Running `src/models/train.py` as-is trains an over-parameterized 51.5M-parameter network on a leaked, mislabeled dataset.

| Component | Current `src/` Implementation | Notebook (`CNN_KDEF_FER.ipynb`) | Root Cause of Discrepancy & Required Fix in `src/` |
| :--- | :--- | :--- | :--- |
| **Dataset Target** | `src/models/train.py:111` & `evaluate.py:61` hardcode `data/balanced-kdef-dataset-by-dolly-prajapati-182` (62,923 images). | `data/official-kdef-dataset-by-ksatrajit0` (4,898 raw studio images; **4,888 clean** after removing 10 corrupt frames; 70 subjects split 48/8/14). | `balanced` is actually 75x75 FER-2013 + 26,772 offline augmentations with **67 MD5 cross-split leaks** (24 train-test). `src/` must default to `official-kdef-dataset-by-ksatrajit0`. |
| **Folder Case & File Globbing** | `src/data/dataset.py:17,33` uses capitalized `['Angry', ...]` and `glob.glob("*.jpg")`. | Lowercase class folders `['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']` and case-insensitive `.jpg/.JPG/.jpeg/.png`. | On Linux/WSL2 ext4, `Angry` vs `angry` and `*.jpg` vs `.JPG` (4,898 files are uppercase `.JPG`) returns **0 files**! `src/data/dataset.py` must use lowercase classes and case-insensitive extension matching. |
| **Corrupt File Filtering** | `src/data/dataset.py` has **no corrupt file filtering**. | Filters MD5 `320b34032e2133fdfb80a58c953d5c94` (10 corrupt pitch-black frames: `max_pixel = 1`, `mean = 0.0005`). | `src/data/dataset.py` must automatically exclude the 10 corrupt dark frames (9 in `train`, 1 in `val`) by MD5 or filename blacklist. |
| **Model Architecture** | `src/models/model.py` (`BaselineCNN`): 3 Conv layers **without BatchNorm**, RGB `3x224x224`, `Flatten()` $\to$ `Linear(128*28*28, 512)` = **51,480,007 params** (99.8% in `fc1`). | `KDEFNet`: 4 stages of `2x(Conv3x3-BN-ReLU) + MaxPool2x2`, grayscale `1x128x128`, `AdaptiveAvgPool2d(1)` $\to$ `Linear(256, 256)` $\to$ `Linear(256, 7)` = **1,241,191 params**. | `BaselineCNN` lacks BatchNorm (causing severe internal covariate shift) and has 100,352 features flattened into `fc1` (guaranteed memorization on 3,349 images). |
| **Face Alignment & Preprocessing** | `src/preprocessing/alignment.py` (`FaceAligner` using `haarcascade_frontalface_default.xml` + `cv2.equalizeHist`) is **unused dead code**. `dataset.py` uses `Resize(224,224)` RGB + ImageNet stats + `ColorJitter`. | Grayscale `1x128x128`, KDEF dataset stats (`mean=[0.3370]`, `std=[0.1871]`). | Frontal Haar Cascade fails (`returns None`) on KDEF full-profile (`FL`/`FR`, $\pm 45^\circ$) and half-profile (`HL`/`HR`) poses (40–80% of data). Instead, deterministic KDEF studio face-region cropping + CLAHE (`clipLimit=2.0, tileGridSize=(8,8)`) works across **100% of all 5 pose angles**. |
| **Training & Evaluation Loop** | `src/models/train.py` uses `Adam(lr=1e-4)` (no weight decay, no label smoothing) and saves best model by **`val_loss`**. `evaluate.py` only computes accuracy & classification report. | `AdamW(lr=1e-3, weight_decay=1e-4)`, `CrossEntropyLoss(label_smoothing=0.1)`, selects checkpoint by **`val_macro_f1`**, logs per-angle & per-gender breakdowns. | `val_loss` diverges from `val_macro_f1` under label smoothing / class difficulty. `src/models/train.py` and `evaluate.py` must track **Macro-F1, Per-Class Recall Spread, Per-Angle Accuracy (`FL, FR, HL, HR, S`), and TTA**. |

### Exact List of 10 Corrupt Files to Exclude (`MD5 = 320b34032e2133fdfb80a58c953d5c94`)
1. `train/disgust/AF20DIHL.JPG`
2. `train/disgust/AM25DIFL.JPG`
3. `train/disgust/AM34DIFR.JPG`
4. `train/disgust/BM21DIFL.JPG`
5. `train/disgust/BM22DIHL.JPG`
6. `train/fear/AF10AFFR.JPG`
7. `train/neutral/AF11NEHL.JPG`
8. `train/neutral/BF13NEHR.JPG`
9. `train/surprise/AF01SUFR.JPG`
10. `val/disgust/BM24DIFL.JPG`

---

## 3. Gap 2: Pre vs. Post Training Activation Shift & Convergence Instability

### 3.1 Pre vs. Post Training Activation Shift (`dead_frac` Jump in Deep Layers)
From `reports/EXPERIMENT_REPORT.md` (§3.4) and `reports/figures/nb_pre_layer_stats.json` (probed on `AF01HAHR.JPG`):

| Layer (`KDEFNet`) | Channels | Param Count (% of Total) | Pre-Train `mean` | Post-Train `mean` | Pre-Train `dead_frac` | Post-Train `dead_frac` | Shift Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `features.0.2` (Block 1, ReLU 1) | 32 | 320 (0.03%) | 0.351 | 0.258 | 0.491 | 0.484 | Healthy (-0.7% dead) |
| `features.0.5` (Block 1, ReLU 2) | 32 | 9,280 (0.75%) | 0.450 | 0.369 | 0.496 | 0.465 | Healthy (-3.1% dead) |
| `features.1.2` (Block 2, ReLU 1) | 64 | 18,560 (1.50%) | 0.495 | 0.354 | 0.483 | 0.506 | Slight increase (+2.3%) |
| `features.1.5` (Block 2, ReLU 2) | 64 | 36,992 (2.98%) | 0.506 | 0.453 | 0.517 | 0.548 | Moderate increase (+3.1%) |
| `features.2.2` (Block 3, ReLU 1) | 128 | 73,984 (5.96%) | 0.525 | 0.391 | 0.505 | 0.503 | Stable (-0.2%) |
| `features.2.5` (Block 3, ReLU 2) | 128 | 147,712 (11.90%) | 0.553 | 0.425 | 0.478 | 0.503 | Stable (+2.5%) |
| **`features.3.2` (Block 4, ReLU 1)** | **256** | **295,424 (23.80%)** | **0.548** | **0.326 (-40.5%)** | **0.512** | **0.602 (+9.0%)** | **Degrading (60.2% dead)** |
| **`features.3.5` (Block 4, ReLU 2)** | **256** | **590,848 (47.60%)** | **0.511** | **0.259 (-49.3%)** | **0.525** | **0.651 (+12.6%)** | **Critical (>65% dead threshold)** |

#### Root Causes of the Deep-Layer Activation Collapse:
1. **Plain VGG-Style Stacking Without Residual Shortcuts:** `Block 4` (`features.3`) contains **71.4% (886,272 / 1,241,191)** of the entire network's parameters. Without identity skip connections (`x + F(x)`), gradients must traverse 8 sequential `Conv-BN-ReLU` layers.
2. **Hard Zero-Clipping `ReLU` + `BatchNorm` Shift Drift Under Weight Decay:** In `features.3.5`, `BatchNorm2d(256)` precedes `ReLU(inplace=True)`, which is immediately followed by `AdaptiveAvgPool2d(1)` and `Dropout(0.4)`. Because `AdaptiveAvgPool2d(1)` averages over the $8 \times 8$ spatial grid of non-negative `ReLU` outputs, the only way for the network to suppress noisy background/hair activations in channel $c$ is to push the `BatchNorm` bias $\beta_c$ negative so that `ReLU` zeroes out 65.1% of the spatial locations (`dead_frac = 0.651`), cutting mean activation nearly in half (`0.511 -> 0.259`).
3. **Architectural Remedy (`@arch-opt`):**
   - Replace plain stacked `ConvBlock` with **Residual Conv Blocks** (`1x1` projection shortcut when channel dimensions or strides change, identity shortcut otherwise).
   - Replace hard `ReLU` with smooth **`SiLU` (Swish)** or **`GELU`** (or `LeakyReLU(0.1)`), which allows small negative pre-activations to pass non-zero gradients and eliminates the hard dead-neuron cliff.
   - Add **Squeeze-and-Excitation (`SE`) / Channel-Spatial Attention** at Stages 3 and 4 so channel suppression happens via learned gating rather than killing `ReLU` units.

---

### 3.2 Early-vs-Late Convergence Instability (`0.1099–0.4163` @ 15 Ep vs. `0.8798` @ 91 Ep) & Fine-Tuning Degradation (`0.8798 -> 0.8736`)
From `reports/figures/nb_tuning_results.csv` and `notebooks/CNN_KDEF_FER.ipynb` training history:

| Run / Stage | Epochs Run | LR Schedule / Init | Best Val Macro-F1 | Instability Observation |
| :--- | :---: | :--- | :---: | :--- |
| **Grid Search (8 configs)** | 5 to 15 (`PATIENCE=4`) | Step `ReduceLROnPlateau(factor=0.5, patience=3)` | `0.1099` – `0.4163` | 6 of 8 configs early-stopped at Ep 5–9 with `val_f1 <= 0.2335`. Even the winning config (`lr=1e-3, wd=1e-4, drop=0.4`) only reached `0.4163` at Ep 15 and suffered a `val_loss = 2.9582` spike (`val_f1 = 0.1453`) at Ep 14. |
| **Baseline Run (`baseline.pt`)** | 16 (stopped early) | `lr=1e-3`, `patience=10` | `0.2143` (Ep 6) | Collapsed to `val_f1 = 0.0358` (`val_loss = 3.8977`) at Ep 11 and never recovered before early stopping at Ep 16. |
| **Full Training (`final_model.pt`)** | 91 (best @ Ep 81) | `lr=1e-3`, `ReduceLROnPlateau(patience=5)` | **`0.8798`** (Ep 81) | Hovering near random guess (`val_f1 = 0.08–0.16`) for **Epochs 1–12**. Massive epoch-to-epoch oscillations (`0.8136 -> 0.7447` at Ep 49–51; `0.8642 -> 0.7321` at Ep 60–61) until LR decayed 6 times to `1.9e-5` (Ep 66–81). |
| **Cell 8.4 Fine-Tuning (`+20 ep`)** | +13 (early stopped) | Fresh `AdamW(lr=7.5e-5)` + `CosineAnnealingLR` | `0.8736` (**-0.62% drop**) | Ep 1 of fine-tuning dropped immediately from `0.8798` to `0.8617` and never recovered `0.8798`. |

#### Root Causes of Convergence Instability & Fine-Tuning Degradation:
1. **Cold-Start AdamW Variance Explosion (Epochs 1–15):** Without **Linear LR Warmup**, starting `AdamW` at `lr = 1e-3` on an unnormalized 8-layer CNN causes huge initial weight updates while Adam's second moment $v_t$ is warming up and `BatchNorm` running statistics (`running_mean`, `running_var`) are still uncalibrated. During Epochs 1–12, training loss barely moves (`1.98 -> 1.72`) while validation predictions collapse into predicting a single dominant class (`val_f1 ~ 0.06–0.16`).
2. **Reactive Step-wise `ReduceLROnPlateau` Oscillations:** `ReduceLROnPlateau` keeps `lr = 1e-3` / `5e-4` constant while the model bounces off steep loss walls (causing 10–15% `val_f1` swings between consecutive epochs), only halving LR after 5 bad epochs.
3. **Why Cell 8.4 Fine-Tuning Degraded (`0.8798 -> 0.8736`):**
   - `final_model.pt` achieved `0.8798` at **Epoch 81** when `lr` had already decayed to **`1.9e-5`** (and finished Epoch 91 at `9.4e-6`).
   - Cell 8.4 instantiated a **brand-new `AdamW` optimizer** (wiping out the accumulated first and second moments $m_t, v_t$!) and set the starting learning rate to **`7.5e-5`** — **4x higher** than the `1.9e-5` at which `final_model.pt` converged! That sudden 4x LR jump with zeroed second moments knocked the weights out of their narrow minimum on Episode 1 (`0.8798 -> 0.8617`).
4. **Training Remedy (`@train-opt`):**
   - Use **Single-Stage Cosine Annealing with Linear Warmup** (`5 epochs` warmup from `1e-5` to peak `lr = 1e-3` or `1.5e-3`, followed by smooth cosine decay to `1e-6` over 60–80 epochs) — never re-initialize `AdamW` mid-training without warmup or state preservation.
   - Maintain an **Exponential Moving Average (`EMA`, decay `0.995`–`0.998`)** or **Stochastic Weight Averaging (`SWA`)** over the last 25% of epochs to eliminate single-epoch validation oscillations.

---

## 4. Gap 3: Train vs. Val vs. Test Generalization Gap (`0.955 -> 0.8798 -> 0.8275`)

### 4.1 Quantitative Breakdown of the Generalization Gap
| Split | Subjects | Clean Images | Gender Ratio (F / M) | Macro-F1 (`KDEFNet`) | Drop from Previous Split |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 48 | 3,349 | 45.8% F / 54.2% M | **`0.9550`** *(Ep 81: `0.9496`, Ep 86: `0.9602`)* | — |
| **Validation** | **8** | **559** | 50.0% F / 50.0% M | **`0.8798`** | **-7.52% (Train $\to$ Val)** |
| **Test** | **14** | **980** | **64.3% F / 35.7% M** (9 F, 5 M) | **`0.8275`** (`Acc = 82.76%`) | **-5.23% (Val $\to$ Test)** / **-12.75% (Train $\to$ Test)** |

### 4.2 Root Causes of the -7.52% (Train-Val) and -5.23% (Val-Test) Gaps
1. **Identity & Background Memorization on 48 Training Subjects:**
   - Every subject in `train` appears **70 times** (7 emotions $\times$ 5 angles $\times$ 2 sessions). Because the raw $562 \times 762$ images include hairstyles, neck/shoulders, skin tone, and studio lighting vignette, a 1.24M-parameter network trained with weak weight decay (`1e-4`) and only basic pixel augmentations (`RandomHorizontalFlip`, `RandomRotation(10)`, `RandomErasing(0.2)`) memorizes **subject-specific baseline appearance** (how Subject `AM01`'s neutral brow looks vs. their angry brow). When tested on unseen subjects (`val` or `test`), that subject-identity baseline is unavailable.
2. **Small Validation Cohort (Only 8 Subjects = 559 Images) & Spike Selection Optimism:**
   - Because `val` has only 8 individuals, selecting the single highest epoch (`Ep 81: 0.8798`, whereas `Ep 80` was `0.8686` and `Ep 82` was `0.8663`) overfits by ~`1.2%–1.5%` to noise on those 8 subjects.
3. **Demographic Shift in Test Split (`64.3% Female` vs `45.8% Female` in Train):**
   - In `reports/test_metrics.json`, Test Accuracy on Males is **`85.14%`** (`298/350`), whereas Test Accuracy on Females is **`81.43%`** (`513/630`, a **-3.71% gender gap**). Because `test` is 64.3% Female (9 of 14 subjects), larger female hairstyles covering the forehead/temples in uncropped $562 \times 762$ studio frames disproportionately pull down the aggregate test score.
4. **Generalization Remedy (`@arch-opt` & `@train-opt`):**
   - **Tight Face-Region Cropping + CLAHE:** Strips away subject-specific hair/shoulders/lighting so the CNN sees only the facial Action Units (brows, eyes, nose bridge, mouth).
   - **Mixup (`alpha=0.2`–`0.4`) / CutMix:** Forces linear interpolation between **different subjects** during training, directly breaking subject-identity memorization on the 48 training actors.
   - **Spatial Dropout (`Dropout2d(0.1–0.15)`) + Stronger Weight Decay (`1e-3` to `5e-3`) + EMA Checkpointing:** Prevents co-adaptation of local spatial filters and eliminates single-epoch validation spike bias.

---

## 5. Gap 4: Per-Class Recall Spread (`27.86%`: `happy 99.29%` vs. `angry/fear 71.43%`, `sad 77.14%`)

### 5.1 Per-Class Test Performance (`reports/test_metrics.json`, $N = 140$ per class)
| Class | Precision | Recall | F1-Score | True Count | Predicted Count | Prediction Bias / Error Pattern |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`happy`** | `0.9929` | **`0.9929`** (`139/140`) | `0.9929` | 140 | 140 | Near-perfect (1 error $\to$ `neutral`). Distinct exposed teeth / zygomatic smile (AU6+AU12). |
| **`disgust`** | `0.7885` | **`0.8786`** (`123/140`) | `0.8311` | 140 | **156 (+11.4% over-predicted)** | Absorbs false positives from `angry` (12) and `sad` (13) due to shared nose-wrinkle / brow-lowering cues. |
| **`neutral`** | `0.8356` | **`0.8714`** (`122/140`) | `0.8531` | 140 | 146 (+4.3%) | Confused with subtle `sad` (12 false negatives $\to$ `sad`, 11 false positives from `sad`). |
| **`surprise`** | `0.8686` | **`0.8500`** (`119/140`) | `0.8592` | 140 | 137 (-2.1%) | **17 samples misclassified as `fear`**; absorbs **18 `fear` samples**. |
| **`sad`** | `0.7248` | **`0.7714`** (`108/140`) | `0.7474` | 140 | **149 (+6.4% over-predicted)** | Lowest precision (`72.48%`). Confused with `disgust` (13), `neutral` (11), and `fear` (6). |
| **`angry`** | **`0.9009`** | **`0.7143`** (`100/140`) | `0.7968` | 140 | **111 (-20.7% under-predicted!)** | High precision (`90.1%`) but severe under-prediction: **13 leaked to `fear`**, **12 to `disgust`**, **7 to `sad`**, **7 to `neutral`**. |
| **`fear`** | `0.7092` | **`0.7143`** (`100/140`) | **`0.7117`** | 140 | 141 (+0.7%) | Lowest F1 (`71.17%`). **18 leaked to `surprise`**, **11 to `sad`**, **5 to `disgust`**, **4 to `angry`**. |
| **Spread / Avg** | `0.8315` | **`0.8276` (Spread: `27.86%`)** | **`0.8275`** | 980 | 980 | `max(Recall) - min(Recall) = 99.29% - 71.43% = 27.86%` |

### 5.2 Top Confusion Pairs in the Test Confusion Matrix (`169` Total Errors)
1. **`fear` $\leftrightarrow$ `surprise`: 35 confused samples** (`18 fear -> surprise` + `17 surprise -> fear` = **20.7% of all test errors!**)
2. **`angry` $\leftrightarrow$ `disgust`: 17 confused samples** (`12 angry -> disgust` + `5 disgust -> angry`)
3. **`sad` $\leftrightarrow$ `disgust`: 24 confused samples** (`13 sad -> disgust` + `11 disgust -> sad`)
4. **`sad` $\leftrightarrow$ `neutral`: 23 confused samples** (`11 sad -> neutral` + `12 neutral -> sad`)
5. **`angry` $\leftrightarrow$ `fear`: 17 confused samples** (`13 angry -> fear` + `4 fear -> angry`)
6. **`fear` $\leftrightarrow$ `sad`: 17 confused samples** (`11 fear -> sad` + `6 sad -> fear`)

### 5.3 Root Causes of the 27.86% Class Recall Spread
1. **Destruction of Upper- vs. Lower-Face Spatial Layout by `AdaptiveAvgPool2d(1)`:**
   - Psychologically (FACS Action Units), `fear` and `surprise` share widened eyes (AU5) and open jaws (AU25/26). They are distinguished almost entirely by **fine upper-face brow tension** (`fear` combines brow raiser AU1+2 with **brow lowerer/furrow AU4** and horizontal mouth stretch **AU20**, whereas `surprise` has smooth arched brows and a relaxed oval jaw drop).
   - Similarly, `angry` vs. `disgust` vs. `sad` all share brow lowering (AU4), differing in **regional combinations** (`angry` = tight lips AU23 + glare AU4/5; `disgust` = nose wrinkle AU9 + upper lip raise AU10; `sad` = inner brow raise AU1 + depressed lip corners AU15).
   - Because `KDEFNet` collapses the $8 \times 8$ feature map at the end of Stage 4 using a single **`AdaptiveAvgPool2d(1)`**, it throws away the spatial coordinate distinction between the **upper face (brows/eyes)** and **lower face (nose/mouth)**.
2. **Low Local Contrast in Dark Studio Portraits (`mean = 0.3370`):**
   - Subtle glabellar furrows (AU4 between eyebrows) and nasolabial folds that separate `fear` from `surprise` and `angry` from `neutral`/`sad` have low pixel intensity differences in raw KDEF lighting and get blurred away when $562 \times 762$ is downsampled directly to $128 \times 128$ without **CLAHE** (`Contrast Limited Adaptive Histogram Equalization`).
3. **Uniform Loss Weighting & `angry` Decision-Boundary Conservatism:**
   - `happy` reaches ~100% training accuracy within the first 20 epochs, while `fear`, `angry`, and `sad` remain hard throughout training. Standard `CrossEntropyLoss(label_smoothing=0.1)` treats all examples equally, leaving `angry` with a conservative decision boundary (predicted only **111 times** vs. 140 true instances; Precision `90.09%`, Recall `71.43%`).
4. **Remedies for Recall Spread (`@arch-opt` & `@train-opt`):**
   - **Multi-Region / Multi-Scale Pooling Head (`AdaptiveAvgPool2d(1) + AdaptiveMaxPool2d(1)` or Horizontal/Vertical Strip Pooling `AdaptiveAvgPool2d((2, 1))`):** Preserves upper-face (brow/eye) vs. lower-face (mouth/chin) feature separation!
   - **Focal Loss / Class-Difficulty Reweighting or Cosine Classifier Temperature Scaling:** Focuses gradient updates on hard examples (`fear`, `angry`, `sad`) and calibrates `angry`'s overly conservative logit threshold.

---

## 6. Gap 5: Pose Angle Asymmetry (`HL/HR 85.20%` vs. `FR 76.53%`, `-8.67%` Gap) & Uncropped 562x762 Studio Framing

### 6.1 Per-Angle Test Accuracy Breakdown (`reports/test_metrics.json`, $N = 196$ per angle)
| Pose Code | Angle Description | Correct / Total | Test Accuracy | Gap from Best Angle |
| :---: | :--- | :---: | :---: | :---: |
| **`HL`** | Half Left ($\approx -22.5^\circ$) | `167 / 196` | **`85.20%`** | `0.00%` (Best) |
| **`HR`** | Half Right ($\approx +22.5^\circ$) | `167 / 196` | **`85.20%`** | `0.00%` (Best) |
| **`S`** | Straight / Frontal ($0^\circ$) | `164 / 196` | **`83.67%`** | `-1.53%` |
| **`FL`** | Full Left Profile ($\approx -45^\circ$) | `163 / 196` | **`83.16%`** | `-2.04%` |
| **`FR`** | Full Right Profile ($\approx +45^\circ$) | `150 / 196` | **`76.53%`** | **`-8.67%` (Worst!)** |

### 6.2 Root Causes of the `FR` Profile Drop and Framing Distortion
1. **Severe Aspect-Ratio Squashing (`562x762` $\to$ `128x128`):**
   - 4,896 of 4,898 KDEF images are **`562 x 762`** portrait studio photographs (aspect ratio `1 : 1.356`; 2 images are `633 x 970`).
   - Applying `transforms.Resize((128, 128))` directly compresses the vertical axis by **26.2%** relative to the horizontal axis, artificially flattening open mouths (`surprise`/`fear`) and eyebrow arches.
   - Furthermore, in raw KDEF $562 \times 762$ portraits, the actual face occupies roughly `y in [0.18H, 0.82H]` and `x in [0.12W, 0.88W]`. Over **50% of the pixels** fed into `KDEFNet` are studio wall background, hair top, neck, and shoulders/t-shirt collar!
2. **Why `FR` (`76.53%`) Lags `FL` (`83.16%`) by `-6.63%` and `HL/HR` (`85.20%`) by `-8.67%`:**
   - **Single-View Inference Without Horizontal Flip Test-Time Augmentation (TTA):** Physically, a Full Right (`FR`) profile image flipped horizontally is geometrically identical to a Full Left (`FL`) profile image! During training, `RandomHorizontalFlip(p=0.5)` exposes the network to flipped images, but internal convolutional filters still develop slight directional biases on the final training epochs. At test time, evaluating only the unflipped `FR` image leaves `FR` vulnerable to asymmetric filter response. Averaging logits of **`[image, horizontal_flip(image)]` (Flip-TTA)** forces exact mathematical symmetry between left and right poses (`FL` $\leftrightarrow$ `FR` and `HL` $\leftrightarrow$ `HR`)!
   - **Off-Center Facial Mass in Full Profiles (`FL`/`FR`):** In $\pm 45^\circ$ profile views, the nose, mouth, and visible eye shift toward one edge of the frame while the back of the head/hair occupies the center-opposite side. Random `RandomErasing(p=0.2, scale=(0.02, 0.15))` in `KDEFNet` occasionally erases the *only* visible eye or mouth corner in `FL`/`FR` profiles without any semantic constraint.

---

## 7. Actionable Optimization Roadmap for Swarm Peers

### For `@arch-opt` (`scripts/exp_arch_preproc.py` $\to$ `reports/exp_arch_results.json`)
1. **Deterministic Studio Face-Region Crop + CLAHE Preprocessing:**
   - Crop out studio background/hair/neck/shoulders from the $562 \times 762$ portrait before resizing (e.g., vertical crop `[int(0.14*H) : int(0.86*H)]`, horizontal crop `[int(0.08*W) : int(0.92*W)]`, which also restores a near-square `~472 x 548` aspect ratio before `Resize((128, 128))`) so works reliably on **all 5 angles (`FL, FR, HL, HR, S`)** where Haar Cascades fail.
   - Apply **CLAHE** (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`) to enhance brow/glabellar/nasolabial micro-contrasts in dark frames.
2. **Residual + Smooth Activation + Attention + Multi-Region Pooling Backbone (`ResKDEFNet` / `SE-ResNet`):**
   - Replace plain `Conv-BN-ReLU` blocks with **Residual Blocks** (`Conv3x3-BN-SiLU-Conv3x3-BN + Shortcut`) to eliminate the `features.3.5` dead-neuron jump (`0.651` $\to$ `0.0` hard dead units with `SiLU`/`GELU`).
   - Add **Squeeze-and-Excitation (`SE`)** or **Coordinate Attention (`CA`)** to preserve vertical (upper-brow vs. lower-mouth) and horizontal (profile pose) awareness.
   - Replace single `AdaptiveAvgPool2d(1)` with **Concatenated Global Avg + Max Pooling** (or **Upper/Lower Face Strip Pooling `AdaptiveAvgPool2d((2,1))`**) so `fear` vs `surprise` and `angry` vs `disgust`/`sad` retain separate brow and mouth features.

### For `@train-opt` (`scripts/exp_train_reg.py` $\to$ `reports/exp_train_results.json`)
1. **Eliminate Early Convergence Instability & Fine-Tuning Degradation:**
   - Replace `ReduceLROnPlateau` + two-stage restart with a **Single-Stage Linear Warmup (5 epochs) + Cosine Annealing Schedule** (`lr: 1e-5 -> 1.5e-3 -> 1e-6`).
   - Add **Model EMA (Exponential Moving Average, decay `0.995`–`0.997`)** or **SWA** to smooth out epoch-to-epoch validation oscillations (`10–15%` swings in baseline).
2. **Bridge the Train-Val-Test Generalization Gap & Class Recall Spread:**
   - Apply **Mixup (`alpha=0.2`–`0.4`)** across training images to break 48-subject identity memorization.
   - Test **Focal Loss ($\gamma = 1.5$–$2.0$) + Label Smoothing (`0.05`–`0.1`)** or class-difficulty-weighted Cross-Entropy to boost `angry` (`71.43%`), `fear` (`71.43%`), and `sad` (`77.14%`) recall and compress the `27.86%` recall spread.
   - Evaluate **Test-Time Augmentation (TTA: Original + Horizontal Flip + Slight Multi-Crop/Scale)** on Val to verify profile (`FR`/`FL`) symmetry gains.

### For `@core-integrator` (`src/` Modernization & Final Verification)
1. **Unify `src/` with the Winning Architecture & Clean Dataset:**
   - Update `src/data/dataset.py` to point to `data/official-kdef-dataset-by-ksatrajit0`, fix lowercase class names + case-insensitive `.JPG` matching, and hard-exclude the **10 corrupt MD5 `320b34032e2133fdfb80a58c953d5c94` files**.
   - Replace `src/preprocessing/alignment.py` with the validated studio face-crop + CLAHE pipeline.
   - Replace `BaselineCNN` (51.48M params) in `src/models/model.py` with the winning architecture (`ResKDEFNet` / `KDEFNet` variant), and modernize `src/models/train.py` and `src/models/evaluate.py` to compute **Test Accuracy, Macro-F1, Per-Class Recall Spread, Per-Angle Accuracy, and Pre/Post Activation Health (`dead_frac`)**.
