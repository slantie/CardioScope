# Classifier experiments: results

Reproduce from the `app` folder:

    ../.venv/bin/python experiments/run_experiments.py           # all experiments, 3 seeds (~13.5 min on CPU)
    ../.venv/bin/python experiments/run_experiments.py --sanity  # label-permutation leakage check (~1 min)

Full output: `run_log.txt`, `sanity_log.txt`, per-seed numbers in `results_raw.csv`.
torch 2.15.0 (CPU) was installed into `../.venv` for the LSTM/CNN.

## Protocol
- **Rows:** exactly the rows of `decision.build_training_table` (195 heart, 195 lung). The script checks this with an `assert` on keys, labels and md5 groups.
- **Split:** `StratifiedGroupKFold(5, shuffle=True, random_state=seed)`, grouped by audio md5, so byte-identical copies are never in both train and test. Seed 42 is the split `decision.py` uses. The whole thing is repeated with seeds 7 and 2024.
- **Windows** (3 s, 50 % overlap, 9 per clip) carry their parent clip's index, and therefore its group and fold. A test recording's prediction is the mean of its window probabilities.
- **Augmentation** (gain ×0.5–1.5, circular shift ≤1 s, noise at 25–40 dB SNR) is applied only to training-fold clips. For the classical models, 2 augmented copies per clip are pre-computed, and a fold uses only the copies of its own training clips. Test data is never augmented.
- **Hyperparameters were fixed up front**, not tuned on the folds: LR C=1; RF 500 trees, sqrt features; HGB 200 iterations, learning rate 0.05, depth 3, L2 1.0; SVM-RBF C=10, gamma='scale'; LSTM/CNN as written in the script.
- **Early stopping** for the neural nets uses an inner grouped split (1/4 of the training fold). The nets are not refit on the full training fold.
- **Nested selection:** one experiment chooses among 5 classical models inside each training fold by inner grouped 4-fold CV. This gives an honest estimate of "pick the best model".
- **Metrics:** type accuracy and normal-vs-abnormal (N/A) balanced accuracy (derived from the type prediction), shown as mean ± std over the 5 folds.
- **Leakage check:** with labels shuffled between unique clips, the recommended pipeline falls to chance: heart 0.104 (chance 0.100), lung 0.185 (chance 0.167).
- **Near duplicates:** among the md5-unique clips, no pair has a normalised cross-correlation above 0.9. A handful of pairs are above 0.5 (6 heart, 6 lung).

## Unique clips per class (no class dropped)
| Heart | unique / rows | Lung | unique / rows |
|---|---|---|---|
| Mid Systolic Murmur | 14 / 21 | Pleural Rub | 18 / 34 |
| Early Systolic Murmur | 14 / 19 | Rhonchi | 17 / 31 |
| Normal | 12 / 22 | Normal | 16 / 40 |
| S3 | 12 / 20 | Coarse Crackles | 16 / 28 |
| Atrial Fibrillation | 11 / 19 | Wheezing | 14 / 35 |
| Late Diastolic Murmur | 10 / 19 | Fine Crackles | 13 / 27 |
| S4 | 10 / 18 | | |
| AV Block | 9 / 16 | | |
| Tachycardia | 8 / 19 | | |
| Late Systolic Murmur | 7 / 22 | | |
| **total** | **107 / 195** | **total** | **94 / 195** |

## Results
The columns marked "seed 42" are mean ± std over the 5 grouped folds. "3 seeds" is the mean type accuracy over seeds 42, 7 and 2024. Times are for one seed of 5-fold CV, both kinds together, plus about 20 s of feature extraction.

| Experiment | Heart acc (seed 42) | Heart N/A bal | Lung acc (seed 42) | Lung N/A bal | Heart 3 seeds | Lung 3 seeds | Time |
|---|---|---|---|---|---|---|---|
| E0 baseline LR (decision.py) | 0.436 ± 0.070 | 0.562 ± 0.079 | 0.667 ± 0.073 | 0.867 ± 0.140 | 0.441 | 0.640 | <1 s |
| E1 LR base+location | 0.281 ± 0.079 | 0.456 ± 0.098 | 0.431 ± 0.102 | 0.708 ± 0.155 | 0.300 | 0.433 | <1 s |
| **E1 LR base+MFCC** | **0.672 ± 0.050** | **0.581 ± 0.109** | **0.959 ± 0.048** | **0.959 ± 0.052** | **0.643** | **0.973** | <1 s |
| E1 RF base+MFCC | 0.574 ± 0.044 | 0.495 ± 0.049 | 0.867 ± 0.052 | 0.840 ± 0.145 | 0.595 | 0.884 | 3 s |
| E1 HGB base+MFCC | 0.532 ± 0.147 | 0.632 ± 0.143 | 0.795 ± 0.081 | 0.806 ± 0.106 | 0.564 | 0.849 | 18 s |
| E1 SVM base+MFCC | 0.589 ± 0.084 | 0.504 ± 0.059 | 0.923 ± 0.092 | 0.949 ± 0.056 | 0.610 | 0.923 | <1 s |
| E1 RF base+MFCC+location | 0.553 ± 0.071 | 0.459 ± 0.025 | 0.877 ± 0.077 | 0.881 ± 0.144 | 0.593 | 0.879 | 3 s |
| E1 SVM base+MFCC+location | 0.609 ± 0.063 | 0.534 ± 0.065 | 0.908 ± 0.093 | 0.912 ± 0.079 | 0.573 | 0.897 | <1 s |
| E1 nested model selection | 0.630 ± 0.065 | 0.521 ± 0.061 | 0.959 ± 0.048 | 0.959 ± 0.052 | 0.610 | 0.973 | 55 s |
| E2 windows LR base+MFCC | 0.646 ± 0.069 | 0.631 ± 0.142 | 0.979 ± 0.025 | 0.975 ± 0.050 | 0.641 | 0.983 | <1 s |
| E2 windows RF base+MFCC | 0.585 ± 0.066 | 0.538 ± 0.086 | 0.913 ± 0.055 | 0.956 ± 0.073 | 0.617 | 0.933 | 14 s |
| E2 windows SVM base+MFCC | 0.682 ± 0.050 | 0.691 ± 0.137 | 0.979 ± 0.025 | 0.975 ± 0.050 | 0.649 | 0.971 | 1 s |
| E4 augment LR base+MFCC | 0.652 ± 0.056 | 0.587 ± 0.101 | 0.969 ± 0.041 | 0.969 ± 0.048 | 0.624 | 0.973 | <1 s |
| E4 augment RF base+MFCC | 0.590 ± 0.043 | 0.641 ± 0.154 | 0.836 ± 0.058 | 0.834 ± 0.132 | 0.598 | 0.870 | 3 s |
| E4 augment SVM base+MFCC | 0.630 ± 0.074 | 0.628 ± 0.052 | 0.938 ± 0.070 | 0.946 ± 0.068 | 0.593 | 0.943 | <1 s |
| E3 LSTM (BiLSTM-32, log-mel, 3 s windows, aug) | 0.543 ± 0.086 | 0.548 ± 0.132 | 0.856 ± 0.093 | 0.796 ± 0.121 | 0.526 | 0.848 | ~110 s |
| E3 CNN (3-block 2D, log-mel, aug), seed 42 only | 0.388 ± 0.186 | 0.483 ± 0.042 | 0.226 ± 0.044 | 0.600 ± 0.108 | (0.388) | (0.226) | ~310 s |
| E5 dedicated binary N/A, LR base | — | 0.586 ± 0.210 | — | — | N/A bal 0.529 | — | <1 s |
| E5 dedicated binary N/A, LR base+MFCC | — | 0.561 ± 0.187 | — | — | N/A bal 0.499 | — | <1 s |
| E5 dedicated binary N/A, RF base+MFCC | — | 0.488 ± 0.017 | — | — | N/A bal 0.485 | — | 3 s |

Chance level is 0.10 for heart type and 0.17 for lung type. Seed-to-seed std of type accuracy is 0.01–0.04 for most rows (see `results_raw.csv`). The baseline reproduces `decision.py` exactly at seed 42 (0.436 / 0.562 / 0.667 / 0.867).

## What worked
- **Adding MFCC statistics** (mean, std and delta-std of 20 MFCCs) to the existing 20 features is the only change that clearly helps.
  - Heart: 0.44 → 0.64–0.67.
  - Lung: 0.64–0.67 → 0.96–0.97.
  - The plain linear model (LR) beat RF, HGB and SVM on these features. In the nested selection, LR was picked in 26 of the 30 outer folds.
- **Windowing and augmentation** do not add a clear gain on top of that. All differences are within one fold std (heart 0.62–0.65 vs 0.64; lung 0.97–0.98 vs 0.97).

## What did not work
- **Location one-hot** made results much worse (heart 0.28, lung 0.43). The location metadata is unreliable: 15 heart and 10 lung md5 groups are byte-identical audio labelled with *different* locations. With ~100 clips, 12 extra noisy columns mostly add overfitting.
- **LSTM** is below the simple LR: heart 0.53, lung 0.85 over 3 seeds. With ~80 unique training clips per fold, it has too little data.
- **CNN** failed. On lung it scored near chance (0.23), which means it did not train properly. It was not debugged within the time budget, so this is a "not made to work" result, not evidence that CNNs cannot work here.
- **Normal vs abnormal (heart)** did not improve. The N/A balanced accuracy is 0.50–0.69, depending on model and seed, with fold std of 0.1–0.2. A dedicated binary model was no better.
  - With the recommended model, Normal heart recall is only 0.27.
  - There are only 12 unique Normal heart clips.

## Recommendation
**Use E1: the existing ~20 features plus 60 MFCC statistics → StandardScaler → LogisticRegression(C=1), one feature row per 15 s clip.** It is in `best_model.py` (`train(kind)`, `predict_proba(signal, sr, kind)`).

Honest grouped 5-fold CV numbers (seed 42, mean ± fold std; mean over 3 seeds in brackets):
- **Heart type accuracy: 0.672 ± 0.050 (3 seeds: 0.643).**
  - Normal-vs-abnormal balanced accuracy: 0.581 ± 0.109.
- **Lung type accuracy: 0.959 ± 0.048 (3 seeds: 0.973).**
  - Normal-vs-abnormal balanced accuracy: 0.959 ± 0.052.
- Conservative figure, since this model was picked as the best of ~17 configurations evaluated on the same folds: nested selection gives **heart 0.610, lung 0.973** (3-seed means).

**Is it worth putting into the app?**
- **Yes.** It is the same model family as now, so the "coef × value" explanation in `decision.py` still works. The new features can be presented as "MFCCs = a compact description of the spectrum shape (timbre)". This fits in a 30-minute presentation.
- The windowed SVM (heart 0.682 at seed 42, 0.649 over 3 seeds) is not clearly better and is harder to explain.
- The LSTM is worse and harder to explain.

**The 70–80 % target**
- **Heart: not reached** (about 0.64–0.67).
  - There are only 7–14 unique clips per class, and each test fold holds about 2 clips per class. One clip therefore moves a class's fold accuracy by about 50 %.
  - The rhythm classes (AV Block 0.31 recall, Atrial Fibrillation 0.32) and Normal (0.27) are confused with each other.
  - The murmur classes are recognised well (0.84–0.95).
- **Lung: exceeded** (about 0.96–0.97). This needs a caveat when presented.
  - The data come from one manikin that replays a small set of recorded sounds at different chest positions.
  - A test clip is therefore a new recording of a sound source the model has already heard.
  - The number measures "recognises the manikin's lung sound programs". It does not measure accuracy on real patients.

## Caveats a reviewer may raise
1. **Selection bias.** About 17 configurations were compared on the same folds. Quote the nested-selection number (heart 0.61) as the lower, honest bound.
2. **Fold std is large.** The heart std is ±0.05–0.15, because each fold holds about 21 unique clips.
3. **Duplicate weighting.** Accuracy is computed over all 195 rows, as `decision.py` does, so duplicated clips count more than once. Counting each unique clip once gives heart 0.673 and lung 0.947 (seed 42).
4. **Neural nets are not refit.** They are trained on 3/4 of each training fold, because the remaining 1/4 is used for early stopping. This slightly handicaps them.
5. **Training-set predictions are in-sample.** `best_model.predict_proba` uses a model fit on all rows. For training recordings the app should keep using out-of-fold models, as `decision.py` already does.

> **Note (after these experiments):** the MFCC features were moved into
> `features.mfcc_features` and are now part of `decision.decision_features`, so the app
> uses LR + MFCC. If you re-run this script, the "baseline ~20 features" setup will now
> also include the MFCCs. The numbers above were recorded before that change.
