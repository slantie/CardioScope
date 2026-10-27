# CardioScope (Simple Version)

M.Tech Sem 1 – EDA & Python Programming project.

A small desktop app to explore heart and lung sounds from the
**HLS-CMDS** dataset (Heart and Lung Sounds recorded from a clinical manikin
with a digital stethoscope).

> This is for learning / acoustic analysis only, not for medical diagnosis.
> The recordings come from a manikin, not real patients.

## Dataset
- 50 heart sounds (HS), 50 lung sounds (LS), 145 mixed heart+lung sounds (Mix)
- All files: mono, 16-bit, 4000 Hz, 15 seconds
- 10 heart sound types, 6 lung sound types
- Note: the paper says 22,050 Hz but the actual wav files are 4000 Hz
- Note: some lung IDs in `LS.csv` use old codes (C, G) that don't match the file
  names, so `Dataset` builds the file name from the type + location instead

The unzipped dataset sits next to this `app/` folder:

```
dataset/
  HS.csv  LS.csv  Mix.csv     labels
  HS/     50 heart wav files
  LS/     50 lung wav files
  Mix/    435 wav files: 145 mixtures (M) + their heart-only (H) and lung-only (L) files
```

## Files
Every file has one job, and most hold one small class.

| File | Class | Its one job | Lines |
|---|---|---|---|
| `dataset.py` | `Dataset` | Reads the label CSVs and the wav files in `dataset/` | 78 |
| `quality.py` | `QualityCheck` | Checks that every wav file can be read, at one sample rate and one length | 46 |
| `recordings.py` | `NewRecordings` | Imports a wav file, makes it look like the dataset, saves microphone recordings | 57 |
| `features.py` | (functions) | Signal maths: filters, FFT, features, MFCC. No state, so no class | 140 |
| `model.py` | `DecisionModel` | Trains and runs the logistic regression for heart or lung | 100 |
| `advisor.py` | `Advisor` | Picks the right model and decides how sure we are (50% gate) | 64 |
| `report.py` | `Report` | Writes the results as the heading and text of the suggestion panel | 84 |
| `filters.py` | `RecordingFilter` | Which recordings match the filters and the search text | 58 |
| `plots.py` | `Plotter` | Draws the charts | 115 |
| `player.py` | `AudioPlayer` | Plays and records sound | 58 |
| `windows.py` | `CompareWindow`, `MatchedWindow`, `QualityWindow` | The extra windows | 121 |
| `app.py` | `CardioScopeApp`, `SuggestionPanel` | Builds the main window and connects the parts | 377 |
| `test_cardioscope.py` | | 14 simple tests (`python -m pytest`) | 129 |
| `make_synthetic.py` | | Makes 2 synthetic test recordings in `samples/` | 59 |
| **Total** | | | **1531** |

## Design and SOLID principles

How the app runs:

1. `main()` in `app.py` creates the parts once: `Dataset`, `Advisor`, `NewRecordings`, `AudioPlayer`,
   and passes them to `CardioScopeApp`
2. The `Advisor` trains its two `DecisionModel`s in a background thread
3. The user picks a recording -> `Dataset.read()`
4. Features and plots -> `features.extract_features()` and `Plotter`
5. The suggestion panel: `Advisor.decide()` -> `Report(results).heading()` and `.text()`

We used three SOLID principles where they fit naturally:

| Principle | Where | Why it helps |
|---|---|---|
| **S – Single Responsibility** | One small class per job (table above). Data, maths, model, decision, wording, filters, plots, sound and windows are all separate | A change in one part never touches the others, and each part can be tested alone |
| **O – Open/Closed** | `FILTERS` in `filters.py`. Each filter is one line: label, column, match function | Adding a new filter = adding one line. The dropdowns, the list and the chart titles work for it without any change |
| **D – Dependency Inversion** | `CardioScopeApp` gets `Dataset`, `Advisor`, `NewRecordings` and `AudioPlayer` from `main()` instead of creating them. `compute_all_features(df, read)` gets the file reader as a parameter | Each part can be replaced or tested on its own. The test `test_features_work_with_any_reader` passes a fake reader that makes sine waves |

We did not force **L** (Liskov) and **I** (Interface Segregation): the project has no
class hierarchy or large interfaces, so adding them would only make the code harder to read.

## How to run
```
pip install -r requirements.txt
python app.py          # the app
python advisor.py      # prints the evaluation of both models
python -m pytest       # runs the tests
python make_synthetic.py  # (re)makes the 2 test recordings in samples/
```

## What works
- **Dataset filters**: category, sound type, auscultation location, gender and text search,
  plus a reset button. Sound type and location options change with the category
  (heart and lung recordings use different locations). For mixed recordings the sound
  type filter matches either the heart or the lung part (e.g. Mixed + Wheezing = 28)
- **EDA charts follow the filters**: class distribution shows counts per sound type and
  per location for the filtered recordings; feature comparison compares categories, or
  sound types when one category is selected
- Waveform, FFT and spectrogram of any recording
- Bandpass filtering to roughly split the heart band (20–200 Hz) and lung band (200–1000 Hz)
- Play / stop audio, with error messages if a file or the audio device fails
- Signal features shown for every recording: RMS, peak, ZCR, dominant frequency,
  spectral centroid, low/high energy ratio
- **Matched H-L-M**: for a mixed recording, shows it with its matched heart-only and
  lung-only recordings, plus waveform correlation and spectrum similarity
- **Data Quality Check**: checks all 535 files and the csv files and lists warnings
  with a suggested action for each
- **Export Table (CSV)**: saves the filtered recordings with their features
  (plots can be saved with the save icon on the plot toolbar)
- Dataset EDA: class distribution and feature box plots by category
- **Suggestion panel** (right side): updates automatically when a recording is selected
  and shows what the sound most likely is, the top 3 matches and why (see below)
- **Compare Two Recordings**: pick recording A and B, see their waveform / FFT /
  spectrogram one above the other plus a small feature table
- **Check a New Recording**: import any wav file or record 15 s from the microphone. Each recording is saved as a wav file in `recordings/` (e.g. `heart_20261116_101500.wav`).
  It is converted to look like the dataset (mono, 4000 Hz, same loudness) and then gets
  the same plots, features and suggestion. You choose if it is a heart, lung or mixed sound.
- **Synthetic test recordings** (`samples/`, made by `make_synthetic.py`): a normal heart
  (S1/S2 at ~72 bpm with small beat jitter) and a lung sound with a wheeze while breathing out

## App layout
| Left | Middle | Right |
|---|---|---|
| Filters (category, sound type, location, gender, search), recording list, dataset EDA buttons, compare, data quality, export | Recording details, plot buttons (incl. Matched H-L-M), play/stop, plot area, signal features | Suggestion panel |

The decision models train in the background when the app starts (about 3 seconds),
so the window opens straight away.

## Observations so far
- Heart sounds have most energy below 200 Hz (median low-energy ratio ≈ 0.93)
- Lung sounds have a higher spectral centroid (median ≈ 158 Hz vs ≈ 94 Hz for heart)
- Mixed recordings sit in between, as expected
- Lung sounds show a lot more spread, especially wheezing (high dominant frequency)
- **Only 344 of the 535 files are unique audio.** 114 groups are exact copies, e.g.
  `HS/F_N_RC.wav` is the same as `Mix/H0096.wav`. Many matched H/L files repeat the
  standalone recordings.
- **Mixtures are not the sum of their H and L files.** Waveform correlation is about 0.01
  (median), but spectrum similarity is about 0.98. So the mixture was recorded
  separately: same sounds, not the same samples.
- 14 lung IDs in `LS.csv` use old codes, and 36 files touch full scale (clipped)
- Among heart sounds, late diastolic and late systolic murmurs (LDM, LSM) have a clearly
  higher spectral centroid and less energy below 200 Hz than the other types

## Decision layer

Our professor suggested that besides the EDA insights, the end user should also get
some decision or suggestion. So we added a small, explainable decision layer.

### What other papers did with this dataset
| Paper | What they did |
|---|---|
| Torabi, Shirani, Reilly – *HLS-CMDS descriptor*, IEEE Data Descriptions 2025 | Introduced the dataset. Suggest PCA/ICA/SVM/CNN, classification and separation. Warn that manikin data is clean and not clinical, and recommend keeping a human in the loop |
| Torabi et al. – NMF using heart/lung periodicity (arXiv 2305.01889) | Blind source separation of heart and lung sounds |
| Torabi et al. – LLM + NMF (arXiv 2502.05757) | LLM guides the NMF separation and gives disease hints |
| Torabi – PhD thesis (arXiv 2602.09210) | Wavelet VAE separation, NMF + k-means clustering, quantum CNN for S3 vs murmur (93%) |

Most of the work is on **separation** (SDR/SIR) or deep models. We did not find a
simple, explainable model that gives an end user a suggestion **and admits when it
is not sure**. That is what we tried here.

### Our approach
1. ~20 interpretable features (spectral shape, band energies, entropy, rhythm/periodicity
   from the envelope autocorrelation, crackle-like transients) **+ 60 MFCC statistics**
   (mean, variation and change of 20 MFCCs, the standard "sound texture" features in
   heart/lung sound papers)
2. One small logistic regression for heart sounds, one for lung sounds
3. **Confidence gate**: if the top probability is below 0.5 → "UNCERTAIN, re-record /
   try another location" instead of a guess
4. **Reasons**: top 3 features pushing the decision (coefficient × standardised value)
5. Output in the panel: **Looks normal** (green), **Possible abnormal sound** (red),
   **Not sure – please re-check** (orange), or **Mixed sound – rough guess** (orange)

Training data = standalone HS/LS + the matched H/L reference files from dataset/Mix
(195 heart, 195 lung). Because many of these are exact copies (only 107 unique heart
and 94 unique lung clips), we use **grouped** cross validation: all copies of the same
audio stay in the same fold, so the model is never tested on a file it has already seen. When the user picks a recording that was in the training set,
the app uses the cross-validation model that did **not** see it, so the suggestion is honest.

Run `python advisor.py` to print the evaluation.

### Results (grouped 5-fold cross validation)
| | Heart | Lung |
|---|---|---|
| Sound type accuracy (10 / 6 classes) | **0.67** | **0.96** |
| Normal vs abnormal (balanced acc.) | 0.59 | 0.96 |
| % of cases where model answers (conf ≥ 0.5) | 94% | 99% |
| Accuracy on those cases | 0.68 | 0.96 |
| *First version, ~20 features only (type accuracy)* | *0.44* | *0.67* |

Chance level is about 0.10 for heart (10 classes) and 0.17 for lung (6 classes).
Our first version used normal (not grouped) cross validation and got much higher
numbers (0.74 heart, 0.84 lung), but those were inflated because copies of the same
audio ended up in both training and test folds. The table above is the honest result.
Adding MFCCs gave the big jump. The lung model is now good **on manikin sounds**; the heart
model is better at naming the type but still weak at normal vs abnormal (only 12 unique
normal heart clips to learn from).

### Model experiments (`experiments/`)
`python experiments/run_experiments.py` compares about 17 setups with the same grouped
5-fold CV. Full results in `experiments/RESULTS.md`.

| Model | Heart | Lung |
|---|---|---|
| Logistic regression, ~20 features (first version) | 0.41 | 0.64 |
| **Logistic regression + MFCC (used in the app)** | **0.65** | **0.97** |
| LSTM on log-mel frames (deep model) | 0.54 | 0.86 |
| Small CNN on spectrograms | did not learn | |

(averaged over several random fold splits, so slightly different from the table above)

- **The LSTM was worse than the simple model**: 107 / 94 unique clips are too few for a
  deep model. Good features + a small model worked better.
- Adding the recording location as a feature made it worse (the location labels are not
  consistent between copies of the same audio).
- **Leakage check**: with shuffled labels the score drops to chance (0.10 / 0.19).
- **Caveat**: we picked the best of ~17 setups on the same folds, which flatters the
  winner a little. With the choice made inside the CV (nested), heart is about 0.61.

### What did not work (honestly)
- **Mixed recordings**: splitting heart/lung with filters and using the same models gives
  only 16% (heart) / 65% (lung) type accuracy, and the model is often confidently wrong.
  So for mixed recordings the app only shows a "rough guess" instead of a verdict. Proper
  source separation (like NMF in the papers) is needed first.
- **Heart-rate rule**: we tried "rate > 100 bpm → tachycardia", but our rate estimate
  flagged 10 normal recordings and missed 8 of 19 tachycardias, so we removed it.
- The manikin probably plays the same source sound for each type at different locations,
  so the scores are likely optimistic. Not tested on real patients.
- **New / synthetic recordings**: the first version (no MFCC) got both synthetic files wrong.
  With MFCCs the synthetic normal heart is correct (Normal 90%), but the synthetic wheeze is
  still called Fine Crackles. The model has only learned the manikin's sounds, so sounds
  from anywhere else can be outside its experience.

## Still to do
- [ ] Source separation of mixed sounds (e.g. NMF), checked against the H/L reference
      files, so the decision layer also works on mixed recordings
- [ ] Statistical tests between sound types (Kruskal-Wallis etc.)
- [ ] Export features / plots to CSV and PNG from the GUI
- [ ] Test the decision layer with added noise (as the dataset paper suggests)
