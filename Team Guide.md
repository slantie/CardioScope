# CardioScope: Team Guide

Who explains what, file by file and function by function.
The code is in `CardioScope Project/app/`. Every file name below is in that folder.

| Who | Area | Files |
|---|---|---|
| **Kandarp** | GUI and everything around it | `dataset.py` ✔, `app.py`, `filters.py`, `player.py`, `recordings.py`, `quality.py`, `QualityWindow` (in `windows.py`), `test_cardioscope.py`, `make_synthetic.py` |
| **Meet** | Graphs and plots | `plots.py`, `CompareWindow` and `MatchedWindow` (in `windows.py`), the two dataset-chart methods in `app.py`, and the plotting helpers in `features.py` (`get_fft`, `bandpass`, `spectrum_similarity`) |
| **Harshal** | Model and features | `features.py`, `model.py`, `advisor.py`, `report.py`, `experiments/` |

**The big picture in one line:**
`Dataset` loads the files → `features.py` turns a signal into numbers → `Advisor` / `DecisionModel` decide → `Report` writes the text → `CardioScopeApp` shows it, with `Plotter` drawing the charts.

```
main()  (app.py)
 ├─ Dataset()            ← Kandarp
 ├─ Advisor(dataset)     ← Harshal
 ├─ NewRecordings()      ← Kandarp
 ├─ AudioPlayer()        ← Kandarp
 └─ CardioScopeApp(root, dataset, advisor, recordings, player)   ← Kandarp
       ├─ Plotter(fig, canvas)           ← Meet
       ├─ RecordingFilter(table)         ← Kandarp
       └─ SuggestionPanel                ← Kandarp, shows Harshal's Report
```

---

# Part 1: Kandarp (GUI and the rest)

## `dataset.py`: done ✔

## `app.py`: the main window

**Constants and helper classes**

| Name | What it does |
|---|---|
| `RECORD_SECONDS = 15` | Length of a microphone recording. It matches the 15 s dataset files. |
| `class SuggestionPanel` | The right-hand panel: a coloured heading plus a read-only text box. |
| `SuggestionPanel.__init__(parent, compact)` | Builds the "Suggestion" label, the heading label and the `Text` widget. They are narrower when `compact`. |
| `SuggestionPanel.show(title, colour, text)` | Changes the heading text and colour and replaces the text. The text box is unlocked, written and locked again, so the user can't type into it. |

**`class CardioScopeApp`: building the window**

| Method | What it does |
|---|---|
| `__init__(root, dataset, advisor, recordings, player)` | Stores the parts it receives (it never creates them itself; that's Dependency Inversion). Then it:<ul><li>creates the `RecordingFilter`</li><li>picks a compact layout if the screen is narrower than 1500 px (projectors)</li><li>builds the heading and the three panels, and fills the list</li><li>starts `advisor.train` in a background thread so the window opens immediately</li></ul> |
| `side_frame(side, fill, expand)` | A small helper that makes a padded `Frame` and packs it on one side. It saves repeating 2 lines 3 times. |
| `build_heading()` | The title "CardioScope", the subtitle with the number of recordings, and a line under them. |
| `build_left_panel()` | Builds the whole left column:<ul><li>one dropdown per entry in `FILTERS` (a loop, so a new filter needs no GUI code)</li><li>the search box and Reset button</li><li>the recording list with a scrollbar</li><li>the 5 EDA buttons</li><li>the "Check a New Recording" box (type dropdown, Import, Record)</li></ul> |
| `build_middle_panel()` | The info line and the plot buttons (Waveform, FFT, Spectrogram, Bands, Matched H-L-M, Play, Stop). Also the matplotlib figure with its toolbar, and the feature line under it. On small screens, Play and Stop move to their own row. |

**`class CardioScopeApp`: filters and the list**

| Method | What it does |
|---|---|
| `chosen()` | `{"Category": "Heart", "Gender": "All", ...}`, the current value of each dropdown. |
| `update_filter_options()` | Refills every dropdown with only the values that exist in the chosen category. If the current choice no longer exists, it resets it to "All". |
| `on_filter_change(label)` | When Category changes, it updates the other dropdowns first, then the list. |
| `reset_filters()` | Sets everything back to "All" and clears the search. |
| `update_list()` | Asks `RecordingFilter.apply()` for the matching rows, refills the list box and updates the "N of 245 recordings" label. |

**`class CardioScopeApp`: one recording**

| Method | What it does |
|---|---|
| `on_select(event)` | Runs when you click a recording. It reads the WAV through `dataset.read` and shows an error box if that fails, then calls `show_recording`. |
| `show_recording(row)` | Fills the info line and the feature line (RMS, peak, ZCR, …), draws the waveform and updates the suggestion. It is used for dataset files and for new recordings alike. |
| `need_signal()` | Returns `False` and shows a message if nothing is selected yet. It guards every button. |
| `plot(draw)` | Receives a *method* such as `self.plotter.fft` and calls it with the current signal. That way the 4 plot buttons share one function. |
| `update_suggestion()` | If the models are still training, it shows "Loading model..." and calls itself again in 300 ms (`root.after`). Otherwise it does three things:<ul><li>builds the key `"folder/file"` (`None` for new recordings)</li><li>calls `advisor.decide`</li><li>wraps the result in `Report` and shows it</li></ul> |

**`class CardioScopeApp`: audio and new recordings**

| Method | What it does |
|---|---|
| `play()` | `player.play(signal, sr)`. It shows a warning instead of crashing if there's no sound device. |
| `new_row(name, source)` | A fake table row for an outside recording: folder `"new"`, sound type `"unknown"`. |
| `import_wav()` | A file dialog that opens in `recordings/`. It runs `recordings.read`, which normalises the file, then `show_recording`. |
| `record()` | Starts the microphone, disables the button, and schedules `finish_recording` after 15.3 s. `root.after` waits without freezing the window. |
| `finish_recording()` | Gets the audio and saves the raw file to `recordings/`. Then it prepares the audio (mono, 4000 Hz, dataset loudness) and shows it. |

**`class CardioScopeApp`: extra windows and dataset charts**

| Method | What it does |
|---|---|
| `open_compare()` | Opens `CompareWindow`, with the current recording already chosen as A. |
| `open_matched()` | Only for dataset Mixed recordings. Opens `MatchedWindow`. |
| `show_data_quality()` | Shows the busy cursor, runs `QualityCheck(dataset).run()` (it reads all 535 files), then opens `QualityWindow`. |
| `busy(on)` | Switches the watch cursor on or off. `root.update()` makes it appear right away. |
| `feature_table()` | Computes the features of all 245 recordings **once** (about 1 s) and caches them in `self.feat_table`. Then it keeps only the filtered rows (`merge`). |
| `show_distribution()` / `show_feature_boxplot()` | Ask Meet's `Plotter` to draw the dataset charts for the filtered rows (Meet explains these). |
| `export_csv()` | Saves the filtered rows plus their features to a CSV of your choice. |

**`main()`** is the only place where the parts are created:
1. If the dataset folder is missing, it shows a dialog and quits.
2. Otherwise it builds `CardioScopeApp(root, dataset, Advisor(dataset), NewRecordings(), AudioPlayer())` and runs `root.mainloop()`, which waits for clicks forever.

## `filters.py`

| Name | What it does |
|---|---|
| `match_equal(value, chosen)` | A plain `==`. |
| `match_part(value, chosen)` | `"S3"` matches `"S3 + Normal"`, so mixed recordings show up when you filter by sound type. |
| `FILTERS` | A list of `(label, column, match function)`, one line per filter. **Open/Closed:** a new filter is one new line. The dropdowns, the filtering and the chart titles all loop over it. |
| `RecordingFilter(table)` | Holds the table. |
| `.parts(sound_type)` | `"S3 + Normal"` → `["S3", "Normal"]`. |
| `.options(label, column, category)` | The values for one dropdown. These are limited to the chosen category, except for the Category dropdown itself. |
| `.apply(chosen, search)` | Applies every filter that isn't "All", then searches the file name and sound type, ignoring case. |
| `.describe(chosen)` | `"Category: Heart, Gender: F"`, used in chart titles. |

## `player.py`

| Name | What it does |
|---|---|
| `try: import sounddevice` | If the import fails, `sd = None`. The app still works without sound. |
| `AudioPlayer.MIC_SR = 44100` | Microphones don't support 4000 Hz, so we record at 44.1 kHz and convert afterwards. |
| `_check()` | Raises a clear error if sounddevice is missing. |
| `_run(start)` | Stops any running sound and calls `start()`. If PortAudio fails because the audio device changed (headphones, Bluetooth), it restarts PortAudio once and tries again. This fixed a real bug we hit. |
| `play(x, sr)` | Resamples to the speaker's own rate (a 4000 Hz stream sometimes fails on macOS). It boosts the volume ×5, because the recordings are quiet, and clips to −1…1. |
| `stop()` | Stops playback. |
| `start_recording(seconds)` | `sd.rec(...)` starts recording in the background and returns immediately. |
| `finish_recording()` | `sd.wait()` waits for the recording to end, then returns `(audio, 44100)`. |

## `recordings.py`

| Name | What it does |
|---|---|
| `RECORDINGS_FOLDER`, `TARGET_SR = 4000`, `TARGET_RMS = 0.003` | Where saved recordings go, and what "looks like the dataset" means. |
| `NewRecordings.prepare(x, sr)` | Turns any audio into dataset-like audio:<ol><li>integers become −1…1</li><li>stereo becomes mono</li><li>the rate becomes 4000 Hz</li><li>the loudness is scaled to RMS 0.003</li></ol>**Why:** a phone recording was about 15× louder, and loudness is a model feature, so without this every outside recording looked "abnormal". |
| `read(path)` | Reads any WAV with scipy (any rate, mono or stereo), with clear error messages, then calls `prepare`. |
| `save(audio, sr, kind)` | Saves the *raw* microphone audio as 16-bit, named like `heart_20261116_103334.wav`, so files never overwrite each other. |

## `quality.py` + `QualityWindow`

| Name | What it does |
|---|---|
| `QualityCheck(dataset).run()` | Returns a list of `(level, message, suggested action)`. |
| `_label_problems()` | Counts the lung IDs whose CSV ID ≠ real file name (14 old codes). It reads LS.csv *raw* to count the extra spaces, which `Dataset.csv` would have stripped. |
| `_all_files()` | The 245 table files plus the 290 matched H and L files = 535. |
| `_audio_problems()` | One pass over all files, checking for:<ul><li>unreadable files</li><li>the sample rates (all 4000, although the paper says 22,050)</li><li>the durations</li><li>silent files</li><li>clipped files (peak ≥ 0.999)</li><li>**exact copies**, found with the md5 hash: only 344 of 535 files are unique</li></ul> |
| `QualityWindow(root, report)` (in `windows.py`) | Prints each line as `[LEVEL] message` followed by `-> Suggested: action`. |

## `test_cardioscope.py` (run `python -m pytest`, 14 tests)

| Test | Checks |
|---|---|
| `test_metadata_counts` | 50 heart, 50 lung, 145 mixed |
| `test_lung_ids_are_fixed` | `M_FC_LUA.wav` exists in the table, so the old code was fixed |
| `test_read_wav_format` | 4000 Hz, 60,000 samples, values within −1…1 |
| `test_missing_file_raises_error` | A wrong name gives `FileNotFoundError` |
| `test_features_on_sine_wave` | A 100 Hz sine has a dominant frequency near 100, the correct RMS, and over 90% of its energy below 200 Hz |
| `test_bandpass_removes_high_frequency` | The 800 Hz part is removed by the 20–200 Hz filter |
| `test_mixed_recordings_have_matched_files` | The H and L partner files exist |
| `test_report_text_has_top_matches` | The report text and heading are right |
| `test_features_work_with_any_reader` | **Dependency Inversion:** a fake reader that makes sine waves works |
| `test_import_any_wav` | A loud stereo 44.1 kHz file becomes mono 4000 Hz at dataset loudness |
| `test_save_recording` | The file is saved with the right name |
| `test_synthetic_samples_load` | The 2 sample files load |
| `test_mfcc_features` | 60 finite MFCC numbers |
| `test_filter_mixed_sound_type` | "S3" also finds "S3 + Normal"; `describe` gives the right text |

`tmp_path` is a pytest feature: an empty temporary folder for each test, so the tests don't leave files behind.

## `make_synthetic.py`
Makes `samples/synthetic_heart_normal.wav` (about 72 bpm, S1 then S2, a bit of random jitter) and `synthetic_lung_wheeze.wav` (breathing noise plus a whistle of about 180 Hz while breathing out). These are fake test sounds, not patients. We use them in the demo for "Import WAV".

---

# Part 2: Meet (graphs and plots)

## The three helpers from `features.py` that the plots use

| Function | What it does | Used by |
|---|---|---|
| `get_fft(x, sr)` | Returns `(frequencies, magnitude)`. `rfft` keeps only the positive frequencies. | FFT plot |
| `bandpass(x, sr, low, high)` | Keeps only `low`–`high` Hz. It's a Butterworth filter run forwards and backwards (`filtfilt`), so it doesn't shift the signal in time. | Bands plot |
| `spectrum_similarity(a, b, sr)` | Correlation of the two log spectra: 1.0 means the same frequencies. | Matched window |

## `plots.py`

| Name | What it does |
|---|---|
| `SHORT_NAMES` | "Late Diastolic Murmur" → "LDM", so the box plots with 10 heart types stay readable. |
| `draw_signal(ax, x, sr, kind, color)` | Draws one recording on one axis:<ul><li>**Waveform:** amplitude over time (x axis = sample number ÷ sr)</li><li>**FFT:** magnitude over frequency, x axis cut at 1000 Hz because almost all the energy is below that</li><li>**Spectrogram:** `specgram` with 256-sample windows, y axis cut at 1000 Hz</li></ul>It's shared by the main window *and* `CompareWindow`, so the plots look the same everywhere. |
| `Plotter(fig, canvas)` | Holds the matplotlib `Figure` and the Tk canvas it's shown on. |
| `_single(...)` | Clears the figure, adds one axis, calls `draw_signal`, sets the title and labels, adds a grid (not on spectrograms) and redraws. The 3 single plots below are one line each because of this. |
| `waveform` / `fft` / `spectrogram` | One line each: `_single` with the right title and labels. |
| `bands(x, sr)` | 3 rows sharing one time axis:<ul><li>the original</li><li>the 20–200 Hz heart band (red)</li><li>the 200–1000 Hz lung band (green)</li></ul>**Why those bands:** in the EDA, heart energy is almost all below 200 Hz (median centroid 94 Hz vs 158 Hz for lung). This is a rough split with filters, not real source separation. |
| `distribution(df, by_category, subtitle)` | Left: recordings per category, or per sound type when one category is chosen. Right: recordings per location. A mixed recording counts once for each of its two sound types. |
| `boxplot(df, by_category, subtitle)` | Box plots of 3 features (`spectral_centroid`, `dominant_freq`, `low_energy_ratio`) per group:<ul><li>3 groups or fewer: side by side</li><li>more than 3 (sound types): stacked vertically with short names, so there is room</li></ul> |
| `_finish(title)` | Adds the overall title ("N recordings (filters)"), applies `tight_layout` and redraws. |

**How to read the charts:**
- **Box plot:** the line is the median, the box is the middle 50%, the whiskers are the range, and the dots are outliers.
- **Spectrogram:** time on the x axis, frequency on the y axis, colour for loudness. Heart beats appear as vertical stripes low down; a wheeze is a horizontal line; crackles are short vertical spikes.

## `windows.py`: Compare and Matched

| Name | What it does |
|---|---|
| `CompareWindow(root, dataset, start_file)` | A new window (`Toplevel`) with two dropdowns of all 245 recordings and a plot-type dropdown. The currently selected recording starts as A. |
| `CompareWindow.compare()` | Reads both recordings and draws them with `draw_signal`, one above the other with a shared x axis (`sharex`) so they line up. Under the plots it shows a small table of 5 features for A and B. |
| `MatchedWindow(root, dataset, row)` | For a mixed file M####, it reads M plus its matched heart-only H#### and lung-only L#### file and plots all three. Then it computes two numbers:<ul><li>**waveform correlation:** same samples? about **0.01**, so no</li><li>**spectrum similarity:** same frequencies? about **0.98**, so yes</li></ul>**Conclusion:** the mixtures were *recorded separately*, not made by adding H and L. This is one of our key EDA findings. |

## The two chart methods in `app.py`
- `show_distribution()` calls `plotter.distribution(...)` by category when Category is "All", otherwise by sound type.
- `show_feature_boxplot()` gets the cached feature table (Harshal's `compute_all_features`) and calls `plotter.boxplot(...)`.
- The matplotlib toolbar (`NavigationToolbar2Tk`) gives zoom, pan and **save as PNG** for free.

## What Meet presents
- **Slide 5 (findings):**
  - 344 of 535 files are unique.
  - The mixtures were recorded separately.
  - Heart sounds sit below 200 Hz.
  - There are small label issues.
- **Slide 6 (Graphs & Plots):** the 6 charts on the slide, drawn by the app's own `Plotter` on real recordings:
  - waveform and FFT of `M_LDM_LC.wav`
  - spectrogram of the wheeze `M_W_LUA.wav`
  - bands of `M0001.wav`
  - class distribution and feature box plots of all 245 recordings
- **The plot part of the demo:** Class Distribution → Bands on a mixed recording → Compare → Matched H-L-M.

---

# Part 3: Harshal (model and features)

## How we got to the final features: the story

Every number below is from grouped 5-fold cross validation. Copies of the same audio are always kept in the same fold. The full log is in `experiments/RESULTS.md`.

1. **Start: about 20 hand-made features plus logistic regression.** Heart **44%**, lung **67%**. That's well above chance (10% and 17%), but weak.
2. **We checked whether our test was honest.**
   - Only 344 of 535 files are unique, so a plain (ungrouped) CV would put copies in both train and test. A later plain-CV check gave 74% heart / 84% lung, inflated by the copies.
   - So we switched to grouping by the **md5 hash** of the audio.
   - Shuffled labels then drop to chance level (heart 0.104, lung 0.185), so nothing leaks.
3. **We tried adding the recording location.** It got *worse* (28% / 43%). The location labels are unreliable: identical audio appears with *different* locations. So we dropped it.
4. **We added MFCCs (60 numbers of "sound texture").** Heart **67%**, lung **96%**. This was the only change that clearly helped.
5. **We tried other models on the same features.**
   - Random Forest (57% / 87%), gradient boosting (53% / 80%) and SVM (59% / 92%) were all worse than plain logistic regression.
   - Windowing the clip into 3 s pieces and augmentation (gain, shift, noise) gave no clear gain.
6. **We tried deep learning.**
   - **LSTM** on log-mel frames: 54% / 86%, worse.
   - **CNN** on spectrogram images: did not learn (near chance on lung).
   - With only about 80 unique training clips per fold there's too little data for a network.
7. **Honest lower bound.** We compared about 17 setups, so picking the best one is itself a small bias. Nested selection (choosing the model inside each training fold) gives heart **61%**, lung **97%**.
8. **Final choice:** about 20 explainable features + 60 MFCC statistics → `StandardScaler` → `LogisticRegression(C=1)`.
   - It's the simplest model.
   - It's the best or equal-best.
   - Every answer can be explained (see "reasons" below).

**What is still weak, and we say so:**
- **Heart normal vs abnormal:** balanced accuracy is only **0.59**. There are only 12 unique normal heart clips, and Normal, AV Block and Atrial Fibrillation get confused with each other. The murmurs are recognised well (84–95%).
- **The lung 96%** means "recognises the manikin's lung sound programs", not real patients.

## `features.py`: every feature, what it measures and why we used it

**Basic features (`extract_features`)**, also shown under the plot:

| Feature | What it measures | Why |
|---|---|---|
| `rms` | Average loudness: √(mean of x²) | Crackles and rubs are louder; also used to normalise new recordings |
| `peak` | The largest absolute value | Sudden loud clicks; also detects clipping |
| `zcr` | Zero crossing rate: how often the signal changes sign | High-pitched or noisy sounds cross zero more often (wheeze, fine crackles) |
| `dominant_freq` | The frequency with the most energy (Welch spectrum) | Heart is low (about 50–100 Hz), a wheeze is a clear pitch |
| `spectral_centroid` | The "centre of mass" of the spectrum, i.e. average pitch | Heart about 94 Hz vs lung about 158 Hz in our EDA |
| `low_energy_ratio` / `high_energy_ratio` | Share of energy below / above 200 Hz | The heart vs lung split we found in the EDA |

**Extra features in `model_features`**: what a listener pays attention to.

| Feature | What it measures | Why |
|---|---|---|
| `log_rms` | log10 of RMS | Loudness varies a lot; the log makes it more even for the model |
| `spec_entropy` | How spread out the spectrum is (0 = one pure tone, 1 = white noise) | A wheeze is tonal (low entropy); breathing noise and rubs are noisy (high) |
| `flatness` | Geometric mean ÷ arithmetic mean of the spectrum | Another "tone vs noise" measure |
| `band_20_50` … `band_800_1600` | Share of energy in 6 bands | A rough shape of the spectrum: S3/S4 are very low, murmurs mid, crackles high |
| `period_strength` | How strongly the loudness curve repeats (autocorrelation peak) | A regular rhythm is high; atrial fibrillation is low |
| `rate_per_min` | Beats or breaths per minute, from the repeat time | Tachycardia (more than 100 bpm); heart searches 35–220 bpm, lung 7.5–40 breaths/min |
| `interval_cv` | How much the gaps between peaks vary (std ÷ mean) | Irregular beats mean atrial fibrillation |
| `transient_rate` | Share of samples where the 300–1500 Hz band jumps above 6 × its median | Crackles are short sharp clicks |

**How the rhythm features work** (`envelope` → autocorrelation):
1. `envelope` filters the band (20–200 Hz for heart, 100–1000 Hz for lung).
2. It takes the Hilbert envelope (the loudness curve) and smooths it with a 10 Hz low-pass filter.
3. It keeps 200 values per second.
4. **Autocorrelation:** slide the curve over itself. The shift where it matches best is one beat (or one breath). `60 × rate ÷ shift` gives beats per minute.
5. `find_peaks` finds the individual beats, so we can measure how even the gaps are.

**MFCC** (`mel_filterbank`, `mfcc_features`): "sound texture", 60 numbers.
1. Cut the signal into 128 ms frames, one every 32 ms (`stft`).
2. Spectrum of each frame (power = |Z|²).
3. **Mel filterbank:** 40 triangular filters, narrow at low frequencies and wide at high ones, the way human hearing works.
4. Take the **log**, because we hear loudness roughly logarithmically.
5. **DCT** (discrete cosine transform) compresses the 40 values into 20 coefficients. Coefficient 0 is about overall loudness, the low ones describe the broad spectrum shape, the higher ones the finer detail.
6. A clip has hundreds of frames, but the model needs **one row**. So for each of the 20 coefficients we keep:
   - `mean`: the typical texture
   - `std`: how much it varies
   - `dstd`: how fast it changes from frame to frame

   20 × 3 = 60.

We wrote MFCC ourselves with NumPy and SciPy instead of using librosa, so we can explain every step and keep the install small.

**Why MFCCs helped so much:** our hand-made features each measure *one* thing. MFCCs describe the *whole* spectrum shape compactly, and that shape is exactly what makes a wheeze, a rub and a crackle sound different.

`compute_all_features(df, read)` runs `extract_features` on every row. It takes the reader *function* as a parameter, so a test can pass a fake one (the Dependency Inversion example).

## `model.py`

| Name | What it does |
|---|---|
| `N_FOLDS = 5` | 5-fold cross validation. |
| `new_pipeline()` | `StandardScaler` then `LogisticRegression(C=1, max_iter=5000)`. The scaler gives every feature mean 0 and std 1, so the weights are comparable. That is what makes the "reasons" meaningful. |
| `training_table(dataset, kind)` | **Rows:** the 50 standalone files plus the 145 matched H (or L) files from Mix.csv, so 195 rows. **X:** `model_features` of each. **y:** the sound type. **keys:** `"folder/file"`. **groups:** the md5 of the audio, so identical copies share a group. |
| `DecisionModel(dataset, kind)` | Trains the **full** model on all 195 rows. It also trains **5 fold models** with `StratifiedGroupKFold` ("stratified" keeps the class mix in each fold; "group" keeps copies together). Two more things are kept:<ul><li>`fold_of[key]`: the fold each file was tested in</li><li>`oof_proba`: every file's prediction from the model that didn't see it</li></ul> |
| `predict(feats, key)` | **The model:** a dataset file uses its *fold model*, which never saw it, so the demo is honest; a new recording uses the full model. **Returns** the top 3 `(type, probability)` and 3 reasons. **A reason** is the scaled feature value × the model's weight for the winning class; the 3 biggest positive pushes are shown as "high X" or "low X". |
| `evaluation(threshold)` | The slide numbers, from the out-of-fold predictions: type accuracy, normal-vs-abnormal balanced accuracy, and how many recordings pass the threshold and how accurate they are. |

## `advisor.py`

| Name | What it does |
|---|---|
| `CONFIDENCE_THRESHOLD = 0.5` | Below 50% the app says "not sure" instead of guessing. |
| `Advisor(dataset)` | Holds `models = {}`, which stays empty until trained. |
| `train()` | Builds the heart and lung `DecisionModel`s (about 4 s; the app runs this in a thread). |
| `is_ready()` | `True` when both models exist. |
| `decide(x, sr, category, key)` | **Heart or Lung:** one judgement with that model. **Mixed:** splits with the two filters, judges each part, and marks both `LOW RELIABILITY`. We tested all 145 mixtures and got only 15% heart / 34% lung right, so we never give a verdict there. |
| `_judge(kind, x, sr, key)` | Features, then `predict`, then the level:<ul><li>confidence < 0.5 → `UNCERTAIN`</li><li>top = Normal → `LIKELY NORMAL`</li><li>otherwise → `CHECK`</li></ul>It returns a dict: kind, level, best, confidence, top3, reasons. |
| `python advisor.py` | Prints the evaluation. |

**About the 50% gate:** it rarely triggers (6% of heart and 1% of lung recordings fall below it), and accuracy on the rest only goes from 67% to 68%. It's a safety net, not a big accuracy boost. Its point is that the app *can* say "I don't know".

`python advisor.py` gives:
- **Heart:** 195 recordings, accuracy 0.67, normal-vs-abnormal 0.59, 94% confident, accuracy when confident 0.68.
- **Lung:** 195 recordings, accuracy 0.96, normal-vs-abnormal 0.96, 99% confident, accuracy when confident 0.96.

## `report.py`

| Name | What it does |
|---|---|
| `FRIENDLY_NAMES` | Column name → words people understand (`zcr` → "zero crossing rate"). |
| `DESCRIPTIONS` | One textbook line per sound type. |
| `friendly_name(column)` | Builds the MFCC names (`mfcc3_mean` → "sound texture (MFCC 3 average)") and looks up the rest. |
| `Report(results)` | Wraps the list of result dicts. |
| `heading()` | One heading for all results, where the most serious level wins:<ul><li>`CHECK`: red, "Possible abnormal sound"</li><li>`LOW RELIABILITY`: orange, "Mixed sound - rough guess"</li><li>`UNCERTAIN`: orange, "Not sure"</li><li>otherwise: green, "Looks normal"</li></ul> |
| `text()` | For each result, it writes the kind, the verdict lines, the top 3 matches as a text bar (6 `#` = 100%) and the 3 reasons. |
| `_verdict(r)` | The wording: never "you have X". Always "most similar to" or "suggest getting it checked by a doctor". |

## What Harshal presents
- **Slide 10:** decision layer.
- **Slide 11:** results and the experiment table.
- **Slide 13:** conclusion and future work. The future work is NMF separation for mixtures, real patients, a noise test and statistics.

**Likely questions:**
- "Why not deep learning?"
- "Is 96% real?"
- "Why is heart worse?"
- "How do you get the reasons?"
- "Why does mixed only get a rough guess?"

All five are answered above.

---

# Shared: who calls whom (the hand-off points)

| From | To | What |
|---|---|---|
| Kandarp `app.py` | Meet `Plotter` | `self.plot(self.plotter.fft)`, `plotter.distribution(...)`, `plotter.boxplot(...)` |
| Kandarp `app.py` | Meet `windows.py` | `CompareWindow(...)`, `MatchedWindow(...)` |
| Kandarp `app.py` | Harshal `Advisor` | `threading.Thread(target=advisor.train)`, `advisor.is_ready()`, `advisor.decide(...)` |
| Harshal `Report` | Kandarp `SuggestionPanel` | `suggestion.show(*report.heading(), report.text())` |
| Meet boxplot | Harshal `features` | `compute_all_features(dataset.table, dataset.read)` |
| Everyone | Kandarp `Dataset` | `dataset.read(folder, file)`, `dataset.table` |
