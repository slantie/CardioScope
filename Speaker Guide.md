# CardioScope: speaker notes (15 slides, about 20 min talk + 10 min demo and questions)

The split follows who built what:

| Who | Slides |
|---|---|
| **Kandarp** (GUI and the rest) | 1, 2, 3, 4, 7, 8, 9, 12 (and drives the demo) |
| **Meet** (graphs and plots) | 5, 6, and the plot part of the demo |
| **Harshal** (model and features) | 10, 11, 13 |
| **All** | 14, 15 |

## 1 · Cover (Kandarp)
CardioScope, Group 2, PC503. One line: "a Python desktop app to explore heart and lung sounds and get a simple, explainable suggestion".

## 2 · Table of contents (Kandarp)
Data first, then how it is built, then the decision layer, results and the demo.

## 3 · Introduction (Kandarp)
- Heart and lung sounds are hard to hear and compare. HLS-CMDS has 535 labelled recordings but no tool to explore them.
- Everything from the proposal is done. Beyond it we added the suggestion and "check a new recording".

## 4 · Dataset overview (Kandarp)
- 12 locations on a manikin: 50 heart, 50 lung and 145 mixed recordings, 15 s each at 4000 Hz.
- Every mixed file has a matched heart-only and lung-only file.
- The data is clean, but it comes from a manikin, not real patients.

## 5 · What we found in the data (Meet)
- Only 344 of 535 files are unique. Copies would inflate the test scores.
- The mixtures were recorded separately: waveform correlation 0.01, spectrum similarity 0.98.
- Heart energy is below 200 Hz.
- Small issues: 14 old lung codes, 36 clipped files, and 4000 Hz instead of the 22,050 Hz the paper states.

## 6 · Graphs & plots (Meet)
The six charts, all drawn by one `Plotter` class:
- **Waveform:** each spike is a heartbeat.
- **FFT:** heart energy is below 200 Hz.
- **Spectrogram:** a wheeze shows as horizontal lines.
- **Bands:** a mixed file split at 200 Hz.
- **Class distribution** and **feature box plots:** these work on whatever the filters select.

The Compare and Matched windows reuse the same drawing function.

## 7 · Core workflow (Kandarp)
Load (`Dataset`) → Measure (`features.py`) → Decide (`Advisor`) → Show (`CardioScopeApp`).

## 8 · Class architecture (Kandarp)
- Five layers. Each one only uses the layers below it.
- 14 small classes in 12 files, one job each.
- `main()` creates the parts once and passes them into the window.

## 9 · SOLID design (Kandarp)
- **S:** one class per job.
- **O:** the `FILTERS` list; a new filter is one line.
- **D:** `main()` passes the parts in, and a test passes a fake file reader.
- **L and I** aren't forced, because there is no class hierarchy.

## 10 · Decision layer (Harshal)
- Recording → 80 features (about 20 explainable + 60 MFCC) → logistic regression (one for heart, one for lung) → 50% check.
- The panel shows one of four outcomes, with the top 3 matches and 3 reasons.
- Mixed recordings only get a rough guess.

## 11 · Results (Harshal)
- Grouped 5-fold CV (copies stay in one fold) gives 67% heart and 96% lung.
- First version 44% / 67%; LSTM 54% / 86%; the CNN did not learn.
- Shuffled labels give chance level, so nothing leaks.
- Say these upfront:
  - Plain CV gave 74% / 84% only because of the copies.
  - Heart normal vs abnormal is still weak (0.59).

## 12 · The app → live demo (Kandarp drives, Meet explains the charts)
1. Class Distribution
2. A normal heart (green)
3. `M_LDM_LC` murmur (red)
4. Bands on a mixed recording
5. Compare Two Recordings
6. Import WAV: `samples/synthetic_heart_normal.wav` ("Looks normal"). The dialog opens in `recordings/`, so go up one folder.

## 13 · Conclusion and future work (Harshal)
- Honest testing and good features beat a deeper model.
- Next steps: NMF separation for mixed recordings, real patients, noise testing, statistical tests.

## 14 · References (all)
The dataset, the descriptor paper, the separation papers and the Python libraries.

## 15 · Thank you (all)
Questions.
