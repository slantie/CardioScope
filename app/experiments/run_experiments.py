# run_experiments.py
# Tries to improve the heart / lung type classifiers of model.py, HONESTLY.
#
# Run from the app folder:
#     ../.venv/bin/python experiments/run_experiments.py            (everything, ~10-15 min)
#     ../.venv/bin/python experiments/run_experiments.py --quick    (only seed 42)
#
# Protocol (the same for every experiment):
#   * rows = the 195 heart (or lung) recordings of model.training_table
#   * StratifiedGroupKFold(5, shuffle=True, random_state=seed), groups = md5 of the audio,
#     so byte-identical copies never sit in train AND test
#   * windows and augmented copies inherit the group (and the fold) of their parent clip;
#     test data is never augmented; window predictions are averaged per recording
#   * all hyperparameters are FIXED up front (written below), never tuned on test folds.
#     Neural nets use early stopping on an inner grouped validation split taken from the
#     training fold only. One extra "nested selection" experiment picks the classical model
#     inside each training fold, to show what happens when the choice itself is honest.
#   * metrics: type accuracy (all recordings), normal-vs-abnormal balanced accuracy,
#     mean +- std over the 5 folds, repeated for 3 seeds of the fold split.

import argparse
import hashlib
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import signal as sp
from scipy.fft import dct
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so we can import the project modules
import features  # noqa: E402
import model  # noqa: E402
from dataset import Dataset  # noqa: E402

warnings.filterwarnings("ignore")
SEEDS = [42, 7, 2024]          # 42 is the split used by model.py
N_FOLDS = 5
LOCATIONS = ["Apex", "LC", "LLA", "LLSB", "LMA", "LUA", "LUSB", "RC", "RLA", "RMA", "RUA", "RUSB"]

# window settings (experiment 2 and 3)
WIN_SEC, HOP_SEC = 3.0, 1.5    # 3 s windows, 50 % overlap -> 9 windows per 15 s clip


# =============================================================== 1. data
def load_rows(kind):
    """Same rows, in the same order, as model.training_table(kind),
    plus the recording location (known metadata) and the raw signal."""
    data = Dataset()
    df = data.table
    mix = data.csv("Mix.csv")
    rows = []
    standalone = df[df["category"] == ("Heart" if kind == "heart" else "Lung")]
    for r in standalone.itertuples():
        rows.append((r.folder, r.file, r.sound_type, r.location))
    id_col = "Heart Sound ID" if kind == "heart" else "Lung Sound ID"
    type_col = "Heart Sound Type" if kind == "heart" else "Lung Sound Type"
    for _, r in mix.iterrows():
        rows.append(("Mix", r[id_col] + ".wav", r[type_col], r["Location"]))

    signals = []
    for z, f, _, _ in rows:
        x, sr = data.read(z, f)
        signals.append(x)
    groups = np.array([hashlib.md5(x.tobytes()).hexdigest() for x in signals])
    return {
        "kind": kind, "sr": sr, "signals": signals, "groups": groups,
        "y": np.array([r[2] for r in rows]), "location": np.array([r[3] for r in rows]),
        "keys": [f"{z}/{f}" for z, f, _, _ in rows],
    }


# =============================================================== 2. features
def mel_filterbank(sr, n_fft, n_mels=40, fmin=20, fmax=2000):
    """Triangular mel filters (same idea as librosa, written by hand)."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return fb


N_FFT, HOP = 512, 128          # 128 ms frames (7.8 Hz resolution), 32 ms hop
_FB = {}


def log_mel(x, sr):
    """Log-mel spectrogram, shape (frames, 40)."""
    if sr not in _FB:
        _FB[sr] = mel_filterbank(sr, N_FFT)
    _, _, Z = sp.stft(x, fs=sr, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                      boundary=None, padded=False)
    power = np.abs(Z) ** 2
    return np.log(_FB[sr] @ power + 1e-10).T


def mfcc_features(x, sr, n_mfcc=20):
    """MFCC mean + std over time, and std of the frame-to-frame change (delta)."""
    m = dct(log_mel(x, sr), type=2, axis=1, norm="ortho")[:, :n_mfcc]
    d = np.diff(m, axis=0)
    f = {}
    for i in range(n_mfcc):
        f[f"mfcc{i}_mean"] = m[:, i].mean()
        f[f"mfcc{i}_std"] = m[:, i].std()
        f[f"mfcc{i}_dstd"] = d[:, i].std()
    return f


def rich_features(x, sr, kind):
    """The ~20 hand-crafted model.py features + 60 MFCC statistics."""
    f = features.model_features(x, sr, kind)
    f.update(mfcc_features(x, sr))
    return f


def location_onehot(locs):
    return pd.DataFrame({f"loc_{l}": (np.asarray(locs) == l).astype(float) for l in LOCATIONS})


def augment(x, rng, sr):
    """Random gain (x0.5..x1.5), circular time shift (up to 1 s) and light noise (SNR 25-40 dB)."""
    y = np.roll(x, rng.integers(-sr, sr)) * rng.uniform(0.5, 1.5)
    snr_db = rng.uniform(25, 40)
    noise = rng.standard_normal(len(y)) * np.sqrt(np.mean(y ** 2) / 10 ** (snr_db / 10))
    return (y + noise).astype(np.float32)


def windows(x, sr):
    w, h = int(WIN_SEC * sr), int(HOP_SEC * sr)
    return [x[s:s + w] for s in range(0, len(x) - w + 1, h)]


# =============================================================== 3. models (FIXED hyperparameters)
# Chosen up front from common defaults for small tabular data, not tuned on the folds.
def make_model(name, seed):
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000))
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                      n_jobs=-1, random_state=seed)
    if name == "HGB":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                              l2_regularization=1.0, early_stopping=False,
                                              random_state=seed)
    if name == "SVM":
        return make_pipeline(StandardScaler(),
                             SVC(C=10, gamma="scale", probability=True, random_state=seed))
    raise ValueError(name)


# =============================================================== 4. evaluation helpers
def folds(d, seed):
    skf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(d["y"])), d["y"], d["groups"]))


def is_normal(a):
    return np.where(np.asarray(a) == "Normal", "Normal", "Abnormal")


def proba_in_order(model, X, classes):
    """predict_proba with columns in the global class order."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, list(classes).index(c)] = p[:, j]
    return out


def score(d, oof, fold_list, classes):
    """Per-fold type accuracy and normal-vs-abnormal balanced accuracy."""
    pred = classes[oof.argmax(1)]
    accs, bals = [], []
    for _, te in fold_list:
        accs.append(accuracy_score(d["y"][te], pred[te]))
        bals.append(balanced_accuracy_score(is_normal(d["y"][te]), is_normal(pred[te])))
    # accuracy counting every unique clip once (copies are averaged within their group)
    correct = pd.Series(pred == d["y"]).groupby(d["groups"]).mean()
    return {"acc": np.mean(accs), "acc_std": np.std(accs), "bal": np.mean(bals),
            "bal_std": np.std(bals), "acc_pooled": accuracy_score(d["y"], pred),
            "acc_unique": correct.mean(), "pred": pred}


def run_cv(d, fit_predict, seed):
    """fit_predict(train_idx, test_idx, seed) -> probabilities for the test rows."""
    classes = np.unique(d["y"])
    fl = folds(d, seed)
    oof = np.zeros((len(d["y"]), len(classes)))
    for tr, te in fl:
        oof[te] = fit_predict(tr, te, seed)
    return score(d, oof, fl, classes)


# =============================================================== 5. experiments
def exp_tabular(d, X, model_name):
    """One feature row per recording, one sklearn model."""
    classes = np.unique(d["y"])

    def fp(tr, te, seed):
        m = make_model(model_name, seed).fit(X.iloc[tr], d["y"][tr])
        return proba_in_order(m, X.iloc[te], classes)
    return fp


def exp_tabular_augmented(d, X, X_aug, aug_parent, model_name):
    """Training fold = its clips + augmented copies OF THOSE CLIPS only. Test = originals."""
    classes = np.unique(d["y"])

    def fp(tr, te, seed):
        keep = np.isin(aug_parent, tr)                     # augmented copies of train clips
        Xtr = pd.concat([X.iloc[tr], X_aug[keep]], ignore_index=True)
        ytr = np.concatenate([d["y"][tr], d["y"][aug_parent[keep]]])
        m = make_model(model_name, seed).fit(Xtr, ytr)
        return proba_in_order(m, X.iloc[te], classes)
    return fp


def exp_windows(d, Xw, w_parent, model_name):
    """Train on windows of the training clips, average window probabilities per test clip."""
    classes = np.unique(d["y"])

    def fp(tr, te, seed):
        wtr = np.isin(w_parent, tr)
        m = make_model(model_name, seed).fit(Xw[wtr], d["y"][w_parent[wtr]])
        out = np.zeros((len(te), len(classes)))
        for k, i in enumerate(te):
            out[k] = proba_in_order(m, Xw[w_parent == i], classes).mean(0)
        return out
    return fp


def exp_nested_selection(d, X_base, X_rich, candidates):
    """Inside each OUTER training fold, choose the model by an INNER grouped 4-fold CV,
    then refit it on the whole outer training fold. Honest estimate of 'pick the best'."""
    classes = np.unique(d["y"])
    feats = {"base": X_base, "rich": X_rich}
    chosen = []

    def fp(tr, te, seed):
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
        best, best_acc = None, -1
        for feat, name in candidates:
            X = feats[feat]
            accs = []
            for itr, ite in inner.split(np.zeros(len(tr)), d["y"][tr], d["groups"][tr]):
                m = make_model(name, seed).fit(X.iloc[tr[itr]], d["y"][tr[itr]])
                accs.append(accuracy_score(d["y"][tr[ite]], m.predict(X.iloc[tr[ite]])))
            if np.mean(accs) > best_acc:
                best, best_acc = (feat, name), np.mean(accs)
        chosen.append(f"{best[1]}-{best[0]}")
        m = make_model(best[1], seed).fit(feats[best[0]].iloc[tr], d["y"][tr])
        return proba_in_order(m, feats[best[0]].iloc[te], classes)
    return fp, chosen


def exp_binary(d, X, model_name):
    """Dedicated Normal-vs-Abnormal model (class_weight balanced), heart only.
    Returns a fit_predict that gives 2-column probabilities [Abnormal, Normal]."""
    yb = is_normal(d["y"])

    def fp(tr, te, seed):
        if model_name == "LR":
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1, max_iter=5000, class_weight="balanced"))
        else:
            m = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=seed)
        m.fit(X.iloc[tr], yb[tr])
        return proba_in_order(m, X.iloc[te], np.array(["Abnormal", "Normal"]))
    return fp


def run_binary_cv(d, fit_predict, seed):
    yb = is_normal(d["y"])
    fl = folds(d, seed)                     # same folds as the type experiments
    bals = []
    for tr, te in fl:
        p = fit_predict(tr, te, seed)
        pred = np.where(p[:, 1] >= 0.5, "Normal", "Abnormal")
        bals.append(balanced_accuracy_score(yb[te], pred))
    return np.mean(bals), np.std(bals)


# =============================================================== 6. neural nets (torch)
def neural_experiment(d, arch, seed, n_aug=2, epochs=40, patience=8):
    """Tiny LSTM or CNN on 3 s log-mel windows.
    Training: random 3 s crops of the training clips (+ their augmented copies),
    early stopping on an inner grouped validation split of the TRAINING fold.
    Test: fixed 50 %-overlap windows of the original test clip, probabilities averaged."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)

    sr = d["sr"]
    classes = np.unique(d["y"])
    y_idx = np.searchsorted(classes, d["y"])
    win = int(WIN_SEC * sr / HOP)          # frames per window (~90)

    # spectrograms of originals and of augmented copies (augmented ones only used for training)
    rng = np.random.default_rng(seed)
    specs = [log_mel(x, sr).astype(np.float32) for x in d["signals"]]
    aug_specs = [[log_mel(augment(x, rng, sr), sr).astype(np.float32) for _ in range(n_aug)]
                 for x in d["signals"]]

    class LSTMNet(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            self.lstm = nn.LSTM(40, 32, num_layers=1, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(128, n_cls)

        def forward(self, x):                      # x: (batch, frames, 40)
            h, _ = self.lstm(x)
            h = torch.cat([h.mean(1), h.max(1).values], 1)
            return self.out(self.drop(h))

    class CNNNet(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                     nn.ReLU(), nn.MaxPool2d(2))
            self.f = nn.Sequential(block(1, 8), block(8, 16), block(16, 32))
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(32, n_cls)

        def forward(self, x):                      # x: (batch, frames, 40)
            h = self.f(x.unsqueeze(1)).mean((2, 3))
            return self.out(self.drop(h))

    def test_windows(s):
        step = win // 2
        return np.stack([s[i:i + win] for i in range(0, len(s) - win + 1, step)])

    def fp(tr, te, seed):
        torch.manual_seed(seed)
        r = np.random.default_rng(seed)
        # inner validation split (by group) from the TRAINING fold only
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
        itr, iva = next(inner.split(np.zeros(len(tr)), d["y"][tr], d["groups"][tr]))
        fit_ids, val_ids = tr[itr], tr[iva]

        # per-band normalisation from the fitting clips only
        allf = np.concatenate([specs[i] for i in fit_ids])
        mu, sd = allf.mean(0), allf.std(0) + 1e-6
        norm = lambda s: (s - mu) / sd

        pool = [(norm(s), y_idx[i]) for i in fit_ids for s in [specs[i]] + aug_specs[i]]
        Xva = [torch.tensor(norm(test_windows(specs[i]))) for i in val_ids]

        net = (LSTMNet if arch == "LSTM" else CNNNet)(len(classes))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
        lossf = nn.CrossEntropyLoss()
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            net.train()
            # one epoch = 4 random 3 s crops from every training spectrogram (time-shift aug)
            xs, ys = [], []
            for s, yy in pool:
                for _ in range(4):
                    st = r.integers(0, len(s) - win + 1)
                    xs.append(s[st:st + win] + r.normal(0, 0.3))  # random gain = shift in log domain
                    ys.append(yy)
            order = r.permutation(len(xs))
            xs, ys = np.stack(xs)[order], np.array(ys)[order]
            for b in range(0, len(xs), 32):
                opt.zero_grad()
                loss = lossf(net(torch.tensor(xs[b:b + 32], dtype=torch.float32)),
                             torch.tensor(ys[b:b + 32]))
                loss.backward()
                opt.step()
            # validation loss per recording (mean window probability)
            net.eval()
            with torch.no_grad():
                pv = torch.stack([torch.softmax(net(x), 1).mean(0) for x in Xva])
                vl = float(nn.functional.nll_loss(torch.log(pv + 1e-9),
                                                  torch.tensor(y_idx[val_ids])))
            if vl < best:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            return np.stack([torch.softmax(net(torch.tensor(norm(test_windows(specs[i])))), 1)
                             .mean(0).numpy() for i in te])

    return run_cv(d, fp, seed)


# =============================================================== 7. sanity check (leakage test)
def permutation_check(n_perm=10):
    """Shuffle the class labels BETWEEN unique clips (copies keep sharing one label) and rerun
    the recommended pipeline. If the pipeline does not leak, accuracy must fall to chance."""
    for kind in ["heart", "lung"]:
        d = load_rows(kind)
        X_rich = pd.DataFrame([rich_features(x, d["sr"], kind) for x in d["signals"]])
        uniq_g = np.unique(d["groups"])
        g_label = {g: d["y"][d["groups"] == g][0] for g in uniq_g}
        real = run_cv(d, exp_tabular(d, X_rich, "LR"), 42)["acc"]
        accs = []
        for k in range(n_perm):
            rng = np.random.default_rng(100 + k)
            shuffled = dict(zip(uniq_g, rng.permutation([g_label[g] for g in uniq_g])))
            dp = dict(d, y=np.array([shuffled[g] for g in d["groups"]]))
            accs.append(run_cv(dp, exp_tabular(dp, X_rich, "LR"), 42)["acc"])
        print(f"{kind}: real labels {real:.3f} | shuffled labels {np.mean(accs):.3f} "
              f"(max {np.max(accs):.3f} over {n_perm} shuffles) | chance {1 / len(set(d['y'])):.3f}")


# =============================================================== 8. main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only seed 42")
    ap.add_argument("--no-neural", action="store_true")
    ap.add_argument("--sanity", action="store_true", help="only run the label-permutation check")
    args = ap.parse_args()
    if args.sanity:
        permutation_check()
        return
    seeds = [42] if args.quick else SEEDS
    t_start = time.time()
    results = []   # (kind, experiment, seed, metrics, seconds)

    def record(kind, name, seed, res, sec):
        results.append({"kind": kind, "experiment": name, "seed": seed, "acc": res["acc"],
                        "acc_std": res["acc_std"], "bal": res["bal"], "bal_std": res["bal_std"],
                        "acc_unique": res["acc_unique"], "seconds": sec})
        print(f"  {name:<34} seed {seed:<5} type acc {res['acc']:.3f} +- {res['acc_std']:.3f}"
              f"  N/A bal {res['bal']:.3f} +- {res['bal_std']:.3f}  ({sec:.1f}s)", flush=True)

    for kind in ["heart", "lung"]:
        print(f"\n================ {kind.upper()} ================", flush=True)
        d = load_rows(kind)

        # --- sanity check: same rows / labels / groups as model.py
        X_base, y_ref, keys_ref, g_ref = model.training_table(Dataset(), kind)
        assert list(keys_ref) == d["keys"] and (y_ref == d["y"]).all() and (g_ref == d["groups"]).all()

        # --- per-class unique clip counts
        uniq = pd.DataFrame({"y": d["y"], "g": d["groups"]}).drop_duplicates("g")["y"].value_counts()
        print(f"rows {len(d['y'])}, unique clips {d['groups'].size and len(set(d['groups']))}")
        print("unique clips per class:", dict(uniq))

        # --- feature tables (one row per recording)
        t = time.time()
        X_rich = pd.DataFrame([rich_features(x, d["sr"], kind) for x in d["signals"]])
        X_rich_loc = pd.concat([X_rich, location_onehot(d["location"])], axis=1)
        X_base_loc = pd.concat([X_base, location_onehot(d["location"])], axis=1)
        # augmented copies: 2 per recording, computed once; a fold only ever uses the copies of
        # ITS OWN training clips (aug_parent), so test clips never contribute augmented data
        rng = np.random.default_rng(0)
        aug_rows, aug_parent = [], []
        for i, x in enumerate(d["signals"]):
            for _ in range(2):
                aug_rows.append(rich_features(augment(x, rng, d["sr"]), d["sr"], kind))
                aug_parent.append(i)
        X_aug, aug_parent = pd.DataFrame(aug_rows), np.array(aug_parent)
        # windows: 9 x 3 s per recording, each window keeps its parent's index (-> group, fold)
        w_rows, w_parent = [], []
        for i, x in enumerate(d["signals"]):
            for w in windows(x, d["sr"]):
                w_rows.append(rich_features(w, d["sr"], kind))
                w_parent.append(i)
        Xw, w_parent = pd.DataFrame(w_rows), np.array(w_parent)
        print(f"features computed in {time.time() - t:.0f}s "
              f"({X_rich.shape[1]} rich features, {len(Xw)} windows)", flush=True)

        experiments = [
            ("E0 baseline LR (model.py)", lambda: exp_tabular(d, X_base, "LR")),
            ("E1 LR  base+location", lambda: exp_tabular(d, X_base_loc, "LR")),
            ("E1 LR  base+MFCC", lambda: exp_tabular(d, X_rich, "LR")),
            ("E1 RF  base+MFCC", lambda: exp_tabular(d, X_rich, "RF")),
            ("E1 HGB base+MFCC", lambda: exp_tabular(d, X_rich, "HGB")),
            ("E1 SVM base+MFCC", lambda: exp_tabular(d, X_rich, "SVM")),
            ("E1 RF  base+MFCC+location", lambda: exp_tabular(d, X_rich_loc, "RF")),
            ("E1 SVM base+MFCC+location", lambda: exp_tabular(d, X_rich_loc, "SVM")),
            ("E2 windows LR  base+MFCC", lambda: exp_windows(d, Xw, w_parent, "LR")),
            ("E2 windows RF  base+MFCC", lambda: exp_windows(d, Xw, w_parent, "RF")),
            ("E2 windows SVM base+MFCC", lambda: exp_windows(d, Xw, w_parent, "SVM")),
            ("E4 augment LR  base+MFCC", lambda: exp_tabular_augmented(d, X_rich, X_aug, aug_parent, "LR")),
            ("E4 augment RF  base+MFCC", lambda: exp_tabular_augmented(d, X_rich, X_aug, aug_parent, "RF")),
            ("E4 augment SVM base+MFCC", lambda: exp_tabular_augmented(d, X_rich, X_aug, aug_parent, "SVM")),
        ]
        for name, make in experiments:
            for seed in seeds:
                t = time.time()
                res = run_cv(d, make(), seed)
                record(kind, name, seed, res, time.time() - t)

        # nested model selection (seed 42 only, it is the slowest classical experiment)
        cands = [("base", "LR"), ("rich", "LR"), ("rich", "RF"), ("rich", "SVM"), ("rich", "HGB")]
        for seed in seeds:
            t = time.time()
            fp, chosen = exp_nested_selection(d, X_base, X_rich, cands)
            res = run_cv(d, fp, seed)
            record(kind, "E1 nested selection (inner CV)", seed, res, time.time() - t)
            print("     models chosen per outer fold:", chosen)

        # per-class recall of the recommended model (LR on base+MFCC), seed 42
        res = run_cv(d, exp_tabular(d, X_rich, "LR"), 42)
        print("     per-class recall, E1 LR base+MFCC, seed 42:")
        for c in np.unique(d["y"]):
            m = d["y"] == c
            print(f"       {c:<24} {np.mean(res['pred'][m] == c):.2f}  (n rows {m.sum()})")
        print(f"     accuracy counting each unique clip once: {res['acc_unique']:.3f}")

        if kind == "heart":
            for name, X, m in [("E5 binary N/A  LR base", X_base, "LR"),
                               ("E5 binary N/A  LR base+MFCC", X_rich, "LR"),
                               ("E5 binary N/A  RF base+MFCC", X_rich, "RF")]:
                for seed in seeds:
                    t = time.time()
                    bal, bal_std = run_binary_cv(d, exp_binary(d, X, m), seed)
                    results.append({"kind": kind, "experiment": name, "seed": seed, "acc": np.nan,
                                    "acc_std": np.nan, "bal": bal, "bal_std": bal_std,
                                    "acc_unique": np.nan, "seconds": time.time() - t})
                    print(f"  {name:<34} seed {seed:<5} N/A bal {bal:.3f} +- {bal_std:.3f}")

        if not args.no_neural:
            # LSTM on all seeds; the CNN is slow (~3 min per seed) so only seed 42
            for arch, arch_seeds in [("LSTM", seeds), ("CNN", [42])]:
                for seed in arch_seeds:
                    t = time.time()
                    res = neural_experiment(d, arch, seed)
                    record(kind, f"E3 {arch} log-mel 3s windows +aug", seed, res, time.time() - t)

    # ------------------------------------------------------------ summary table
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(HERE, "results_raw.csv"), index=False)
    s42 = df[df.seed == 42].set_index(["experiment", "kind"])
    allseeds = df.groupby(["experiment", "kind"])[["acc", "bal"]].mean()
    print("\n\nSUMMARY  (seed 42: mean +- std over 5 grouped folds | mean over all seeds)")
    print(f"{'experiment':<34} {'heart acc':>16} {'heart N/A bal':>16} {'lung acc':>16} "
          f"{'lung N/A bal':>16} {'heart 3-seed':>13} {'lung 3-seed':>12}")
    for exp in dict.fromkeys(df.experiment):
        cells = []
        for kind in ["heart", "lung"]:
            for col in ["acc", "bal"]:
                if (exp, kind) in s42.index and not np.isnan(s42.loc[(exp, kind), col]):
                    r = s42.loc[(exp, kind)]
                    cells.append(f"{r[col]:.3f} +- {r[col + '_std']:.3f}")
                else:
                    cells.append("-")
        seeds_cells = [f"{allseeds.loc[(exp, k), 'acc']:.3f}" if (exp, k) in allseeds.index
                       and not np.isnan(allseeds.loc[(exp, k), "acc"]) else "-" for k in ["heart", "lung"]]
        print(f"{exp:<34} " + " ".join(f"{c:>16}" for c in cells)
              + f" {seeds_cells[0]:>13} {seeds_cells[1]:>12}")
    print(f"\ntotal run time {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
