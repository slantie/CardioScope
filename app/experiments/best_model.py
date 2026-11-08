# best_model.py
# Recommended model from run_experiments.py ("E1 LR base+MFCC"):
#   the ~20 hand-crafted model.py features + 60 MFCC statistics (mean, std and
#   delta-std of 20 MFCCs over the whole 15 s clip) -> StandardScaler -> LogisticRegression(C=1).
# Same model family as model.py, so the "coef x value" explanation still works.
# Honest grouped-CV numbers: see RESULTS.md. NOT integrated into the app yet.
#
# Usage (from the app folder):
#     from experiments import best_model
#     best_model.train("heart")
#     best_model.predict_proba(signal, sr, "heart")   -> {class: probability}

import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import load_rows, rich_features  # noqa: E402

_models = {}  # kind -> (fitted pipeline, column order)


def train(kind):
    """Fit on all 195 heart (or lung) training recordings. kind = 'heart' or 'lung'."""
    d = load_rows(kind)
    X = pd.DataFrame([rich_features(x, d["sr"], kind) for x in d["signals"]])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000))
    model.fit(X, d["y"])
    _models[kind] = (model, list(X.columns))
    return model


def predict_proba(signal, sr, kind):
    """Class probabilities for one recording (any length >= ~3 s), sorted high to low.
    NOTE: for recordings that are in the training set this is NOT out-of-fold."""
    if kind not in _models:
        train(kind)
    model, cols = _models[kind]
    X = pd.DataFrame([rich_features(signal, sr, kind)])[cols]
    p = model.predict_proba(X)[0]
    return dict(sorted(((str(c), float(v)) for c, v in zip(model.classes_, p)),
                       key=lambda t: -t[1]))


if __name__ == "__main__":
    from dataset import Dataset
    x, sr = Dataset().read("HS", "F_N_RC.wav")
    print(predict_proba(x, sr, "heart"))
