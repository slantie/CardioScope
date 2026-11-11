# model.py
# DecisionModel: a small, explainable model for one kind of sound (heart or lung).
#
#   logistic regression on ~20 explainable features + 60 MFCC statistics.
#   MFCCs raised grouped-CV accuracy from 41% to 65% (heart) and 64% to 97% (lung).
#   We also tried an LSTM and a CNN: both were worse, too little data (see experiments/).
#
# Training data: standalone HS/LS files + the matched H/L files in dataset/Mix
# (195 heart + 195 lung). Many are exact copies of each other, so we use GROUPED 5-fold
# cross validation: all copies of the same audio stay in the same fold.
# Manikin data only: a learning tool, NOT a medical diagnosis.

import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features

N_FOLDS = 5


def new_pipeline():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000))


def training_table(dataset, kind):
    """Features (X), labels (y), 'folder/file' keys and copy groups for one kind."""
    table = dataset.table
    rows = [(r.folder, r.file, r.sound_type)
            for r in table[table["category"] == kind.capitalize()].itertuples()]
    name = kind.capitalize()  # 'Heart' or 'Lung'
    for _, r in dataset.csv("Mix.csv").iterrows():
        rows.append(("Mix", r[f"{name} Sound ID"] + ".wav", r[f"{name} Sound Type"]))

    X, groups = [], []
    for folder, file, _ in rows:
        x, sr = dataset.read(folder, file)
        X.append(features.model_features(x, sr, kind))
        groups.append(hashlib.md5(x.tobytes()).hexdigest())  # identical audio -> same group
    y = np.array([t for _, _, t in rows])
    keys = [f"{folder}/{file}" for folder, file, _ in rows]
    return pd.DataFrame(X), y, keys, np.array(groups)


class DecisionModel:
    """One full model + 5 fold models for heart or lung.

    If the user picks a recording that was used for training, we answer with the fold
    model that never saw it, so the suggestion is always honest (out-of-fold)."""

    def __init__(self, dataset, kind):
        self.kind = kind
        X, self.y, keys, groups = training_table(dataset, kind)
        self.columns = list(X.columns)
        self.full = new_pipeline().fit(X, self.y)

        self.fold_models, self.fold_of = [], {}
        self.oof_proba = np.zeros((len(self.y), len(self.full.classes_)))
        folds = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=42)
        for i, (train, test) in enumerate(folds.split(X, self.y, groups)):
            m = new_pipeline().fit(X.iloc[train], self.y[train])
            self.fold_models.append(m)
            self.oof_proba[test] = m.predict_proba(X.iloc[test])
            for j in test:
                self.fold_of[keys[j]] = i

    def predict(self, feats, key=None):
        """Returns the top 3 (sound type, probability) and the 3 features that pushed
        the answer most, as (column, 'high' or 'low')."""
        model = self.fold_models[self.fold_of[key]] if key in self.fold_of else self.full
        X = pd.DataFrame([feats])[self.columns]
        proba = model.predict_proba(X)[0]
        order = np.argsort(proba)[::-1]
        top3 = [(model.classes_[i], float(proba[i])) for i in order[:3]]

        # reason = feature value (scaled) x its weight for the top class
        z = model.named_steps["standardscaler"].transform(X)[0]
        weights = model.named_steps["logisticregression"].coef_[order[0]]
        reasons = [(self.columns[k], "high" if z[k] > 0 else "low")
                   for k in np.argsort(weights * z)[::-1][:3]]
        return top3, reasons

    def evaluation(self, threshold):
        """Grouped-CV scores, using the out-of-fold predictions."""
        pred = self.full.classes_[self.oof_proba.argmax(1)]
        keep = self.oof_proba.max(1) >= threshold
        normal = lambda a: np.where(np.asarray(a) == "Normal", "Normal", "Abnormal")
        return {
            "recordings": len(self.y),
            "type_accuracy": accuracy_score(self.y, pred),
            "normal_vs_abnormal_bal_acc": balanced_accuracy_score(normal(self.y), normal(pred)),
            "coverage_when_confident": keep.mean(),
            "accuracy_when_confident": accuracy_score(self.y[keep], pred[keep]),
        }
