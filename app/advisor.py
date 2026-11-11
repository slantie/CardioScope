# advisor.py
# Advisor: picks the right model for a recording and decides how sure we are.

import features
from model import N_FOLDS, DecisionModel

CONFIDENCE_THRESHOLD = 0.5  # below this we say "not sure" instead of guessing


class Advisor:
    """Holds the heart and lung models and turns a recording into results.

    Each result is a dict: kind, level, best, confidence, top3, reasons.
    level = 'LIKELY NORMAL', 'CHECK', 'UNCERTAIN' or 'LOW RELIABILITY' (mixed)."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.models = {}

    def train(self):
        """About 4 seconds. The app runs this in a background thread."""
        for kind in ("heart", "lung"):
            self.models[kind] = DecisionModel(self.dataset, kind)

    def is_ready(self):
        return len(self.models) == 2

    def decide(self, x, sr, category, key=None):
        """category: 'Heart', 'Lung' or 'Mixed'. key: 'folder/file' for dataset recordings."""
        if category == "Heart":
            return [self._judge("heart", x, sr, key)]
        if category == "Lung":
            return [self._judge("lung", x, sr, key)]
        # mixed: rough split with filters, then judge each part. Tested on all 145
        # mixtures: only 15% (heart) and 34% (lung) right, so never give a verdict here
        results = [self._judge("heart", features.bandpass(x, sr, 20, 200), sr),
                   self._judge("lung", features.bandpass(x, sr, 200, 1000), sr)]
        for r in results:
            r["level"] = "LOW RELIABILITY"
        return results

    def _judge(self, kind, x, sr, key=None):
        top3, reasons = self.models[kind].predict(features.model_features(x, sr, kind), key)
        best, confidence = top3[0]
        if confidence < CONFIDENCE_THRESHOLD:
            level = "UNCERTAIN"
        elif best == "Normal":
            level = "LIKELY NORMAL"
        else:
            level = "CHECK"
        return {"kind": kind, "level": level, "best": best, "confidence": confidence,
                "top3": top3, "reasons": reasons}


if __name__ == "__main__":
    # print the evaluation of both models
    from dataset import Dataset

    advisor = Advisor(Dataset())
    advisor.train()
    for kind, model in advisor.models.items():
        print(f"\n{kind.upper()} model ({N_FOLDS}-fold grouped CV)")
        for k, v in model.evaluation(CONFIDENCE_THRESHOLD).items():
            print(f"  {k:<30} {v:.2f}" if isinstance(v, float) else f"  {k:<30} {v}")
