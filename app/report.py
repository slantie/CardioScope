# report.py
# Report: writes the results of the Advisor as text for the suggestion panel.

FRIENDLY_NAMES = {
    "rms": "loudness (RMS)", "log_rms": "loudness (log RMS)", "peak": "peak amplitude",
    "zcr": "zero crossing rate", "dominant_freq": "dominant frequency",
    "spectral_centroid": "spectral centroid (pitch)", "low_energy_ratio": "energy below 200 Hz",
    "high_energy_ratio": "energy above 200 Hz", "spec_entropy": "spectral entropy (noisiness)",
    "flatness": "spectral flatness", "band_20_50": "energy 20-50 Hz", "band_50_100": "energy 50-100 Hz",
    "band_100_200": "energy 100-200 Hz", "band_200_400": "energy 200-400 Hz",
    "band_400_800": "energy 400-800 Hz", "band_800_1600": "energy 800-1600 Hz",
    "period_strength": "rhythm regularity", "rate_per_min": "beat / breath rate",
    "interval_cv": "interval variability", "transient_rate": "short clicks / crackles",
}

# short textbook descriptions shown to the user
DESCRIPTIONS = {
    "Normal": "Normal sound pattern.",
    "Late Diastolic Murmur": "Extra sound late in the relaxation phase of the heart.",
    "Mid Systolic Murmur": "Whooshing sound in the middle of the heart's contraction.",
    "Late Systolic Murmur": "Whooshing sound late in the heart's contraction.",
    "Early Systolic Murmur": "Whooshing sound early in the heart's contraction.",
    "Atrial Fibrillation": "Irregular heart rhythm.",
    "S3": "Extra (third) heart sound shortly after S2.",
    "S4": "Extra (fourth) heart sound just before S1.",
    "Tachycardia": "Fast heart rate (over 100 beats per minute).",
    "AV Block": "Delayed / blocked conduction between atria and ventricles.",
    "Wheezing": "Continuous high-pitched musical sound (narrowed airways).",
    "Fine Crackles": "Short, high-pitched popping sounds.",
    "Coarse Crackles": "Louder, lower-pitched bubbling / popping sounds.",
    "Rhonchi": "Low-pitched snoring-like continuous sound.",
    "Pleural Rub": "Creaking / grating sound, like leather rubbing.",
}


def friendly_name(column):
    """'mfcc3_mean' -> 'sound texture (MFCC 3 average)'"""
    if column.startswith("mfcc"):
        num, stat = column[4:].split("_")
        stat = {"mean": "average", "std": "variation", "dstd": "change"}[stat]
        return f"sound texture (MFCC {num} {stat})"
    return FRIENDLY_NAMES.get(column, column)


class Report:
    """heading() -> (title, colour) and text() -> the lines shown under it."""

    def __init__(self, results):
        self.results = results

    def heading(self):
        levels = [r["level"] for r in self.results]
        if "CHECK" in levels:
            return "Possible abnormal sound", "#c0392b"
        if "LOW RELIABILITY" in levels:
            return "Mixed sound - rough guess", "#e67e22"
        if "UNCERTAIN" in levels:
            return "Not sure - please re-check", "#e67e22"
        return "Looks normal", "#27ae60"

    def text(self):
        lines = []
        for r in self.results:
            lines.append(r["kind"].upper())
            lines += self._verdict(r)
            lines += ["", "Top matches:"]
            lines += [f" {'#' * int(round(p * 6)):<6}{p:>4.0%} {c}" for c, p in r["top3"]]
            lines += ["", "Why:"]
            lines += [f"  - {direction} {friendly_name(col)}" for col, direction in r["reasons"]]
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _verdict(r):
        if r["level"] == "LOW RELIABILITY":
            return [f"Most similar to: {r['best']}",
                    "Heart and lung sounds overlap here, record them separately for a better result."]
        if r["level"] == "UNCERTAIN":
            return [f"Not sure ({r['confidence']:.0%})",
                    "Try re-recording in a quiet room or at another location."]
        if r["level"] == "LIKELY NORMAL":
            return [f"Normal ({r['confidence']:.0%})", "No abnormal pattern found."]
        return [f"{r['best']} ({r['confidence']:.0%})", DESCRIPTIONS.get(r["best"], ""),
                "Suggest getting it checked by a doctor."]
