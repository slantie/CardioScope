# Data quality check: makes sure every recording can be used by the app.

import hashlib

import numpy as np


class QualityCheck:
    def __init__(self, dataset):
        self.dataset = dataset

    def _all_files(self):
        df = self.dataset.table
        mixed = df[df["category"] == "Mixed"]
        return (list(zip(df["folder"], df["file"])) + [("Mix", f) for f in mixed["heart_ref"]] + [("Mix", f) for f in mixed["lung_ref"]])

    def run(self):
        files = self._all_files()
        unreadable, clipped = 0, 0
        rates, lengths, unique = set(), set(), set()
        for folder, name in files:
            try:
                x, sr = self.dataset.read(folder, name)
            except (FileNotFoundError, ValueError):
                unreadable += 1
                continue
            rates.add(sr)
            lengths.add(round(len(x) / sr, 2))
            if np.max(np.abs(x)) >= 0.999:
                clipped += 1
            unique.add(hashlib.md5(x.tobytes()).hexdigest())

        # Required checks: the app only works if all three pass.
        all_readable = unreadable == 0
        one_sample_rate = len(rates) == 1
        one_length = len(lengths) == 1

        readable_text = f"{len(files) - unreadable} of {len(files)} wav files can be read"
        rate_text = f"sample rate: {sorted(rates)} Hz"
        length_text = f"length: {sorted(lengths)} seconds"

        checks = [
            (all_readable, readable_text),
            (one_sample_rate, rate_text),
            (one_length, length_text),
        ]

        # tolerated problems: the app should still run
        df = self.dataset.table
        old_names = (df["category"] == "Lung") & (df["csv_id"] + ".wav" != df["file"])
        has_old_names = old_names.any()
        has_clipped = clipped > 0
        has_duplicates = len(unique) < len(files) - unreadable
        warning_count = sum([has_old_names, has_clipped, has_duplicates])

        lines = []
        for passed, text in checks:
            prefix = "OK    " if passed else "FAIL  "
            lines.append(prefix + text)

        if all(passed for passed, _ in checks):
            summary = "All checks passed"
        else:
            summary = "Some checks failed"
        lines.append("")
        lines.append(f"{summary}, with {warning_count} warnings (handled by the app).")
        return lines
