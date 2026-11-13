# plots.py
# Plotter: draws every chart of the main window on one matplotlib figure.

import numpy as np
import pandas as pd

import features

# short codes (same as in the file names), used when a chart has many groups
SHORT_NAMES = {
    "Normal": "N", "Late Diastolic Murmur": "LDM", "Mid Systolic Murmur": "MSM",
    "Late Systolic Murmur": "LSM", "Early Systolic Murmur": "ESM", "Atrial Fibrillation": "AF",
    "Tachycardia": "T", "AV Block": "AVB", "S3": "S3", "S4": "S4", "Wheezing": "W",
    "Fine Crackles": "FC", "Coarse Crackles": "CC", "Rhonchi": "R", "Pleural Rub": "PR",
}


def draw_signal(ax, x, sr, kind, color="tab:blue"):
    """One recording on one axis as a 'Waveform', 'FFT' or 'Spectrogram'."""
    if kind == "Waveform":
        ax.plot(np.arange(len(x)) / sr, x, linewidth=0.6, color=color)
    elif kind == "FFT":
        freqs, mag = features.get_fft(x, sr)
        ax.plot(freqs, mag, linewidth=0.6, color=color)
        ax.set_xlim(0, 1000)  # most of the energy is below 1000 Hz
    else:
        ax.specgram(x, NFFT=256, Fs=sr, noverlap=128, cmap="magma")
        ax.set_ylim(0, 1000)


class Plotter:
    def __init__(self, fig, canvas):
        self.fig = fig
        self.canvas = canvas

    def _single(self, x, sr, kind, title, xlabel, ylabel, color="tab:blue"):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        draw_signal(ax, x, sr, kind, color)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if kind != "Spectrogram":
            ax.grid(alpha=0.3)
        self.canvas.draw()

    # ---- one recording
    def waveform(self, x, sr):
        self._single(x, sr, "Waveform", "Waveform", "Time (s)", "Amplitude")

    def fft(self, x, sr):
        self._single(x, sr, "FFT", "Frequency Spectrum (FFT)", "Frequency (Hz)", "Magnitude", "tab:red")

    def spectrogram(self, x, sr):
        self._single(x, sr, "Spectrogram", "Spectrogram", "Time (s)", "Frequency (Hz)")

    def bands(self, x, sr):
        """Rough split into heart and lung parts with filters (not real source separation)."""
        t = np.arange(len(x)) / sr
        parts = [(x, "Original", "tab:blue"),
                 (features.bandpass(x, sr, 20, 200), "Heart band (20-200 Hz)", "tab:red"),
                 (features.bandpass(x, sr, 200, 1000), "Lung band (200-1000 Hz)", "tab:green")]
        self.fig.clear()
        first = None
        for i, (y, title, color) in enumerate(parts):
            ax = self.fig.add_subplot(3, 1, i + 1, sharex=first)
            first = first or ax
            ax.plot(t, y, linewidth=0.5, color=color)
            ax.set_title(title)
        ax.set_xlabel("Time (s)")
        self.fig.tight_layout()
        self.canvas.draw()

    # ---- the whole (filtered) dataset
    def distribution(self, df, by_category, subtitle):
        """Recordings per category (or per sound type) and per location."""
        self.fig.clear()
        ax1 = self.fig.add_subplot(121)
        if by_category:
            counts = df["category"].value_counts()
            ax1.bar(counts.index, counts.values, color="tab:purple")
            ax1.set_title("Recordings per category")
        else:
            # a mixed recording counts once for its heart part and once for its lung part
            types = [t for st in df["sound_type"] for t in st.split(" + ")]
            counts = pd.Series(types).value_counts().sort_values()
            ax1.barh(counts.index, counts.values, color="tab:red")
            ax1.set_title("Recordings per sound type")
            ax1.tick_params(axis="y", labelsize=8)
        ax2 = self.fig.add_subplot(122)
        loc = df["location"].value_counts().sort_values()
        ax2.barh(loc.index, loc.values, color="tab:green")
        ax2.set_title("Recordings per location")
        self._finish(f"{len(df)} recordings  ({subtitle})")

    def boxplot(self, df, by_category, subtitle):
        """3 features per group: categories, or sound types inside one category."""
        group = df["category"] if by_category else df["sound_type"]
        names = sorted(group.unique())
        many = len(names) > 3  # many groups -> stack the 3 plots for more room
        labels = [SHORT_NAMES.get(n, n) for n in names] if many else names
        self.fig.clear()
        for i, col in enumerate(["spectral_centroid", "dominant_freq", "low_energy_ratio"]):
            ax = self.fig.add_subplot(3, 1, i + 1) if many else self.fig.add_subplot(1, 3, i + 1)
            ax.boxplot([df[group == n][col] for n in names], tick_labels=labels)
            ax.grid(alpha=0.3)
            if many:
                ax.set_ylabel(col.replace("_", " "), fontsize=8)
                ax.tick_params(labelsize=8)
            else:
                ax.set_title(col.replace("_", " "))
        self._finish(f"{len(df)} recordings  ({subtitle})")

    def _finish(self, title):
        self.fig.suptitle(title, fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()
