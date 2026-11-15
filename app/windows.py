# windows.py
# The extra windows: compare two recordings, matched heart-lung-mixed, data quality.

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import features
from plots import draw_signal


class CompareWindow:
    """Two recordings from the dataset, one above the other, with a small feature table."""

    def __init__(self, root, dataset, start_file=None):
        self.dataset = dataset
        self.win = tk.Toplevel(root)
        self.win.title("Compare Two Recordings")
        self.win.geometry("1000x680")
        table = dataset.table
        names = [f"{r['file']} - {r['sound_type']}" for _, r in table.iterrows()]

        top = tk.Frame(self.win, padx=8, pady=8)
        top.pack(fill=tk.X)
        self.boxes = []
        for i, label in enumerate(["Recording A", "Recording B"]):
            tk.Label(top, text=label).grid(row=i, column=0, sticky="w")
            box = ttk.Combobox(top, values=names, width=45, state="readonly")
            box.grid(row=i, column=1, padx=5)
            self.boxes.append(box)
        if start_file in set(table["file"]):  # start with the selected recording as A
            self.boxes[0].current(int(table.index[table["file"] == start_file][0]))

        tk.Label(top, text="Plot").grid(row=0, column=2, padx=(15, 0))
        self.kind = ttk.Combobox(top, values=["Waveform", "FFT", "Spectrogram"], width=12,
                                 state="readonly")
        self.kind.current(0)
        self.kind.grid(row=0, column=3)
        tk.Button(top, text="Compare", command=self.compare).grid(row=1, column=2, columnspan=2,
                                                                 padx=(15, 0), sticky="we")
        self.fig = Figure(figsize=(9, 4.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.table = tk.Label(self.win, font=("Courier", 11), justify="left", anchor="w")
        self.table.pack(fill=tk.X, padx=8, pady=8)

    def compare(self):
        if min(box.current() for box in self.boxes) < 0:
            messagebox.showinfo("Compare", "Please choose both recordings.", parent=self.win)
            return
        self.fig.clear()
        feats, first = [], None
        for i, box in enumerate(self.boxes):
            row = self.dataset.table.iloc[box.current()]
            x, sr = self.dataset.read(row["folder"], row["file"])
            feats.append(features.extract_features(x, sr))
            ax = self.fig.add_subplot(2, 1, i + 1, sharex=first)
            first = first or ax
            draw_signal(ax, x, sr, self.kind.get(), "tab:blue" if i == 0 else "tab:orange")
            ax.set_title(f"{'AB'[i]}: {row['file']} ({row['sound_type']})", fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()

        lines = [f"{'feature':<20}{'A':>12}{'B':>12}"]
        for k in ["rms", "zcr", "dominant_freq", "spectral_centroid", "low_energy_ratio"]:
            lines.append(f"{k:<20}{feats[0][k]:>12.4f}{feats[1][k]:>12.4f}")
        self.table.config(text="\n".join(lines))


class MatchedWindow:
    """A mixed recording M#### with its matched heart-only H#### and lung-only L#### files."""

    def __init__(self, root, dataset, row):
        h, sr = dataset.read("Mix", row["heart_ref"])
        l, _ = dataset.read("Mix", row["lung_ref"])
        m, _ = dataset.read("Mix", row["file"])

        win = tk.Toplevel(root)
        win.title(f"Matched recordings - {row['file']}")
        fig = Figure(figsize=(9, 6), dpi=100)
        parts = [(h, f"Heart only: {row['heart_ref']}", "tab:red"),
                 (l, f"Lung only: {row['lung_ref']}", "tab:green"),
                 (m, f"Mixed: {row['file']}", "tab:blue")]
        first = None
        for i, (x, title, color) in enumerate(parts):
            ax = fig.add_subplot(3, 1, i + 1, sharex=first)
            first = first or ax
            ax.plot(np.arange(len(x)) / sr, x, linewidth=0.5, color=color)
            ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=win).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # waveform correlation: same samples? spectrum similarity: same frequencies?
        corr = [np.corrcoef(m, part)[0, 1] for part in (h, l)]
        spec = [features.spectrum_similarity(m, part, sr) for part in (h, l)]
        tk.Label(win, font=("Courier", 11), justify="left",
                 text=f"                      with heart   with lung\n"
                      f"Waveform correlation      {corr[0]:5.2f}       {corr[1]:5.2f}\n"
                      f"Spectrum similarity       {spec[0]:5.2f}       {spec[1]:5.2f}\n\n"
                      "Low waveform correlation but high spectrum similarity means the mixture\n"
                      "was recorded separately, not made by adding the two files together."
                 ).pack(pady=8)


class QualityWindow:
    def __init__(self, root, lines):
        win = tk.Toplevel(root)
        win.title("Data Quality Check")
        tk.Label(win, text="\n".join(lines), font=("Courier", 12), justify="left").pack(padx=20, pady=15)
