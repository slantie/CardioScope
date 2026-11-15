# app.py
# CardioScope - explore heart and lung sounds (HLS-CMDS dataset) and get a suggestion.
# Run:  python app.py

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import features
from advisor import Advisor
from dataset import Dataset
from filters import FILTERS, RecordingFilter
from player import AudioPlayer
from plots import Plotter
from quality import QualityCheck
from recordings import NewRecordings
from report import Report
from windows import CompareWindow, MatchedWindow, QualityWindow

RECORD_SECONDS = 15


class SuggestionPanel:
    """The coloured heading and the text on the right side of the window."""

    def __init__(self, parent, compact):
        tk.Label(parent, text="Suggestion", font=("Arial", 12, "bold")).pack(anchor="w")
        self.title = tk.Label(parent, text="Select a recording", bg="gray", fg="white",
                              font=("Arial", 15, "bold"), anchor="w", padx=10, pady=8,
                              width=22 if compact else 26)
        self.title.pack(fill=tk.X, pady=(4, 0))
        self.text = tk.Text(parent, width=34 if compact else 42, font=("Courier", 11),
                            wrap="word", relief="flat", padx=8, pady=8, state="disabled")
        self.text.pack(fill=tk.BOTH, expand=True)

    def show(self, title, colour, text):
        self.title.config(text=title, bg=colour)
        self.text.config(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", text)
        self.text.config(state="disabled")


class CardioScopeApp:
    """Builds the main window and connects the parts. The parts are passed in
    (Dependency Inversion), so each one can be tested or replaced on its own."""

    def __init__(self, root, dataset, advisor, recordings, player):
        self.root, self.dataset, self.advisor = root, dataset, advisor
        self.recordings, self.player = recordings, player
        self.filter = RecordingFilter(dataset.table)
        self.filtered = dataset.table
        self.feat_table = None             # computed the first time a dataset chart needs it
        self.signal = self.sr = self.current = None

        root.title("CardioScope - Heart & Lung Sound Explorer")
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        self.compact = screen_w < 1500    # a projector may be only 1280x720
        root.geometry(f"{min(1450, screen_w - 20)}x{min(760, screen_h - 80)}+10+10")

        self.build_heading()
        self.build_left_panel()
        self.suggestion = SuggestionPanel(self.side_frame(tk.RIGHT, tk.Y), self.compact)
        self.build_middle_panel()
        self.update_list()

        # train the models in the background so the window opens at once
        threading.Thread(target=advisor.train, daemon=True).start()

    # ================================================================ building the window
    def side_frame(self, side, fill, expand=False):
        frame = tk.Frame(self.root, padx=8, pady=8)
        frame.pack(side=side, fill=fill, expand=expand)
        return frame

    def build_heading(self):
        head = tk.Frame(self.root, padx=10, pady=6)
        head.pack(side=tk.TOP, fill=tk.X)
        tk.Label(head, text="CardioScope", font=("Arial", 18, "bold")).pack(side=tk.LEFT)
        tk.Label(head, text=f"   Heart & Lung Sound Explorer  -  HLS-CMDS dataset "
                            f"({len(self.dataset.table)} recordings)",
                 font=("Arial", 12), fg="gray55").pack(side=tk.LEFT, pady=(5, 0))
        ttk.Separator(self.root, orient="horizontal").pack(side=tk.TOP, fill=tk.X)

    def build_left_panel(self):
        left = self.side_frame(tk.LEFT, tk.Y)
        tk.Label(left, text="Filters", font=("Arial", 12, "bold")).pack(anchor="w")
        grid = tk.Frame(left)
        grid.pack(fill=tk.X)
        grid.columnconfigure(1, weight=1)
        self.filter_vars, self.filter_boxes = {}, {}
        for row, (label, column, _) in enumerate(FILTERS):  # one dropdown per filter
            self.filter_vars[label] = tk.StringVar(value="All")
            tk.Label(grid, text=label).grid(row=row, column=0, sticky="w", pady=2)
            box = ttk.Combobox(grid, textvariable=self.filter_vars[label], state="readonly")
            box.grid(row=row, column=1, sticky="we", pady=2)
            box.bind("<<ComboboxSelected>>", lambda e, lb=label: self.on_filter_change(lb))
            self.filter_boxes[label] = box
        self.search_var = tk.StringVar()
        tk.Label(grid, text="Search").grid(row=len(FILTERS), column=0, sticky="w", pady=2)
        search = tk.Entry(grid, textvariable=self.search_var)
        search.grid(row=len(FILTERS), column=1, sticky="we", pady=2)
        search.bind("<KeyRelease>", lambda e: self.update_list())
        tk.Button(left, text="Reset Filters", command=self.reset_filters).pack(fill=tk.X, pady=(4, 0))
        self.update_filter_options()

        tk.Label(left, text="Recordings").pack(anchor="w", pady=(8, 0))
        list_frame = tk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(list_frame, width=30 if self.compact else 38,
                                  height=4 if self.compact else 8)
        scroll = tk.Scrollbar(list_frame, command=self.listbox.yview)
        self.listbox.config(yscrollcommand=scroll.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.count_label = tk.Label(left)
        self.count_label.pack(anchor="w")

        tk.Label(left, text="Dataset EDA").pack(anchor="w", pady=(10, 0))
        for text, command in [("Class Distribution", self.show_distribution),
                              ("Feature Comparison", self.show_feature_boxplot),
                              ("Compare Two Recordings", self.open_compare),
                              ("Data Quality Check", self.show_data_quality),
                              ("Export Table (CSV)", self.export_csv)]:
            tk.Button(left, text=text, command=command).pack(fill=tk.X)

        tk.Label(left, text="Check a New Recording").pack(anchor="w", pady=(10, 0))
        row = tk.Frame(left)
        row.pack(fill=tk.X)
        tk.Label(row, text="It is a").pack(side=tk.LEFT)
        self.new_type = ttk.Combobox(row, values=["Heart", "Lung", "Mixed"], state="readonly", width=8)
        self.new_type.set("Heart")
        self.new_type.pack(side=tk.LEFT, padx=4)
        tk.Label(row, text="sound").pack(side=tk.LEFT)
        tk.Button(left, text="Import WAV File...", command=self.import_wav).pack(fill=tk.X)
        self.record_button = tk.Button(left, text="Record 15 s (microphone)", command=self.record)
        self.record_button.pack(fill=tk.X)

    def build_middle_panel(self):
        middle = self.side_frame(tk.RIGHT, tk.BOTH, expand=True)
        self.info_label = tk.Label(middle, text="Select a recording from the list",
                                   font=("Arial", 12, "bold"), anchor="w", justify="left")
        self.info_label.pack(fill=tk.X)

        buttons = tk.Frame(middle)
        buttons.pack(fill=tk.X, pady=5)
        fig = Figure(figsize=(6, 3.4) if self.compact else (8, 5), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=middle)
        self.plotter = Plotter(fig, canvas)
        for text, draw in [("Waveform", self.plotter.waveform), ("FFT", self.plotter.fft),
                           ("Spectrogram", self.plotter.spectrogram), ("Bands", self.plotter.bands)]:
            tk.Button(buttons, text=text, command=lambda d=draw: self.plot(d)).pack(side=tk.LEFT)
        tk.Button(buttons, text="Matched H-L-M", command=self.open_matched).pack(side=tk.LEFT)
        audio_row = buttons
        if self.compact:  # small screen: Play/Stop on their own row so nothing is cut off
            audio_row = tk.Frame(middle)
            audio_row.pack(fill=tk.X)
        tk.Button(audio_row, text="▶ Play", command=self.play).pack(
            side=tk.LEFT, padx=(0 if self.compact else 15, 0))
        tk.Button(audio_row, text="■ Stop", command=self.player.stop).pack(side=tk.LEFT)

        NavigationToolbar2Tk(canvas, middle).update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.feat_label = tk.Label(middle, anchor="w", justify="left", font=("Courier", 11))
        self.feat_label.pack(fill=tk.X)

    # ================================================================ filters and list
    def chosen(self):
        return {label: var.get() for label, var in self.filter_vars.items()}

    def update_filter_options(self):
        category = self.filter_vars["Category"].get()
        for label, column, _ in FILTERS:
            values = self.filter.options(label, column, category)
            self.filter_boxes[label].config(values=values)
            if self.filter_vars[label].get() not in values:  # not in this category any more
                self.filter_vars[label].set("All")

    def on_filter_change(self, label):
        if label == "Category":
            self.update_filter_options()
        self.update_list()

    def reset_filters(self):
        for var in self.filter_vars.values():
            var.set("All")
        self.search_var.set("")
        self.update_filter_options()
        self.update_list()

    def update_list(self):
        self.filtered = self.filter.apply(self.chosen(), self.search_var.get())
        self.listbox.delete(0, tk.END)
        for _, r in self.filtered.iterrows():
            self.listbox.insert(tk.END, f"{r['file']:<15} {r['sound_type']}")
        self.count_label.config(text=f"{len(self.filtered)} of {len(self.dataset.table)} recordings")

    # ================================================================ one recording
    def on_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
        row = self.filtered.iloc[selection[0]]
        try:
            self.signal, self.sr = self.dataset.read(row["folder"], row["file"])
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("CardioScope", f"Could not read this recording:\n{e}")
            return
        self.show_recording(row)

    def show_recording(self, row):
        """Info line, features, waveform and suggestion for self.signal."""
        self.current = row
        self.info_label.config(text=f"{row['file']}  |  {row['category']}  |  {row['sound_type']}  |  "
                                    f"{row['location']}  |  {row['gender']}")
        f = features.extract_features(self.signal, self.sr)
        self.feat_label.config(
            text=f"RMS: {f['rms']:.4f}   Peak: {f['peak']:.4f}   ZCR: {f['zcr']:.4f}   "
                 f"Dominant freq: {f['dominant_freq']:.1f} Hz\n"
                 f"Centroid: {f['spectral_centroid']:.1f} Hz   Low/High energy: "
                 f"{f['low_energy_ratio']:.2f}/{f['high_energy_ratio']:.2f}")
        self.plotter.waveform(self.signal, self.sr)
        self.update_suggestion()

    def need_signal(self):
        if self.signal is None:
            messagebox.showinfo("CardioScope", "Please select a recording first.")
        return self.signal is not None

    def plot(self, draw):
        if self.need_signal():
            draw(self.signal, self.sr)

    def update_suggestion(self):
        if not self.advisor.is_ready():  # only in the first seconds after start
            self.suggestion.show("Loading model...", "gray", "Please wait a moment.")
            self.root.after(300, self.update_suggestion)
            return
        row = self.current
        key = None if row["folder"] == "new" else f"{row['folder']}/{row['file']}" # pyright: ignore[reportOptionalSubscript]
        report = Report(self.advisor.decide(self.signal, self.sr, row["category"], key)) # pyright: ignore[reportOptionalSubscript]
        self.suggestion.show(*report.heading(), report.text())

    # ================================================================ audio
    def play(self):
        if not self.need_signal():
            return
        try:
            self.player.play(self.signal, self.sr)
        except Exception as e:  # no sounddevice or no audio device
            messagebox.showwarning("CardioScope", f"Could not play audio:\n{e}")

    # ================================================================ new recordings
    def new_row(self, name, source):
        return {"folder": "new", "file": name, "category": self.new_type.get(),
                "sound_type": "unknown", "location": source, "gender": "-"}

    def import_wav(self):
        path = filedialog.askopenfilename(title="Choose a wav recording",
                                          initialdir=self.recordings.folder,
                                          filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        try:
            self.signal, self.sr = self.recordings.read(path)
        except ValueError as e:
            messagebox.showerror("CardioScope", str(e))
            return
        self.listbox.selection_clear(0, tk.END)
        self.show_recording(self.new_row(path.split("/")[-1], "imported"))

    def record(self):
        try:
            self.player.start_recording(RECORD_SECONDS)
        except Exception as e:  # no sounddevice, no microphone or no permission
            messagebox.showwarning("CardioScope", f"Could not record:\n{e}")
            return
        self.record_button.config(text=f"Recording... ({RECORD_SECONDS} s)", state="disabled")
        self.root.after(RECORD_SECONDS * 1000 + 300, self.finish_recording)

    def finish_recording(self):
        audio, mic_sr = self.player.finish_recording()
        self.record_button.config(text="Record 15 s (microphone)", state="normal")
        try:
            name = self.recordings.save(audio, mic_sr, self.new_type.get())
        except OSError as e:
            messagebox.showwarning("CardioScope", f"Could not save the recording:\n{e}")
            name = "microphone recording"
        self.signal, self.sr = self.recordings.prepare(audio, mic_sr)
        self.listbox.selection_clear(0, tk.END)
        self.show_recording(self.new_row(name, "recorded"))

    # ================================================================ extra windows
    def open_compare(self):
        start = self.current["file"] if self.current is not None else None
        CompareWindow(self.root, self.dataset, start)

    def open_matched(self):
        if not self.need_signal():
            return
        if self.current["category"] != "Mixed" or self.current["folder"] == "new": # pyright: ignore[reportOptionalSubscript]
            messagebox.showinfo("CardioScope", "Please select a Mixed recording (M####).")
            return
        MatchedWindow(self.root, self.dataset, self.current)

    def show_data_quality(self):
        self.busy(True)
        report = QualityCheck(self.dataset).run()
        self.busy(False)
        QualityWindow(self.root, report)

    # ================================================================ dataset charts
    def busy(self, on):
        self.root.config(cursor="watch" if on else "")
        self.root.update()

    def feature_table(self):
        """The filtered recordings with their features (computed once, ~1 s)."""
        if self.feat_table is None:
            self.busy(True)
            self.feat_table = features.compute_all_features(self.dataset.table, self.dataset.read)
            self.busy(False)
        return self.feat_table.merge(self.filtered[["folder", "file"]], on=["folder", "file"])

    def show_distribution(self):
        if len(self.filtered) == 0:
            messagebox.showinfo("CardioScope", "No recordings match the filters.")
            return
        by_category = self.filter_vars["Category"].get() == "All"
        self.plotter.distribution(self.filtered, by_category, RecordingFilter.describe(self.chosen()))
        self.info_label.config(text="Dataset: class distribution")

    def show_feature_boxplot(self):
        df = self.feature_table()
        if len(df) == 0:
            messagebox.showinfo("CardioScope", "No recordings match the filters.")
            return
        by_category = self.filter_vars["Category"].get() in ("All", "Mixed")
        self.plotter.boxplot(df, by_category, RecordingFilter.describe(self.chosen()))
        self.info_label.config(text="Dataset: feature comparison")

    def export_csv(self):
        """Save the filtered recordings + their features. (Plots: use the toolbar save icon.)"""
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="cardioscope_export.csv",
                                            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        table = self.feature_table().drop(columns=["heart_ref", "lung_ref"], errors="ignore")
        try:
            table.to_csv(path, index=False)
        except OSError as e:
            messagebox.showerror("CardioScope", f"Could not save the file:\n{e}")
            return
        messagebox.showinfo("CardioScope", f"Saved {len(table)} recordings to\n{path}")


def main():
    root = tk.Tk()
    try:
        dataset = Dataset()
    except FileNotFoundError as e:
        messagebox.showerror("CardioScope", str(e))
        root.destroy()
        return
    CardioScopeApp(root, dataset, Advisor(dataset), NewRecordings(), AudioPlayer())
    root.mainloop()


if __name__ == "__main__":
    main()
