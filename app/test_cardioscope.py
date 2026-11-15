# test_cardioscope.py
# Simple tests. Run with:  python -m pytest

import os

import numpy as np
import pandas as pd
import pytest

import features
from dataset import Dataset
from filters import RecordingFilter
from recordings import TARGET_RMS, NewRecordings
from report import Report

DATA = Dataset()


def test_metadata_counts():
    counts = DATA.table["category"].value_counts()
    assert counts["Heart"] == 50
    assert counts["Lung"] == 50
    assert counts["Mixed"] == 145


def test_lung_ids_are_fixed():
    # LS.csv uses old codes like M_C_LUA, the real file is M_FC_LUA.wav
    assert "M_FC_LUA.wav" in set(DATA.table["file"])


def test_read_wav_format():
    x, sr = DATA.read("HS", "F_N_RC.wav")
    assert sr == 4000
    assert len(x) == 15 * 4000
    assert np.max(np.abs(x)) <= 1.0


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        DATA.read("HS", "does_not_exist.wav")


def test_features_on_sine_wave():
    # a 100 Hz sine wave should have its dominant frequency near 100 Hz
    sr = 4000
    t = np.arange(sr * 2) / sr
    x = 0.5 * np.sin(2 * np.pi * 100 * t)
    f = features.extract_features(x, sr)
    assert abs(f["dominant_freq"] - 100) < 5
    assert abs(f["rms"] - 0.5 / np.sqrt(2)) < 0.01
    assert f["low_energy_ratio"] > 0.9  # 100 Hz is below 200 Hz


def test_bandpass_removes_high_frequency():
    sr = 4000
    t = np.arange(sr) / sr
    low = np.sin(2 * np.pi * 50 * t)
    high = np.sin(2 * np.pi * 800 * t)
    filtered = features.bandpass(low + high, sr, 20, 200)
    assert np.std(filtered - low) < 0.1  # most of the 800 Hz part is gone


def test_mixed_recordings_have_matched_files():
    row = DATA.table[DATA.table["category"] == "Mixed"].iloc[0]
    h, _ = DATA.read("Mix", row["heart_ref"])
    l, _ = DATA.read("Mix", row["lung_ref"])
    assert len(h) == len(l) == 60000


def test_report_text_has_top_matches():
    results = [{"kind": "lung", "level": "LIKELY NORMAL", "best": "Normal", "confidence": 0.9,
                "top3": [("Normal", 0.9), ("Wheezing", 0.07), ("Rhonchi", 0.03)],
                "reasons": [("zcr", "high")]}]
    report = Report(results)
    assert "Top matches" in report.text()
    assert "high zero crossing rate" in report.text()
    assert report.heading()[0] == "Looks normal"


def test_features_work_with_any_reader():
    # Dependency Inversion: compute_all_features() only needs "a function that returns
    # (signal, sample_rate)", so a fake reader that makes sine waves works too.
    def fake_reader(folder, file_name):
        t = np.arange(4000) / 4000
        return np.sin(2 * np.pi * 300 * t), 4000

    table = pd.DataFrame({"folder": ["X", "Y"], "file": ["a.wav", "b.wav"]})
    result = features.compute_all_features(table, fake_reader)
    assert len(result) == 2
    assert abs(result["dominant_freq"].iloc[0] - 300) < 5


def test_import_any_wav(tmp_path):
    # a loud stereo 44.1 kHz file (like a phone recording) becomes mono 4000 Hz, dataset loudness
    from scipy.io import wavfile
    t = np.arange(44100 * 2) / 44100
    tone = (0.5 * np.sin(2 * np.pi * 100 * t) * 32767).astype(np.int16)
    path = tmp_path / "phone.wav"
    wavfile.write(path, 44100, np.column_stack([tone, tone]))
    x, sr = NewRecordings(str(tmp_path)).read(str(path))
    assert sr == 4000 and x.ndim == 1 and abs(len(x) - 8000) <= 1
    assert abs(np.sqrt(np.mean(x ** 2)) - TARGET_RMS) < 1e-4


def test_save_recording(tmp_path):
    rec = NewRecordings(str(tmp_path))
    name = rec.save(np.zeros(44100, dtype=np.float32), 44100, "Heart")
    assert name.startswith("heart_") and os.path.isfile(tmp_path / name)


def test_synthetic_samples_load():
    for name in ["synthetic_heart_normal.wav", "synthetic_lung_wheeze.wav"]:
        x, sr = NewRecordings().read(os.path.join(os.path.dirname(__file__), "samples", name))
        assert sr == 4000 and len(x) == 15 * 4000


def test_mfcc_features():
    t = np.arange(4000 * 3) / 4000
    f = features.mfcc_features(np.sin(2 * np.pi * 100 * t).astype(np.float32), 4000)
    assert len(f) == 60 and all(np.isfinite(v) for v in f.values())


def test_filter_mixed_sound_type():
    # Open/Closed filters: 'S3' must also find mixed recordings like 'S3 + Normal'
    flt = RecordingFilter(DATA.table)
    chosen = {"Category": "All", "Sound type": "S3", "Location": "All", "Gender": "All"}
    found = flt.apply(chosen)
    assert len(found) > 0 and set(found["category"]) >= {"Heart", "Mixed"}
    assert RecordingFilter.describe(chosen) == "Sound type: S3"
