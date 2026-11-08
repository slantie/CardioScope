# features.py
# Signal maths: filters, spectrum and the numbers (features) we measure on a recording.
# Plain functions: they keep no state, so they don't need a class.

from math import gcd

import numpy as np
import pandas as pd
from scipy import signal as sp
from scipy.fft import dct


# ---------------------------------------------------------------- basic tools
def bandpass(x, sr, low, high, order=4):
    """Butterworth bandpass filter. Heart sounds are mostly 20-200 Hz, lung sounds higher."""
    nyq = sr / 2
    b, a = sp.butter(order, [low / nyq, high / nyq], btype="band")
    return sp.filtfilt(b, a, x)


def resample(x, sr, new_sr):
    """Change the sample rate (e.g. 44100 -> 4000). Returns (signal, new_sr)."""
    if int(sr) == int(new_sr):
        return x, sr
    g = gcd(int(sr), int(new_sr))
    return sp.resample_poly(x, int(new_sr) // g, int(sr) // g).astype(np.float32), new_sr


def get_fft(x, sr):
    """Frequency axis and magnitude (one-sided)."""
    return np.fft.rfftfreq(len(x), 1 / sr), np.abs(np.fft.rfft(x))


def spectrum_similarity(a, b, sr):
    """Correlation of the (log) power spectra. 1.0 = same frequency content."""
    _, pa = sp.welch(a, fs=sr, nperseg=1024)
    _, pb = sp.welch(b, fs=sr, nperseg=1024)
    return float(np.corrcoef(np.log(pa + 1e-20), np.log(pb + 1e-20))[0, 1])


# ---------------------------------------------------------------- basic features
def extract_features(x, sr):
    """Loudness, zero crossings and a few spectrum numbers. Returns a dict."""
    f, psd = sp.welch(x, fs=sr, nperseg=1024)  # smoother than the raw FFT
    total = np.sum(psd)
    return {
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "peak": float(np.max(np.abs(x))),
        "zcr": float(np.mean(np.abs(np.diff(np.sign(x)))) / 2),  # how often the sign changes
        "dominant_freq": float(f[np.argmax(psd)]),
        "spectral_centroid": float(np.sum(f * psd) / total),
        "low_energy_ratio": float(np.sum(psd[f < 200]) / total),    # heart region
        "high_energy_ratio": float(np.sum(psd[f >= 200]) / total),  # lung region
    }


def compute_all_features(df, read):
    """extract_features for every row of the table. `read(folder, file)` returns
    (signal, sr), so any reader works (Dependency Inversion: a test passes a fake one)."""
    feats = [extract_features(*read(r["folder"], r["file"])) for _, r in df.iterrows()]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


# ---------------------------------------------------------------- MFCC ("sound texture")
N_FFT, HOP = 512, 128  # 128 ms frames, 32 ms hop at 4000 Hz


def mel_filterbank(sr, n_fft, n_mels=40, fmin=20, fmax=2000):
    """Triangular filters spaced like human hearing (mel scale)."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return fb


def mfcc_features(x, sr, n_mfcc=20):
    """For each of 20 MFCCs: mean and std over time, and std of the change -> 60 numbers."""
    _, _, Z = sp.stft(x, fs=sr, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                      boundary=None, padded=False)
    log_mel = np.log(mel_filterbank(sr, N_FFT) @ (np.abs(Z) ** 2) + 1e-10).T
    m = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    d = np.diff(m, axis=0)
    f = {}
    for i in range(n_mfcc):
        f[f"mfcc{i}_mean"] = float(m[:, i].mean())
        f[f"mfcc{i}_std"] = float(m[:, i].std())
        f[f"mfcc{i}_dstd"] = float(d[:, i].std())
    return f


# ---------------------------------------------------------------- features for the model
def envelope(x, sr, low, high):
    """Smoothed loudness curve of one band, at 200 samples per second."""
    env = np.abs(sp.hilbert(bandpass(x, sr, low, high)))
    b, a = sp.butter(2, 10 / (sr / 2))
    step = sr // 200
    return sp.filtfilt(b, a, env)[::step], sr / step


def model_features(x, sr, kind):
    """kind = 'heart' or 'lung'. ~20 explainable features + 60 MFCC statistics."""
    f = extract_features(x, sr)
    f["log_rms"] = float(np.log10(f["rms"] + 1e-9))

    # spectrum shape
    freqs, psd = sp.welch(x, fs=sr, nperseg=1024)
    p = psd / psd.sum()
    f["spec_entropy"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    f["flatness"] = float(np.exp(np.mean(np.log(psd + 1e-20))) / np.mean(psd))
    for lo, hi in [(20, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1600)]:
        f[f"band_{lo}_{hi}"] = float(p[(freqs >= lo) & (freqs < hi)].sum())

    # rhythm: autocorrelation of the envelope gives beats (or breaths) per minute
    if kind == "heart":
        env, esr = envelope(x, sr, 20, 200)
        min_lag, max_lag, min_dist = int(esr * 60 / 220), int(esr * 60 / 35), int(esr * 0.25)  # 35-220 bpm
    else:
        env, esr = envelope(x, sr, 100, 1000)
        min_lag, max_lag, min_dist = int(esr * 1.5), int(esr * 8), int(esr * 1.0)  # 7.5-40 breaths/min
    env = env - env.mean()
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
    f["period_strength"] = float(ac[lag])
    f["rate_per_min"] = float(60 * esr / lag)
    peaks, _ = sp.find_peaks(env, distance=min_dist, height=np.percentile(env, 70))
    intervals = np.diff(peaks) / esr
    f["interval_cv"] = float(intervals.std() / intervals.mean()) if len(intervals) > 2 else 1.0

    # crackle-like short clicks
    hp = np.abs(bandpass(x, sr, 300, 1500))
    f["transient_rate"] = float((hp > 6 * np.median(hp)).mean())

    f.update(mfcc_features(x, sr))
    return f
