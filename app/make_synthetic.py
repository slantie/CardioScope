# make_synthetic.py
# Makes 2 fake (synthetic) recordings to test "Check a New Recording" in the app.
# They are simple sound models, NOT real patients.
# Run:  python make_synthetic.py   -> samples/*.wav  (4000 Hz, 15 s, 16-bit mono)

import os

import numpy as np
from scipy.io import wavfile

import features

SR = 4000
SECONDS = 15
t = np.arange(SR * SECONDS) / SR
rng = np.random.default_rng(0)


def burst(centre, freq, width):
    """A short decaying tone = one heart sound (S1 or S2)."""
    return np.exp(-((t - centre) / width) ** 2) * np.sin(2 * np.pi * freq * (t - centre))


def save(name, x):
    x = x / np.sqrt(np.mean(x ** 2)) * 0.003  # same quiet level as the dataset (RMS)
    os.makedirs("samples", exist_ok=True)
    wavfile.write(os.path.join("samples", name), SR, (x * 32767).astype(np.int16))
    print("wrote samples/" + name)


# 1) normal heart: about 72 beats per minute, S1 then S2 0.3 s later.
#    Real beats are not perfectly regular, so each beat gets a little random jitter.
heart = np.zeros_like(t)
start = 0.2
while start < SECONDS:
    heart += burst(start, 60, 0.012) + 0.5 * burst(start, 120, 0.008)            # S1
    heart += 0.7 * (burst(start + 0.30, 80, 0.010) + 0.5 * burst(start + 0.30, 150, 0.006))  # S2
    start += (60 / 72) * rng.uniform(0.8, 1.2)
heart += 0.05 * features.bandpass(rng.standard_normal(len(t)), SR, 20, 400)  # body noise
save("synthetic_heart_normal.wav", heart)

# 2) lung with wheeze: breathing noise (about 15 breaths/min, uneven) and a
#    ~180 Hz whistle (the typical wheeze pitch in the dataset) while breathing out
breath_noise = features.bandpass(rng.standard_normal(len(t)), SR, 80, 500)
envelope = np.zeros_like(t)
wheeze_env = np.zeros_like(t)
start = 0.0
while start < SECONDS:
    length = 4 * rng.uniform(0.8, 1.2)
    phase = np.clip((t - start) / length, 0, 1) * ((t >= start) & (t < start + length))
    envelope += (phase > 0) * (phase < 0.4) * np.sin(np.pi * phase / 0.4)          # breathe in
    out = (phase >= 0.4) * np.sin(np.pi * (phase - 0.4) / 0.6)                      # breathe out
    envelope += 0.7 * out
    wheeze_env += out
    start += length
pitch = 180 + 20 * np.sin(2 * np.pi * 0.3 * t)  # the whistle wobbles a little
lung = breath_noise * envelope
lung += 1.5 * breath_noise.std() * wheeze_env * np.sin(2 * np.pi * np.cumsum(pitch) / SR)
save("synthetic_lung_wheeze.wav", lung)
