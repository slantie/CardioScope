# recordings.py
# NewRecordings: recordings from outside the dataset (imported wav files and the microphone).

import os
from datetime import datetime

import numpy as np
from scipy.io import wavfile

import features

RECORDINGS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
TARGET_SR = 4000     # all dataset recordings are 4000 Hz
TARGET_RMS = 0.003   # median loudness of the dataset recordings


class NewRecordings:
    """Makes an outside recording look like the dataset, and saves microphone recordings."""

    def __init__(self, folder=RECORDINGS_FOLDER):
        self.folder = folder

    @staticmethod
    def prepare(x, sr):
        """Any recording -> mono, -1..1, 4000 Hz, same loudness as the dataset."""
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.integer):
            x = x / float(np.iinfo(x.dtype).max)
        x = x.astype(np.float32)
        if x.ndim == 2:  # stereo -> mono
            x = x.mean(axis=1)
        x, _ = features.resample(x, sr, TARGET_SR)
        # a phone or microphone is much louder than the dataset, and loudness is one of
        # the model's features, so scale it to the typical dataset level
        rms = np.sqrt(np.mean(x ** 2))
        if rms > 0:
            x = x * (TARGET_RMS / rms)
        return x, TARGET_SR

    def read(self, path):
        """Read a wav file chosen by the user (any sample rate, mono or stereo)."""
        try:
            sr, x = wavfile.read(path)
        except Exception as e:
            raise ValueError(f"Not a readable wav file: {e}")
        if len(x) == 0:
            raise ValueError("The wav file is empty.")
        return self.prepare(x, sr)

    def save(self, audio, sr, kind):
        """Save raw microphone audio as a 16-bit wav, e.g. heart_20261116_103334.wav.
        Returns the file name."""
        name = f"{kind.lower()}_{datetime.now():%Y%m%d_%H%M%S}.wav"
        os.makedirs(self.folder, exist_ok=True)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        wavfile.write(os.path.join(self.folder, name), sr, pcm)
        return name
