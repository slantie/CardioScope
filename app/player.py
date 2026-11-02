# player.py
# AudioPlayer: plays a recording on the speakers and records from the microphone.

import numpy as np

import features

try:
    import sounddevice as sd
except Exception:  # sound is optional, the rest of the app still works
    sd = None


class AudioPlayer:
    MIC_SR = 44100  # most microphones don't support 4000 Hz directly

    def __init__(self):
        self.recording = None

    def _check(self):
        if sd is None:
            raise RuntimeError("sounddevice is not installed.")

    @staticmethod
    def _run(start):
        """If the audio device changed while the app was open (headphones, Bluetooth...)
        PortAudio fails with an internal error, so restart it once and try again."""
        sd.stop()
        try:
            return start()
        except sd.PortAudioError:
            sd._terminate()
            sd._initialize()
            return start()

    def play(self, x, sr):
        self._check()

        def start():
            # play at the speaker's own rate: a 4000 Hz stream sometimes fails on macOS
            out_sr = int(sd.query_devices(kind="output")["default_samplerate"])
            audio, _ = features.resample(x, sr, out_sr)
            sd.play(np.clip(audio * 5, -1, 1), out_sr)  # the recordings are quiet, boost a bit
        self._run(start)

    def stop(self):
        if sd is not None:
            sd.stop()

    def start_recording(self, seconds):
        self._check()
        self.recording = self._run(lambda: sd.rec(int(seconds * self.MIC_SR),
                                                  samplerate=self.MIC_SR, channels=1))

    def finish_recording(self):
        """Wait for the recording to end. Returns (audio, sample_rate)."""
        sd.wait()
        return self.recording[:, 0], self.MIC_SR
