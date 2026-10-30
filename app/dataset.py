# Reads the HLS-CMDS labels (3 csv files) and the wav files. We extracting the dataset once is simpler and faster.

import os
import wave

import numpy as np
import pandas as pd

# relative to this file, so the app works from any folder
DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")

LUNG_CODES = {"Normal": "N", "Wheezing": "W", "Fine Crackles": "FC", "Rhonchi": "R", "Pleural Rub": "PR", "Coarse Crackles": "CC"}

class Dataset:
    def __init__(self, folder=DATASET_FOLDER):
        if not os.path.isdir(folder):
            raise FileNotFoundError("Dataset folder not found. Put it next to the app folder:\n" + folder)
        self.folder = folder
        self.table = self._load_table()

    def csv(self, name):
        # one ID in LS.csv had a trailing space, so strip every column
        df = pd.read_csv(os.path.join(self.folder, name))
        return df.apply(lambda col: col.astype(str).str.strip())

    def read(self, folder, file_name):
        # all files are 16-bit mono, dividing by 32768 scales them to -1..1
        path = os.path.join(self.folder, folder, file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{file_name} is not in dataset/{folder}")
        with wave.open(path, "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise ValueError(f"{file_name}: expected mono 16-bit audio")
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, sr

    def _load_table(self):
        rows = []
        for _, r in self.csv("HS.csv").iterrows():
            rows.append({"file": r["Heart Sound ID"] + ".wav", "folder": "HS", "category": "Heart",
                         "sound_type": r["Heart Sound Type"], "location": r["Location"],
                         "gender": r["Gender"], "csv_id": r["Heart Sound ID"]})

        # 14 lung IDs in LS.csv use old codes (M_C_LUA is really M_FC_LUA.wav),
        # so we build the file name from gender + type + location instead
        for _, r in self.csv("LS.csv").iterrows():
            code = LUNG_CODES.get(r["Lung Sound Type"], "N")
            rows.append({"file": f"{r['Gender']}_{code}_{r['Location']}.wav", "folder": "LS",
                         "category": "Lung", "sound_type": r["Lung Sound Type"],
                         "location": r["Location"], "gender": r["Gender"],
                         "csv_id": r["Lung Sound ID"]})

        # each mixed file keeps the names of its matched heart-only and lung-only files
        for _, r in self.csv("Mix.csv").iterrows():
            rows.append({"file": r["Mixed Sound ID"] + ".wav", "folder": "Mix", "category": "Mixed",
                         "sound_type": r["Heart Sound Type"] + " + " + r["Lung Sound Type"],
                         "location": r["Location"], "gender": r["Gender"],
                         "csv_id": r["Mixed Sound ID"],
                         "heart_ref": r["Heart Sound ID"] + ".wav",
                         "lung_ref": r["Lung Sound ID"] + ".wav"})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    data = Dataset()
    print(data.table["category"].value_counts())
    x, sr = data.read("HS", data.table.iloc[0]["file"])
    print("samples:", len(x), "sample rate:", sr)
