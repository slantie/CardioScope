# CardioScope

CardioScope is a small desktop app for exploring heart and lung recordings. It loads the supplied WAV dataset, shows waveform and frequency plots, extracts signal features, and provides a simple model-based suggestion report.

## Run locally

```bash
python -m pip install -r app/requirements.txt
python app/app.py
```

The project is organised around three areas: `slantie` handles the application and recording workflow, `ThakorMeet` handles plots and comparison views, and `HarshalRaval` handles features, models, and reports.

This project is for demonstration and coursework use; its suggestions are not medical advice.
