# HEARSAY: Audio Authentication Pipeline
This repository contains our submission for the HEARSAY Audio Authentication challenge.

## Architecture
Our system uses a **Learned Weighter Ensemble** approach that extracts diverse forensic signals from audio and combines them using an XGBoost classifier. This satisfies the challenge's request for forensic diversity and explainable AI.

The ensemble combines features from:
1. **Self-Supervised Learning (SSL):** Microsoft WavLM transformer embeddings (captures deep semantic/acoustic anomalies).
2. **Signal Processing:** `librosa` acoustic features including MFCCs, Spectral Centroid, and Zero Crossing Rate (captures vocoder artifacts and prosody).
3. **Metadata Analysis:** File structure and ffprobe telemetry (captures laundering and container spoofing).

## Setup
1. Create a python virtual environment and activate it.
2. Run `pip install -r requirements.txt`
3. Ensure `ffmpeg` and `ffprobe` are installed on your system.

## Usage

### 1. Training the Ensemble
Place the provided labeled training data in the `data/real/` and `data/synthetic/` directories.
Run:
```bash
python main.py
```
This extracts features, splits a local Dev set for evaluation, trains the XGBoost weighter, and saves it to `ensemble_model.pkl`.

### 2. Inference & Submission
Place the unlabelled sample test set in the `data/test/` directory.
Run:
```bash
python predict.py
```
This will output `teamName_predictions.tsv` in the exact format required by the judges.

## Docker Reproduction
To run inference via Docker:
```bash
docker build -t hearsay-ensemble .
docker run -v $(pwd)/data/test:/app/data/test hearsay-ensemble
```
