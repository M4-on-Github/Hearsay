# HEARSAY: Audio Authentication Pipeline

This repository contains our submission for the HEARSAY Audio Authentication challenge.

## 🌟 Core Innovation: SICKLE-Inspired Acoustic Curation
A major bottleneck in audio forensic analysis is compute efficiency when processing massive datasets (e.g., 65,000+ synthetic files). Blindly subsampling data destroys acoustic diversity. 

Inspired by ORNL's SICKLE framework (Sparse Intelligent Curation Framework for Learning), we implemented a **Fast Acoustic Profiling & MaxEnt Curation Pipeline**:
1. **Fast Sweep**: We run an ultra-fast `SignalExtractor` (Librosa MFCCs, Spectral Centroid, Zero Crossing Rate) across tens of thousands of audio files to build lightweight acoustic profiles.
2. **MaxEnt Curation**: We use `MiniBatchKMeans` clustering and `pairwise_distances_argmin_min` to mathematically select the absolute most acoustically diverse and information-dense subsets (centroids).
3. **Targeted Deep Learning**: We only pass these intelligently curated "golden files" through our heavy Deep Learning extractors, guaranteeing maximum forensic diversity without the CPU bottleneck.

## 🏗️ Architecture: The Learned Weighter Ensemble
Our system extracts highly diverse forensic signals from the intelligently curated audio and combines them using an **XGBoost Classifier**. This satisfies the challenge's request for forensic diversity, scalability, and explainable AI.

The ensemble combines concatenated features from five distinct forensic domains:
1. **Self-Supervised Learning (SSL):** Microsoft `WavLM` transformer embeddings (captures deep semantic/acoustic anomalies).
2. **Automatic Speech Recognition (ASR):** OpenAI `Whisper` (captures unnatural phonetic alignments and hallucinated pronunciations).
3. **Speaker Verification:** `ECAPA-TDNN` (captures unnatural vocal tract modeling and speaker embedding artifacts).
4. **Digital Signal Processing:** `librosa` acoustic features (captures vocoder artifacts, pitch anomalies, and prosody).
5. **Metadata Analysis:** File structure telemetry (captures laundering and container spoofing).

By relying on XGBoost as our aggregator, we naturally handle the massive variance in feature scales (from deep transformer logits to raw metadata byte counts) while retaining full feature-importance explainability.

## ⚙️ Usage

### 1. Training the Ensemble (Locally)
Place the labeled training data in the `data/real/` and `data/synthetic/` directories.
Run:
```bash
python main.py
```
This triggers the SICKLE curation phase, extracts the Deep Learning features, splits a local Dev set for evaluation, trains the XGBoost weighter, and saves it to `ensemble_model.pkl`.

### 2. Inference (Locally)
Place the unlabelled sample test set in the `data/test/` directory.
Run:
```bash
python predict.py
```
This will run inference on the test files and output `teamName_predictions.tsv` in the exact format required by the judges.

## 🐳 Docker Reproduction (For Judges)
We have provided a standalone Docker image that reliably runs our inference system on the hidden test set without configuration changes.

Build the image:
```bash
docker build -t hearsay-ensemble .
```

Run inference (mount the hidden test dataset to `/app/data/test`):
```bash
docker run --rm -v $(pwd)/data/test:/app/data/test hearsay-ensemble
```
The predictions will be saved to `teamName_predictions.tsv` inside the container or mounted directory.
