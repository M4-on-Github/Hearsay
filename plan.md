# HEARSAY: Audio Authentication Challenge Plan

## 1. Data Sourcing Strategy
- **Hackathon Provided Data:** NSA will provide a curated, labeled sample training dataset (with ground truth) and an unlabeled sample test set. Access this via HexLabs, the NSA booth, or the NSA Discord.
- **External Datasets (For Calibration/Pre-training):** 
  - *Spoof sets:* ASVspoof (2019/2021/5), WaveFake, In-the-Wild, MLAAD, ADD Challenge sets.
  - *Bona fide sets:* VCTK, LibriSpeech, VOiCES.

## 2. Submission Requirements
Submissions must include:
1. **Source Code (GitHub/GitLab):** Full repository including a README documenting approach, system architecture, and run instructions.
2. **Docker Container:** An image that allows judges to reliably run the inference system on their hidden test set without configuration changes.
3. **Prediction File (`.tsv`):** A tab-delimited file generated from the sample test set. It must include:
   - `filename`: The audio file name (e.g., `file1.wav`).
   - `cm-score`: Synthetic probability float from 0.0 to 1.0 (1.0 = synthetic).
   - *Note:* Utilize the **one-time review** offered by NSA for a draft `.tsv` to assess performance before final submission.

## 3. State-of-the-Art (SOTA) Context (2024 ASVspoof)
Recent top-performing systems in audio deepfake detection emphasize:
- **Self-Supervised Learning (SSL) Front-ends:** Foundation models like **WavLM**, **XLSR**, and **Wav2Vec 2.0** are currently the top feature extractors due to superior generalization to unseen fakes.
- **AASIST Architectures:** AASIST (and its 2024 iteration AASIST3 with Kolmogorov-Arnold Networks) remains the gold standard for integrated spectro-temporal graph attention networks.
- **Ensemble/Fusion:** The very best systems use fusion strategies (e.g., WavLM + ResNet18 + AASIST), making our learned weighter ensemble approach perfectly aligned with current SOTA methodologies.

## 4. Execution Pipeline: Learned Weighter & Ablations

### Phase 1: Tool Orchestration & Feature Extraction
Treat each tool as a feature extractor. For every audio file, generate a vector of outputs based on:
- **Deep Learning Detectors:** AASIST, RawNet2/3, WavLM, Whisper encoders, ECAPA-TDNN. Extract their probability scores or embeddings.
- **Signal Processing & Inspection:** `librosa`, `torchaudio`, `Praat`, `SoX`. Extract metrics like spectral centroid, pitch contour, phase breaks.
- **Metadata Analysis:** `ExifTool`, `FFmpeg/ffprobe`. Output boolean flags or categorical features regarding codec chains, container tags, and MAC timestamps.

### Phase 2: Data Splitting (Train / Dev / Test)
Since the test set is unlabeled, we must create a local validation split from the provided training data:
- **Train Split (e.g., 70-80% of labeled data):** Train the learned "weighter" (e.g., Logistic Regression, Random Forest, XGBoost, or a shallow MLP). These algorithms will output feature importance weights based on the tools' outputs.
- **Dev/Validation Split (e.g., 20-30% of labeled data):** Used exclusively to tune hyperparameters and run ablation studies.
- **Test Set (Provided unlabeled data):** Used only for generating the final `.tsv` submission.

### Phase 3: Methodical Ablation Study
Run tests on the Dev split to find the optimal combination of tools:
1. **Leave-One-Out (LOO) Ablation:** Train the weighter using all tools *except one*. Do this for every tool. If dropping a tool improves Dev performance, that tool is adding noise and should be removed from the final ensemble.
2. **Single-Tool Baselines:** Train the weighter using *only one* tool at a time to identify the strongest individual predictors.
3. **Additive Approach:** Start with the best single tool and iteratively add the next best tools until performance plateaus.

### Phase 4: Agentic Orchestration (Bonus Optimization)
To satisfy the "Agentic orchestration bonus," consider using an LLM framework (LangChain, LangGraph, or CrewAI) as an adjudicator. The agent can look at the extracted tool vector and metadata to logically decide the final score, rather than brute-forcing a traditional ML model, adding high explainability.
