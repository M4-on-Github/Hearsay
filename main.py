import os
import glob
import numpy as np
import pickle
from src.features.wavlm_extractor import WavLMExtractor
from src.features.signal_extractor import SignalExtractor
from src.features.metadata_extractor import MetadataExtractor
from src.features.whisper_extractor import WhisperExtractor
from src.features.ecapa_extractor import ECAPATDNNExtractor
from src.models.ensemble import AudioAuthenticationEnsemble
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    real_files = glob.glob(os.path.join(data_dir, "real", "**", "*.wav"), recursive=True)
    synth_files = glob.glob(os.path.join(data_dir, "synthetic", "**", "*.wav"), recursive=True)
    
    files = real_files + synth_files
    labels = [0] * len(real_files) + [1] * len(synth_files)
    return files, np.array(labels)

def main():
    print("Initializing components...")
    wavlm = WavLMExtractor(model_name="microsoft/wavlm-base-plus")
    signal_ext = SignalExtractor()
    meta_ext = MetadataExtractor()
    whisper_ext = WhisperExtractor(model_name="openai/whisper-tiny")
    ecapa_ext = ECAPATDNNExtractor()
    
    data_dir = "data/"
    if not os.path.exists(os.path.join(data_dir, "real")):
        print(f"Warning: {data_dir} does not have 'real' or 'synthetic' subdirectories.")
        os.makedirs(os.path.join(data_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "synthetic"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
        print("Created data directories. Place files and re-run.")
        return

    files, labels = load_data(data_dir)
    print(f"Found {len(files)} audio files.")
    
    if len(files) == 0:
        print("No files found. Exiting.")
        return

    print("Extracting features (this may take a while)...")
    features = []
    valid_labels = []
    
    for i, file_path in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {os.path.basename(file_path)}")
        w_feat = wavlm.extract_features(file_path)
        s_feat = signal_ext.extract_features(file_path)
        m_feat = meta_ext.extract_features(file_path)
        wh_feat = whisper_ext.extract_features(file_path)
        ec_feat = ecapa_ext.extract_features(file_path)
        
        if w_feat is not None and s_feat is not None and m_feat is not None and wh_feat is not None and ec_feat is not None:
            combined = np.hstack([w_feat, s_feat, m_feat, wh_feat, ec_feat])
            features.append(combined)
            valid_labels.append(labels[i])
            
    X = np.array(features)
    y = np.array(valid_labels)
    
    if len(X) == 0:
        print("Failed to extract features from any files.")
        return

    w_len = len(w_feat)
    s_len = len(s_feat)
    m_len = len(m_feat)
    wh_len = len(wh_feat)
    ec_len = len(ec_feat)
    
    feature_names = [f"wavlm_dim_{i}" for i in range(w_len)] + \
                    [f"signal_dim_{i}" for i in range(s_len)] + \
                    [f"meta_dim_{i}" for i in range(m_len)] + \
                    [f"whisper_dim_{i}" for i in range(wh_len)] + \
                    [f"ecapa_dim_{i}" for i in range(ec_len)]
    
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    ensemble = AudioAuthenticationEnsemble()
    ensemble.train(X_train, y_train, feature_names=feature_names)
    
    auc = ensemble.evaluate(X_dev, y_dev)
    
    importance = ensemble.get_feature_importance()
    print("\nTop 15 Most Important Features:")
    for name, imp in importance[:15]:
        print(f"{name}: {imp:.4f}")
        
    with open("ensemble_model.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    print("Saved trained ensemble to ensemble_model.pkl")

if __name__ == "__main__":
    main()
