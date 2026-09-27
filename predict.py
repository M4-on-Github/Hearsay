import os
import glob
import pandas as pd
import numpy as np
import pickle
from src.features.wavlm_extractor import WavLMExtractor
from src.features.signal_extractor import SignalExtractor
from src.features.metadata_extractor import MetadataExtractor
from src.features.whisper_extractor import WhisperExtractor
from src.features.ecapa_extractor import ECAPATDNNExtractor

def main():
    test_dir = "data/test/"
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found! Please create it and add test audio files.")
        os.makedirs(test_dir, exist_ok=True)
        return

    test_files = glob.glob(os.path.join(test_dir, "**", "*.wav"), recursive=True)
    if not test_files:
        print("No audio files found in test directory.")
        return

    print("Loading models...")
    wavlm = WavLMExtractor(model_name="microsoft/wavlm-base-plus", device="cpu")
    signal_ext = SignalExtractor()
    meta_ext = MetadataExtractor()
    whisper_ext = WhisperExtractor(model_name="openai/whisper-tiny", device="cpu")
    ecapa_ext = ECAPATDNNExtractor(device="cpu")
    
    # Load ensemble
    try:
        with open("ensemble_model.pkl", "rb") as f:
            ensemble = pickle.load(f)
    except FileNotFoundError:
        print("Ensemble model not found. Run main.py to train it first.")
        return
        
    # Load template
    template_path = r"data\HearsayScoreKey4TeamX.tsv"
    if not os.path.exists(template_path):
        print(f"Template not found at {template_path}. Cannot generate valid submission.")
        return
        
    df = pd.read_csv(template_path, sep='\t')
    
    predictions_map = {}
    
    print("Running inference...")
    for i, f_path in enumerate(test_files):
        filename = os.path.basename(f_path)
        print(f"Processing {i+1}/{len(test_files)}: {filename}")
        
        # Extract features
        w_feat = wavlm.extract_features(f_path)
        s_feat = signal_ext.extract_features(f_path)
        m_feat = meta_ext.extract_features(f_path)
        wh_feat = whisper_ext.extract_features(f_path)
        ec_feat = ecapa_ext.extract_features(f_path)
        
        if w_feat is None or s_feat is None or m_feat is None or wh_feat is None or ec_feat is None:
            print(f"Skipping {filename} due to extraction error.")
            continue
            
        # Combine
        combined_feat = np.hstack([w_feat, s_feat, m_feat, wh_feat, ec_feat]).reshape(1, -1)
        
        # Predict probability of being synthetic (Class 1)
        prob = ensemble.predict_proba(combined_feat)[0]
        predictions_map[filename] = float(prob)
        
    # Generate TSV
    for idx, row in df.iterrows():
        fname = row['filename']
        if fname in predictions_map:
            df.at[idx, 'cm-score'] = predictions_map[fname]
            
    out_file = "teamName_predictions.tsv"
    df.to_csv(out_file, sep='\t', index=False)
    print(f"Saved predictions to {out_file}")

if __name__ == "__main__":
    main()
