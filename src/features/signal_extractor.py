import librosa
import numpy as np

class SignalExtractor:
    def __init__(self):
        print("Initializing Signal Extractor...")

    def extract_features(self, audio_path):
        """
        Extracts acoustic features using librosa.
        """
        try:
            # Load audio (sr=None preserves original sampling rate)
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract features
            # 1. MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            
            # 2. Spectral Centroid
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            cent_mean = np.mean(cent)
            
            # 3. Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rolloff_mean = np.mean(rolloff)
            
            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            
            # Combine into a single feature vector
            features = np.hstack([mfccs_mean, cent_mean, rolloff_mean, zcr_mean])
            return features
            
        except Exception as e:
            print(f"Error extracting signal features for {audio_path}: {e}")
            return None
