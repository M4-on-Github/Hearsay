import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel
import numpy as np

class WhisperExtractor:
    def __init__(self, model_name="openai/whisper-tiny", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper model: {model_name} on {self.device}")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(self, audio_path):
        """
        Extracts the mean-pooled last hidden state from Whisper encoder.
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Whisper expects 16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                
            if waveform.shape[0] > 1:
                waveform = waveform[0, :].unsqueeze(0)
                
            inputs = self.feature_extractor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # We only need the encoder outputs for audio representations
                outputs = self.model.encoder(**inputs)
                
            last_hidden_state = outputs.last_hidden_state
            pooled_features = torch.mean(last_hidden_state, dim=1).squeeze()
            
            return pooled_features.cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting Whisper features for {audio_path}: {e}")
            return None
