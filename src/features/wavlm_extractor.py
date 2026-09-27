import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import numpy as np

class WavLMExtractor:
    def __init__(self, model_name="microsoft/wavlm-base-plus", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM model: {model_name} on {self.device}")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(self, audio_path):
        """
        Extracts the mean-pooled last hidden state from WavLM for the given audio file.
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed (WavLM expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                
            # If stereo, take the first channel
            if waveform.shape[0] > 1:
                waveform = waveform[0, :].unsqueeze(0)
                
            # Prepare inputs
            inputs = self.feature_extractor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)
            
            # Extract features without computing gradients
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Mean pool over the time dimension
            last_hidden_state = outputs.last_hidden_state
            pooled_features = torch.mean(last_hidden_state, dim=1).squeeze()
            
            return pooled_features.cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting WavLM features for {audio_path}: {e}")
            return None
