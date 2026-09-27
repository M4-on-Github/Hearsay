import torch
import torchaudio
import numpy as np
import soundfile as sf
import warnings

# Suppress speechbrain warnings
warnings.filterwarnings("ignore")

import pathlib
import shutil

# Monkeypatch symlink_to on Windows to prevent WinError 1314 (Privilege not held)
_orig_symlink_to = pathlib.Path.symlink_to
def _copy_instead_of_symlink(self, target, target_is_directory=False):
    if self.exists():
        return
    if target.is_dir():
        shutil.copytree(target, self)
    else:
        shutil.copy2(target, self)
pathlib.Path.symlink_to = _copy_instead_of_symlink

try:
    from speechbrain.inference.speaker import EncoderClassifier
    HAS_SB = True
except ImportError:
    HAS_SB = False

class ECAPATDNNExtractor:
    def __init__(self, model_name="speechbrain/spkrec-ecapa-voxceleb", device=None):
        if not HAS_SB:
            print("Speechbrain not installed. ECAPA-TDNN extractor will return zeros.")
            self.classifier = None
            return
            
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading ECAPA-TDNN model: {model_name} on {self.device}")
        run_opts = {"device": self.device}
        self.classifier = EncoderClassifier.from_hparams(source=model_name, savedir="pretrained_models/spkrec-ecapa-voxceleb", run_opts=run_opts)

    def extract_features(self, audio_path):
        """
        Extracts speaker embeddings (x-vectors) using ECAPA-TDNN.
        """
        if not self.classifier:
            # Return dummy vector if speechbrain isn't installed
            return np.zeros(192)
            
        try:
            data, fs = sf.read(audio_path)
            if len(data.shape) == 1:
                signal = torch.tensor(data).unsqueeze(0).float()
            else:
                signal = torch.tensor(data).transpose(0, 1).float()
            # Embeddings extraction
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(signal)
            return embeddings.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error extracting ECAPA-TDNN features for {audio_path}: {e}")
            return None
