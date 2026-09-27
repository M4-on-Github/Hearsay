import os
import subprocess
import numpy as np

class MetadataExtractor:
    def __init__(self):
        print("Initializing Metadata Extractor...")
        # Check if ffprobe is installed
        try:
            subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.has_ffprobe = True
        except FileNotFoundError:
            self.has_ffprobe = False
            print("Warning: ffprobe not found in PATH. Metadata extraction will be limited.")

    def extract_features(self, audio_path):
        """
        Extract basic metadata features. 
        Returns numeric array representing metadata clues.
        """
        features = []
        try:
            # 1. File size (anomalously small/large files might be suspicious)
            size_bytes = os.path.getsize(audio_path)
            features.append(size_bytes)
            
            # 2. Extension check
            ext = audio_path.split('.')[-1].lower()
            ext_map = {'wav': 1, 'mp3': 2, 'm4a': 3, 'ogg': 4, 'flac': 5}
            features.append(ext_map.get(ext, 0))
            
            # If ffprobe is available, we could extract bit_rate, codec, etc.
            if self.has_ffprobe:
                # Example: get bitrate
                cmd = ["ffprobe", "-v", "error", "-show_entries", "format=bit_rate", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    bitrate = float(res.stdout.strip())
                    features.append(bitrate)
                except ValueError:
                    features.append(-1.0)
            else:
                features.append(-1.0) # placeholder for bitrate
                
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting metadata for {audio_path}: {e}")
            return None
