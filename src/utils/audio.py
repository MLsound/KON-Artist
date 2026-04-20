"""
Signal Standardization and Pre-emphasis Module.

This module implements the canonical preprocessing pipeline for AASIST3-based
audio spoofing detection. It ensures input signals are standardized across
the data pipeline by applying resampling, mono-conversion, high-pass
pre-emphasis filtering, and temporal shaping to a fixed sample length.
"""
import torch
import torch.nn.functional as F
import torchaudio

def preprocess_audio(audio_tensor: torch.Tensor, sr: int, target_sr: int = 16000, target_len: int = 64600) -> torch.Tensor:
    """
    Standard AASIST3 preprocessing pipeline.
    """
    # 1. Resampling
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio_tensor = resampler(audio_tensor)

    # 2. Mono Conversion
    if audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

    # 3. Pre-emphasis Filter
    # Balances the high-frequency components often diminished in synthetic audio
    audio_tensor = torch.cat((audio_tensor[:, :1], audio_tensor[:, 1:] - 0.97 * audio_tensor[:, :-1]), dim=1)

    # 4. Z-Score Normalization
    audio_tensor = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-7)
    
    # 5. Temporal Shaping (Fixed 64,600 samples ~4 seconds)
    current_len = audio_tensor.shape[1]
    if current_len < target_len:
        audio_tensor = F.pad(audio_tensor, (0, target_len - current_len))
    else:
        audio_tensor = audio_tensor[:, :target_len]

    return audio_tensor