"""
Digital Signal Processing (DSP) Module for Audio Spoofing Detection.

This module provides a suite of audio transformations designed to augment 
synthetic and real speech data. It includes simulations for codecs, harmonics,
spectraltilt, jitter-shimmer effects, and stochastic noise. It specifically
simulates physical and digital artifacts to evaluate and improve the robustness
of spoofing detection models like AASIST3.
"""

import io
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.DEBUG)

class DSPPipeline(torch.nn.Module):
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sr = sample_rate
        # Codec to hide artifacts (torchaudio 2.11 API)
        # self.codec = torchaudio.transforms.Codec(format="mp3")

    def apply_codec_simulation(self, x: torch.Tensor, format: str = "mp3", bitrate: int = 64000) -> torch.Tensor:
        """
        Simulates codec compression degradation.
        Note: The round-trip through BytesIO is computationally expensive for training.
        """
        config = torchaudio.io.CodecConfig(bit_rate=bitrate)
        buffer = io.BytesIO()
        torchaudio.save(buffer, x.cpu(), self.sr, format=format, compression=config)
        buffer.seek(0)
        x_compressed, _ = torchaudio.load(buffer)
        return x_compressed.to(x.device)

    def apply_spectral_tilt(self, x: torch.Tensor, tilt_db_per_octave: float) -> torch.Tensor:
        """
        Adjusts the spectral tilt using a low-pass biquad approximation.
        The agent controls high-frequency gain.
        """
        return F.lowpass_biquad(x, self.sr, cutoff_freq=2000, Q=0.707) * (1 + tilt_db_per_octave)

    def apply_harmonics(self, x: torch.Tensor, drive: float) -> torch.Tensor:
        """Generates harmonics via non-linear saturation distortion."""
        return torch.tanh(x * (1 + drive))

    def apply_jitter_shimmer(self, x: torch.Tensor, jitter_level: float, shimmer_level: float) -> torch.Tensor:
        """
        Applies micro-variations in frequency (jitter) and amplitude (shimmer).
        
        Shimmer: Stochastic amplitude modulation.
        Jitter: Phase jitter approximation via random shift.
        """
        if shimmer_level > 0:
            noise = torch.randn_like(x) * float(shimmer_level) 
            x = x * (1 + noise)
            
        if jitter_level > 0:
            phase_shift = torch.randn_like(x) * float(jitter_level)
            x = x + phase_shift

        return x
    
    def apply_compression(self, x: torch.Tensor, threshold_db: float, ratio: float) -> torch.Tensor:
        """
        Applies a basic dynamic range compressor.
        """
        # Convert to logarithmic scale: 20 * log10(|x| + epsilon)
        x_db = 20 * torch.log10(torch.abs(x) + 1e-6)
        
        mask = x_db > threshold_db
        # Gain reduction formula: (Current_dB - Threshold) * (1 - 1/Ratio)
        gain_reduction = (x_db[mask] - threshold_db) * (1 - 1 / ratio)
        x_db[mask] -= gain_reduction
        
        # Return to linear scale: 10^(dB/20) * sign(x)
        return 10**(x_db / 20) * torch.sign(x)
    
    def apply_noise(self, x: torch.Tensor, noise_tensor: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Injects noise at a specific Signal-to-Noise Ratio (SNR).
        Note: This is a simple additive noise model.
        For more realism, consider using a noise profile that matches the target environment.
        """
        if noise_tensor.size(-1) > x.size(-1):
            noise_tensor = noise_tensor[:, :x.size(-1)]
        
        p_signal = x.pow(2).mean()
        p_noise = noise_tensor.pow(2).mean()
        
        snr_linear = 10 ** (snr_db / 10)
        gain = torch.sqrt(p_signal / (snr_linear * p_noise + 1e-9))
        
        return x + gain * noise_tensor
        
    def apply_reverb(self, x: torch.Tensor, ir_tensor: torch.Tensor, wet_dry: float = 0.3) -> torch.Tensor:
        """Applies reverb via FFT convolution."""
        ir_tensor = ir_tensor / (torch.norm(ir_tensor, p=2) + 1e-9)
        wet = F.fftconvolve(x, ir_tensor, mode="full")
        wet = wet[..., :x.size(-1)]
        
        return (1 - wet_dry) * x + wet_dry * wet

    def forward(self, audio: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Orchestrates transformations. 
        Note: Normalization is applied at the end to ensure AASIST3 compatibility.
        """
        x = audio

        x = self.apply_jitter_shimmer(x, params['jitter'], params['shimmer'])
        x = self.apply_spectral_tilt(x, params['tilt'])
        x = self.apply_harmonics(x, params['harmonics'])
        x = self.apply_compression(x, params['threshold'], params['ratio'])
        
        bitrate = params.get('bitrate', 128000)
        x = self.apply_codec_simulation(x, bitrate=bitrate)

        # Peak normalization to prevent volume-based bias
        if x.abs().max() > 0:
            x = x / x.abs().max()

        return x


# Possible Improvements
# 1.  **Jitter Implementation**:
# Currently `apply_jitter_shimmer` adds `phase_shift` directly to the amplitude. This is technically **additive noise**, not jitter. True jitter involves temporal displacement ($x[t + \delta]$). For a thesis, consider implementing jitter using `F.resample` or a variable delay line if it need to be precise about phase artifacts.
# 2.  **Performance Bottleneck**:
# The `apply_codec_simulation` method involves `io.BytesIO` and a file-save simulation. Running this inside a training loop for RL will be extremely slow. It is better to pre-compute codec artifacts or use a differentiable approximation if speed becomes an issue.
# 3.  **Spectral Tilt**:
# The current `lowpass_biquad` approach is more of a fixed-slope filter than a true spectral tilt. A standard tilt usually pivots around a frequency (e.g., 1kHz) and applies a linear gain/attenuation slope across the entire spectrum.