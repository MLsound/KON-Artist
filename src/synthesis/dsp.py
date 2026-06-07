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
        Optimized codec simulation using a differentiable approximation.
        Replaces slow io.BytesIO round-trips with adaptive LPF and quantization noise.
        
        - LPF: Simulates bandwidth limiting of low-bitrate codecs.
        - Quantization: Simulates compression artifacts and bit-depth reduction.
        """
        # 1. Adaptive Bandwidth Limiting (LPF)
        # For 16kHz audio, Nyquist is 8kHz.
        # Heuristic: Cutoff frequency scales with bitrate.
        # e.g., 32kbps -> ~6kHz cutoff, 64kbps+ -> ~8kHz (no cut)
        max_bw = self.sr / 2
        cutoff = min(max_bw - 100, (bitrate / 48000) * max_bw)
        x = F.lowpass_biquad(x, self.sr, cutoff_freq=max(2000.0, cutoff))

        # 2. Quantization Noise Simulation
        # Simulate sub-band quantization artifacts.
        # Higher bitrate = more 'effective bits' and less noise.
        # Range: ~2 bits at 8kbps to ~12 bits at 128kbps
        eff_bits = max(2.0, min(12.0, bitrate / 10000))
        levels = 2 ** eff_bits
        
        # Straight-Through Estimator (STE) for differentiability
        # This allows gradients to flow back if the agent's policy ever requires it.
        x_quant = torch.round(x * levels) / levels
        x = x + (x_quant - x).detach()
        
        return x
    def apply_spectral_tilt(self, x: torch.Tensor, tilt_db_per_octave: float, pivot_freq: float = 1000.0) -> torch.Tensor:
        """
        Applies a spectral tilt pivoting around pivot_freq.
        A positive tilt boosts highs and cuts lows.
        Uses an FFT-based frequency domain gain ramp for technical precision.
        """
        if tilt_db_per_octave == 0:
            return x
            
        # x: (C, L) or (B, C, L)
        original_shape = x.shape
        length = x.shape[-1]
        
        # We use rfft on the last dimension
        x_fft = torch.fft.rfft(x, dim=-1)
        n_freq = x_fft.shape[-1]
        
        # Frequencies corresponding to FFT bins
        freqs = torch.linspace(0, self.sr / 2, n_freq, device=x.device)
        
        # Gain calculation: 
        # Gain_dB = slope * log2(f / f_pivot)
        # We use a small epsilon for f=0 to avoid log(0)
        gain_db = tilt_db_per_octave * torch.log2((freqs + 1e-6) / pivot_freq)
        gain = 10 ** (gain_db / 20)
        
        # Apply gain to the complex spectrum
        # Ensure gain is broadcastable over batch/channels
        x_fft_tilted = x_fft * gain
        
        # Inverse FFT to return to time domain
        x_tilted = torch.fft.irfft(x_fft_tilted, n=length, dim=-1)
        
        return x_tilted.view(original_shape)

    def apply_harmonics(self, x: torch.Tensor, drive: float) -> torch.Tensor:
        """Generates harmonics via non-linear saturation distortion."""
        return torch.tanh(x * (1 + drive))

    def apply_jitter(self, x: torch.Tensor, jitter_level: float) -> torch.Tensor:
        """
        Applies true jitter via stochastic temporal displacement x[t + delta].
        Delta is a small random shift per sample, implemented via bilinear interpolation.
        """
        if jitter_level <= 0:
            return x
            
        original_shape = x.shape
        # Ensure x is (Batch, Channels, Length) for grid_sample logic
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0) # [1, 1, L]
        elif x.dim() == 2:
            x = x.unsqueeze(0) # [1, C, L]
            
        batch, channels, length = x.shape
        
        # Stochastic displacement: delta ~ Uniform(-jitter_level, jitter_level)
        # Scaled to micro-variations (max shift of 2 samples)
        max_shift = 2.0 
        displacement = (torch.rand(batch, 1, length, device=x.device) * 2 - 1) * jitter_level * max_shift
        
        # Create a time grid and apply displacement
        indices = torch.linspace(0, length - 1, length, device=x.device).repeat(batch, 1, 1)
        new_indices = indices + displacement
        
        # Normalize to [-1, 1] range for grid_sample
        normalized_indices = (new_indices / (length - 1)) * 2 - 1
        
        # Prepare 4D input (N, C, 1, L) and 4D grid (N, 1, L, 2) for grid_sample
        x_4d = x.unsqueeze(2) 
        y_coords = torch.zeros_like(normalized_indices)
        v_grid = torch.stack([normalized_indices, y_coords], dim=-1) # (N, 1, L, 2)
        
        x_jittered = torch.nn.functional.grid_sample(
            x_4d, 
            v_grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        out = x_jittered.squeeze(2)
        return out.view(original_shape)

    def apply_jitter_shimmer(self, x: torch.Tensor, jitter_level: float, shimmer_level: float) -> torch.Tensor:
        """
        Applies micro-variations in frequency (jitter) and amplitude (shimmer).
        
        Shimmer: Stochastic amplitude modulation.
        Jitter: True temporal displacement x[t + delta].
        """
        if shimmer_level > 0:
            noise = torch.randn_like(x) * float(shimmer_level) 
            x = x * (1 + noise)
            
        if jitter_level > 0:
            # Refined from additive noise to true temporal displacement
            x = self.apply_jitter(x, jitter_level)

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