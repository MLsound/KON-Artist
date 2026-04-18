"""
ASVspoof Data Ingestion and Streaming Orchestrator.

Part of the KON-Artist ETL architecture, this module manages
the streaming ingestion of the ASVspoof 2019 LA dataset. It implements
stochastic shuffling via a memory-mapped buffer to handle large-scale
audio datasets without exhaustive local storage.
"""
import torch
from datasets import load_dataset
from src.utils.audio import preprocess_audio

def get_asvspoof_loader(split: str = "test", buffer_size: int = 10000, batch_size: int = 1):
    """
    Loads and shuffles the ASVspoof 2019 LA dataset in streaming mode.
    """
    # load_dataset provides the entry point to Hugging Face datasets
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split=split, streaming=True)
    
    # Shuffle buffer is required to bypass the ordered nature of the dataset
    shuffled_ds = ds.shuffle(seed=42, buffer_size=buffer_size)
    
    return shuffled_ds

def generator_from_ds(shuffled_ds):
    """Yields preprocessed tensors and labels for the RL Env."""
    for sample in shuffled_ds:
        audio_data = torch.from_numpy(sample["audio"]["array"]).float().unsqueeze(0)
        sr = sample["audio"]["sampling_rate"]
        label = sample["key"] # 1 for Bonafide, 0 for Spoof
        
        processed = preprocess_audio(audio_data, sr)
        yield processed, label