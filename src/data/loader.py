"""
ASVspoof Data Ingestion and Streaming Orchestrator.

Part of the KON-Artist ETL architecture, this module manages
the streaming ingestion of the ASVspoof 2019 LA dataset. It implements
stochastic shuffling via a memory-mapped buffer to handle large-scale
audio datasets without exhaustive local storage.
"""
import io
import torchaudio
from datasets import load_dataset, Audio
from src.utils.audio import preprocess_audio

def get_asvspoof_loader(split: str = "train", buffer_size: int = 10000, seed: int = 42):
    """
    Loads and shuffles the ASVspoof 2019 LA dataset in streaming mode.
    Args:
    - split: Dataset split to load (e.g., "train", "validation", "test")
    - buffer_size: Size of the shuffle buffer for streaming datasets
    - seed: Seed for the shuffle buffer
    Returns:
    - A Hugging Face Dataset object with streaming and shuffling enabled.
    """
    # load_dataset provides the entry point to Hugging Face datasets
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split=split, streaming=True)
    
    # Disable decoding: This returns raw bytes/path instead of a decoded dictionary.
    # This ensures sample["audio"] returns a dict with {'bytes': ..., 'path': ...}
    ds = ds.cast_column("audio", Audio(decode=False))

    # Shuffle buffer is required to bypass the ordered nature of the dataset
    shuffled_ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    
    return shuffled_ds

def generator_from_ds(shuffled_ds):
    """
    Yields preprocessed tensors (STRICTLY Spoof) and labels for the RL Env.
    Ensures that Bonafide signals are ignored to comply with categorical exclusion rules.
    """
    for sample in shuffled_ds:
        # ASVspoof 2019 LA: 0 is Bonafide, 1 is Spoof
        # MANDATORY: Only yield spoofed signals.
        if sample["key"] == 0:
            continue 
            
        # Access the raw bytes directly
        audio_bytes = sample["audio"]["bytes"]
        label = sample["key"] # Should be 1 (Spoof)
        
        # Load directly into torch tensor (Mono-conversion happens here)
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

        # Standardize via existing pipeline
        processed = preprocess_audio(waveform, sr)

        yield processed, label

def eval_generator(shuffled_ds):
    """Yields all samples (Bonafide and Spoof) for evaluation."""
    for sample in shuffled_ds:
        audio_bytes = sample["audio"]["bytes"]
        label = sample["key"] # 1 for Bonafide, 0 for Spoof

        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        processed = preprocess_audio(waveform, sr)

        yield processed, label