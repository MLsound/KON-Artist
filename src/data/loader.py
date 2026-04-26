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

def get_asvspoof_loader(split: str = "train", buffer_size: int = 10000):
    """
    Loads and shuffles the ASVspoof 2019 LA dataset in streaming mode.
    Args:
    - split: Dataset split to load (e.g., "train", "validation", "test")
    - buffer_size: Size of the shuffle buffer for streaming datasets
    Returns:
    - A Hugging Face Dataset object with streaming and shuffling enabled.
    """
    # load_dataset provides the entry point to Hugging Face datasets
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split=split, streaming=True)
    
    # Disable decoding: This returns raw bytes/path instead of a decoded dictionary.
    # This ensures sample["audio"] returns a dict with {'bytes': ..., 'path': ...}
    ds = ds.cast_column("audio", Audio(decode=False))

    # Shuffle buffer is required to bypass the ordered nature of the dataset
    shuffled_ds = ds.shuffle(seed=42, buffer_size=buffer_size)
    
    return shuffled_ds

def generator_from_ds(shuffled_ds):
    """Yields preprocessed tensors (ONLY spoofed) and labels for the RL Env."""
    for sample in shuffled_ds:
        if sample["key"] == 1: # 1 is Bonafide, 0 is Spoof
            continue # For now, we yield all samples; filtering can be done in the environment if needed
        else:
            # Access audio data array and sampling rate from the decoded audio column
            # audio_data = torch.from_numpy(sample["audio"]["array"]).float().unsqueeze(0)
            # sr = sample["audio"]["sampling_rate"]

            # Access the raw bytes directly
            audio_bytes = sample["audio"]["bytes"]
            label = sample["key"] # 1 for Bonafide, 0 for Spoof
            
            # Load directly into torch tensor (Mono-conversion happens here)
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

            # Standardize via your existing pipeline
            # processed = preprocess_audio(audio_data, sr)
            processed = preprocess_audio(waveform, sr)

            yield processed, label