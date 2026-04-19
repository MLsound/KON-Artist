"""
AASIST3 Model Wrapper for Spoofing Detection.

This module provides an interface for the AASIST3 (Audio Anti-Spoofing 
using Integrated Spectro-Temporal Graph Attention Networks) model. 
It facilitates model loading, inference, and the extraction of high-dimensional 
embeddings to be used as state representations in Reinforcement Learning (RL) 
experiments.
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.DEBUG)

class AASISTWrapper:
    """
    Wrapper for the AASIST3 model. Handles inference and embedding extraction for the RL environment state.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        # Ensure the device is available; fallback to CPU if necessary
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
            
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        self._embedding = None
        self._setup_hooks()
        logger.info(f"AASIST3 model loaded on {self.device}")

    def _load_model(self, path: str):
        # Resolve the absolute path to KON-Artist/third_party/AASIST3
        project_root = Path(__file__).resolve().parents[2]
        repo_path = str(project_root / "third_party" / "AASIST3")
        logger.debug(f"Attempting to load AASIST3 model from: {repo_path}")
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path) # Insert at 0 to prioritize this 'model' module
            logger.debug(f"Added {repo_path} to sys.path for AASIST3 imports.")
        
        try:
            logger.debug("Importing AASIST3 model from the third_party repository.")
            from model import aasist3
            logger.info("Successfully imported aasist3 from the AASIST3 repository.")
        except ImportError as e:
            logger.error(f"Failed to find AASIST3 in third_party: {e}")
            raise

        # Instantiate via Hugging Face or local checkpoint
        if os.path.exists(path):
            model = aasist3()
            model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Loaded AASIST3 model from local checkpoint: {path}")
        else:
            model = aasist3.from_pretrained(path) # Standard HF load
            logger.info(f"Loaded AASIST3 model from Hugging Face: {path}")
        
        model.to(self.device).eval()
        return model
    
    def _setup_hooks(self):
        """
        Registers a pre-hook on the final layer to capture the embedding.
        """
        for name, module in self.model.named_modules():
            if name == "out_layer":
                # register_forward_pre_hook captures the inputs passed to the layer
                module.register_forward_pre_hook(self._hook_fn)
                return
        logger.warning("Could not find out_layer for embeddings; state might be None.")

    def _hook_fn(self, module, args):
        """
        Intercepts the input arguments to the out_layer.
        args is a tuple; args[0] is the embedding tensor.
        """
        self._embedding = args[0].flatten(1)
        
    def get_score_and_embedding(self, waveform: torch.Tensor):
        """
        Performs inference. Returns the Bonafide probability score and the embedding.
        """
        with torch.no_grad():
            output = self.model(waveform.to(self.device))
            
            # 1. Standard model output is logits. Embedding is captured strictly via the pre-hook.
            logits = output[0] if isinstance(output, tuple) else output
            
            if self._embedding is None:
                raise RuntimeError("Embedding hook failed to fire. State representation is missing.")
            
            # Captured embedding from the hook
            embedding = self._embedding

            # 2. Convert logits to Bonafide probability
            # AASIST3 typically outputs [Spoof, Bonafide]
            probabilities = F.softmax(logits, dim=1)
            
            # Extract the probability for the Bonafide class (index 1)
            # Use .item() to convert the 1D tensor to a standard Python float
            score = probabilities[0, 1].item()
                
            return score, embedding
