"""
Offline Acoustic Clustering Pipeline for KON-Artist.

This script extracts high-dimensional embeddings from the AASIST3 model
for the entire ASVspoof 2019 LA training split (spoofed samples),
reduces their dimensionality via PCA, and fits a Gaussian Mixture Model (GMM)
to identify acoustic sub-manifolds.
"""

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.models.aasist import AASISTWrapper
from src.data.loader import get_asvspoof_loader, generator_from_ds
from src.utils.logger import get_logger
import logging
import yaml

def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
logger = get_logger(name=__file__, level=logging.INFO)

def extract_all_embeddings(detector, loader, device="cuda"):
    """
    Extracts embeddings for all spoofed samples in the loader.
    Uses batching (implicitly via generator) and torch.no_grad() to prevent VRAM saturation.
    """
    embeddings = []
    logger.info("Starting embedding extraction for the training split...")
    
    # We use the generator which yields (waveform, label)
    # The detector wrapper handle [1, L] or [B, L]
    
    with torch.no_grad():
        for waveform, label in tqdm(generator_from_ds(loader), desc="Extracting embeddings"):
            # AASISTWrapper expects waveform as tensor
            # generator_from_ds yields preprocessed tensors
            _, embedding = detector.get_score_and_embedding(waveform)
            embeddings.append(embedding.cpu().numpy().flatten())
            
    return np.array(embeddings).astype(np.float32)

def run_clustering_pipeline():
    # 1. Hardware and Detector Setup
    device = cfg['model'].get('device', 'cpu')
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
        
    detector = AASISTWrapper(cfg['model']['detector_name'], device=device)
    
    # 2. Data Loading
    loader = get_asvspoof_loader(split="train", buffer_size=cfg['audio']['buffer_size'])
    
    # 3. Extraction
    embeddings = extract_all_embeddings(detector, loader, device=device)
    logger.info(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")
    
    # 4. Fit PCA + GMM Pipeline
    pca_dim = cfg['clustering']['pca_components']
    n_clusters = cfg['clustering']['n_components']
    
    logger.info(f"Fitting PCA (dim={pca_dim}) and GMM (K={n_clusters})...")
    
    pipeline = Pipeline([
        ('pca', PCA(n_components=pca_dim, random_state=42)),
        ('gmm', GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42))
    ])
    
    pipeline.fit(embeddings)
    
    # 5. Verification & Analytics
    labels = pipeline.predict(embeddings)
    soft_labels = pipeline.predict_proba(embeddings)
    
    # Silhouette Score
    # Note: silhouette_score can be slow on large datasets, so we might sample
    if embeddings.shape[0] > 10000:
        indices = np.random.choice(embeddings.shape[0], 10000, replace=False)
        sil_score = silhouette_score(embeddings[indices], labels[indices])
    else:
        sil_score = silhouette_score(embeddings, labels)
        
    logger.info(f"Cluster Analysis: Silhouette Score = {sil_score:.4f}")
    
    # 6. Visualization (t-SNE)
    logger.info("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    # Sample for visualization
    indices = np.random.choice(embeddings.shape[0], min(2000, embeddings.shape[0]), replace=False)
    embeddings_2d = tsne.fit_transform(embeddings[indices])
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels[indices], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'Acoustic Clusters (t-SNE) - K={n_clusters}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    viz_path = "outputs/plots/acoustic_clusters_tsne.png"
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    plt.savefig(viz_path)
    logger.info(f"Visualization saved to {viz_path}")
    
    # 7. Export
    export_path = cfg['clustering']['model_path']
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, "wb") as f:
        pickle.dump(pipeline, f)
        
    logger.info(f"GMM Registry successfully exported to {export_path}")

if __name__ == "__main__":
    run_clustering_pipeline()
