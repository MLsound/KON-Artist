"""
Offline Acoustic Clustering Pipeline for KON-Artist.

This script extracts high-dimensional embeddings from the AASIST3 model
for the entire ASVspoof 2019 LA training split (spoofed samples),
reduces their dimensionality via PCA, and fits a Gaussian Mixture Model (GMM)
to identify acoustic sub-manifolds.
"""

import os

# SET OMP_NUM_THREADS=1 BEFORE ANY SCIPY/NUMPY/TORCH IMPORTS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import numpy as np
import torch
import warnings
from tqdm import tqdm

# Suppress the specific FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
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

def extract_all_embeddings(detector, loader, device="cuda", batch_size=64, max_samples=10000):
    """
    Extracts embeddings for spoofed samples in the loader using batched inference.
    Limits processing to max_samples to ensure representative but fast manifold identification.
    """
    embeddings = []
    logger.info(f"Starting batched embedding extraction (batch_size={batch_size}, max_samples={max_samples})...")
    
    waveforms_batch = []
    total_extracted = 0
    
    with torch.no_grad():
        pbar = tqdm(total=max_samples, desc="Extracting embeddings")
        for waveform, label in generator_from_ds(loader):
            # waveform is already [1, L] from generator_from_ds
            waveforms_batch.append(waveform)
            
            # If batch is full or we reached max_samples, process it
            if len(waveforms_batch) == batch_size or (total_extracted + len(waveforms_batch)) >= max_samples:
                # Handle cases where we might overshoot max_samples in the last batch
                current_batch_size = len(waveforms_batch)
                if (total_extracted + current_batch_size) > max_samples:
                    needed = max_samples - total_extracted
                    waveforms_batch = waveforms_batch[:needed]
                    current_batch_size = needed
                
                # Stack waveforms into [B, 1, L] for AASISTWrapper
                batch_tensor = torch.cat(waveforms_batch, dim=0).unsqueeze(1).to(device)
                
                # AASISTWrapper handles batched [B, 1, L]
                _, embedding = detector.get_score_and_embedding(batch_tensor)
                embeddings.append(embedding.cpu().numpy())
                
                total_extracted += current_batch_size
                pbar.update(current_batch_size)
                waveforms_batch = []
                
            if total_extracted >= max_samples:
                break
        
        pbar.close()
            
    # Concatenate list of batch embeddings into a single [N, D] array
    return np.vstack(embeddings).astype(np.float32)

def optimize_clustering_params(embeddings):
    """
    Analyzes embeddings to find optimal PCA and GMM configurations.
    """
    logger.info("Optimizing clustering parameters...")
    
    # 0. Pre-scaling for stability
    scaler = StandardScaler().fit(embeddings)
    scaled_embeddings = scaler.transform(embeddings)
    
    # 1. Optimize PCA
    pca = PCA().fit(scaled_embeddings)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find components needed for 95% variance
    optimal_pca = int(np.argmax(cumulative_variance >= 0.95) + 1)
    logger.info(f"Optimal PCA components (95% variance): {optimal_pca}")
    
    # Apply optimal PCA for the GMM test
    reduced_embeddings = PCA(n_components=optimal_pca).fit_transform(scaled_embeddings)
    
    # 2. Optimize GMM (Testing K=2 to K=10)
    n_components_range = range(2, 11)
    bic_scores = []
    
    for k in n_components_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(reduced_embeddings)
        bic_scores.append(gmm.bic(reduced_embeddings))
        
    optimal_k = int(n_components_range[np.argmin(bic_scores)])
    logger.info(f"Optimal GMM components (Lowest BIC): {optimal_k}")
    
    # Plotting for visual confirmation
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_variance)
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.title("PCA Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    
    plt.subplot(1, 2, 2)
    plt.plot(n_components_range, bic_scores, marker='o')
    plt.title("GMM BIC Scores")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("BIC Score")
    
    viz_dir = "outputs/plots/"
    os.makedirs(viz_dir, exist_ok=True)
    plt.savefig(os.path.join(viz_dir, "clustering_optimization.png"))
    logger.info(f"Optimization plots saved to {viz_dir}clustering_optimization.png")
    
    return optimal_pca, optimal_k

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
    # Using optimized parameters: batch_size=64 for GPU throughput, max_samples=10000 for manifold representation
    embeddings = extract_all_embeddings(
        detector, 
        loader, 
        device=device, 
        batch_size=64, 
        max_samples=10000
    )
    logger.info(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")
    
    # 4. Parameter Optimization
    pca_dim, n_clusters = optimize_clustering_params(embeddings)
    
    # 5. Fit Scaling + PCA + GMM Pipeline
    logger.info(f"Fitting final Pipeline with Scaler, PCA (dim={pca_dim}) and GMM (K={n_clusters})...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_dim, random_state=42)),
        ('gmm', GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42))
    ])
    
    pipeline.fit(embeddings)
    
    # 6. Verification & Analytics
    labels = pipeline.predict(embeddings)
    
    # Cluster Distribution
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    logger.info(f"Cluster Distribution: {dist}")
    
    # Silhouette Score
    # Note: silhouette_score can be slow on large datasets, so we might sample
    if embeddings.shape[0] > 10000:
        indices = np.random.choice(embeddings.shape[0], 10000, replace=False)
        sil_score = silhouette_score(embeddings[indices], labels[indices])
    else:
        sil_score = silhouette_score(embeddings, labels)
        
    logger.info(f"Cluster Analysis: Silhouette Score = {sil_score:.4f}")
    
    # 7. Visualization (t-SNE)
    logger.info("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    # Sample for visualization
    indices = np.random.choice(embeddings.shape[0], min(2000, embeddings.shape[0]), replace=False)
    # We must scale before t-SNE to match GMM input manifold
    scaled_emb_sample = pipeline.named_steps['scaler'].transform(embeddings[indices])
    embeddings_2d = tsne.fit_transform(scaled_emb_sample)
    
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
    
    # 8. Export
    export_path = cfg['clustering']['model_path']
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, "wb") as f:
        pickle.dump(pipeline, f)
        
    logger.info(f"GMM Registry successfully exported to {export_path}")

if __name__ == "__main__":
    run_clustering_pipeline()
