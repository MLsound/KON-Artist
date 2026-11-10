# Improvement phase
## FEAT 1: High-Throughput Vectorized Pipeline & Configurable Training Refactor **✓**

This document unifies the architectural specifications and implementation notes for the transition from a sequential environment execution model to a parallelized, batched-inference pipeline. The refactor decouples CPU-bound Digital Signal Processing (DSP) and data ingestion from GPU-bound spoofing detector inference, optimizing throughput and ensuring experimental reproducibility.

## 1. Objective
The primary bottleneck in the prototype architecture was GPU underutilization caused by sequential execution, where the GPU remained idle while single CPU processes completed audio transformations and dataset streaming.

This refactor centralizes inference for the AASIST3 detector to maximize GPU batch efficiency, leverages multiprocessing to parallelize environment instances, and externalizes hyperparameter management to establish a production-ready, reproducible research pipeline.

## 2. Component Architecture & Ingestion Pipeline
### AASISTWrapper (`aasist.py`)
* **Batched Inference:** Enhanced `get_score_and_embedding` to accept input tensors of shape `[B, 1, L]`. The wrapper automatically squeezes these to `[B, L]` to match backbone configuration requirements.
* **Embedding Extraction:** Integrated forward pre-hooks to reliably capture high-dimensional embeddings from the backbone, serving as state representations for downstream Reinforcement Learning (RL) agents.

### AudioAttackEnv (`audio_attack.py`)
* **Multiprocessing Stability:** Implemented **lazy stream initialization** within `reset()`. Deferring Hugging Face dataset loading until the worker process explicitly initializes prevents pickling and serialization errors within `SubprocVecEnv`.
* **Decoupled Mode:** Enabled execution with `detector=None`. This allows individual environment workers to act as raw audio providers, handing off downstream inference tasks to a centralized batch coordinator.
* **Pure-Torch ETL:** Optimized data ingestion by streaming raw bytes directly into `torchaudio.load()`. This maintains audio phase integrity and eliminates intermediate disk-write operations, reducing memory overhead.

### VecDetectorWrapper (`wrappers.py`)
* **GPU Batching:** A custom Stable-Baselines3 (SB3) `VecEnvWrapper` that intercepts raw waveforms from parallel workers, stacks them, and executes a single GPU forward pass for the entire batch.
* **Observation Mapping:** Centralizes the transformation of raw audio into RL-ready embeddings, isolating the environment workers from deep learning framework overhead.

## 3. Centralized Reward Logic & Mathematical Consistency
To ensure strict mathematical parity between single-process debugging and high-throughput vectorized training, all evaluation metrics were isolated into a dedicated module.

### Reward Formulation (`reward_logic.py`)
The reward shaping mechanism provides a dense gradient signal based on the detector's output probability.

```latex
R = \log(P + \epsilon) + 25.0
```

Where $P$ is the detector's target classification score or probability, and $\epsilon$ is a small constant to prevent logarithmic divergence.

* **Success Bonus:** An additional flat additive bonus of $+50.0$ is awarded immediately when classification scores cross the $0.5$ threshold.
* **Telemetry Interoperability:** Utilizes Python's `getattr` primitives to dynamically extract runtime metrics from either a standalone `AudioAttackEnv` or a vectorized `VecDetectorWrapper`.

> **Architectural Note:** While `getattr` duck-typing provides interface flexibility, ensure explicit default fallbacks are defined within `reward_logic.py` to prevent missing telemetry attributes from causing silent runtime failures during environment step evaluations.

## 4. Reproducibility & Configuration Management
### Externalized Configuration (YAML)
Hardcoded variables have been extracted from execution scripts into a centralized file: `configs/train_config.yaml`.

* **Parameter Scope:** Manages PPO hyperparameters, environment dimensions, model paths, and logging metadata.
* **Dynamic Auto-Scaling:** The orchestration script performs host CPU-core detection to automatically scale the number of parallel `SubprocVecEnv` workers to match local hardware availability.

### Deterministic Search
* **Worker-Specific Seeding:** Implemented a rank-based seed distribution strategy across parallel environments. This guarantees deterministic sample generation and reproducible adversarial trajectories, which is essential for thesis validation.

## 5. Lifecycle Management & Telemetry
### Training Resumption & Checkpointing
To mitigate long-term training risks, a fail-safe execution loop was established:

* **Automated Checkpointing:** Employs SB3’s `CheckpointCallback` to serialize full model states, covering network weights, optimizer states, and current learning rate schedules at specified step intervals.
* **Seamless Resumption:** The training loop automatically scans for the latest checkpoint upon initialization, restoring the global step counter and schedule states without configuration adjustments.

### Enhanced Telemetry & Observability

* **Worker Isolation:** Logs explicitly append worker identifiers (`Worker #`) to trace behavioral variance across parallel environments.
* **Progress Tracking:** Replaced static status markers with dynamic execution tracking metrics, showing real-time completion percentages and precise step tracking (`Current / Total`).


---
## FEAT 2: Policy Optimization & Strategic Exploration Tuning **✓**

This phase introduced advanced optimization schedules and exploration controls to enhance the agent's ability to "settle" into high-fidelity adversarial pockets while maintaining mathematical stability during long-term training.

### 1. Adaptive Optimization Schedules
To improve convergence stability and prevent overshooting in the late stages of training:
* **Linear Learning Rate Annealing**: Implemented a `linear_schedule` function in `scripts/train.py` that decays the learning rate from 3e-4 toward 0 over `TOTAL_TIMESTEPS`. This prevents abrupt weight updates as the policy matures.
* **Vectorized Scaling Logic**: Refined PPO scaling logic to ensure `BATCH_SIZE` is always valid relative to the scaled rollout buffer (N_envs x N_steps), ensuring consistent gradient quality across different hardware.

### 2. Strategic Exploration via Entropy Decay
To balance discovery of new manifolds with solidification of adversarial wins:
* **EntropyDecayCallback**: Developed a specialized callback in `src/utils/callbacks.py` that manages the entropy coefficient (beta).
* **Exploration vs. Exploitation**: The schedule reduces beta from 0.01 (high exploration) to 0.001 (high exploitation), forcing the policy to converge on high-confidence parameters as training concludes.

### 3. Training Robustness & Documentation
Standardized the internal logic and tracked progress against the project roadmap:
* **Naming & Logic Consistency**: Consolidated variable naming to use `BATCH_SIZE` consistently across the codebase and ensured metric continuity during checkpoint resumption.


---
## FEAT 3: Evaluation Framework & Generalization Testing

This phase shifted the project focus from training stability to rigorous performance validation. The primary objective was to design and implement an automated assessment pipeline to quantify the adversarial impact of the **KON-Artist** agent against the **AASIST3** detector, benchmarking baseline robustness against iterative DSP modifications on unseen data from the ASVspoof 2019 test split.

### 1. Robust Evaluation Infrastructure
Developed a dedicated validation orchestrator to automate large-scale inference and capture standardized performance degradation:

* **`scripts/evaluate.py`**: A new entry point that manages the loading of trained policies and executes batched evaluation across diverse acoustic environments.
* **Standardized Metrics**: Integrated calculations via `src/utils/metrics.py` adhering strictly to ASVspoof challenge protocols:
* **Equal Error Rate (EER)**: Computes and compares baseline EER against post-attack EER to measure the direct "degradation of trust" in the detector.
* **Attack Success Rate (ASR)**: Quantifies the percentage of synthetic samples that successfully cross the 0.5 decision boundary post-attack.

* **Optimal Roll-out Execution**: Grants the agent up to 10 interaction steps per evaluation sample (matching training limits) to iteratively refine DSP parameters for maximum adversarial impact.

### 2. Unbiased Data Ingestion Pipeline
Refactored the data streaming system to support statistically valid metric calculations without introducing distribution bias:

* **Eval Generator**: Introduced `eval_generator` within `src/data/loader.py` to handle unbiased sampling of both **Bonafide** and **Spoof** classes. This un-filtered distribution is mathematically required for accurate EER calculation.
* **Stream Integrity**: Maintained the pure-Torch streaming architecture to ensure that evaluation audio is processed with the exact same phase and sample-rate fidelity as training data, avoiding synthetic processing artifacts.

### 3. Environment Propagation & Compliance Tracking
Enhanced internal state tracking and logging to guarantee academic reproducibility and enable deep generalization analysis:

* **Label and State Transparency**: Modified `AudioAttackEnv.reset` to propagate ground-truth labels and initial detector scores inside the `info` dictionary. This enables the evaluation framework to map the exact "before and after" delta in detector confidence.
* **Session Persistence**: Implemented detailed session logging to `outputs/eval_session.log`, capturing the full trace of step-by-step agent decisions, environment transitions, and intermediate metric outputs.
* **Generalization Analysis**: The architecture fully supports testing across different ASVspoof subsets, enabling critical assessment of whether the learned attacks exploit sample-specific artifacts or expose systemic, generalized vulnerabilities in **KAN/GAT** architectures.

---
## FEAT 4


---
## FEAT 5: Advanced DSP Pipeline & Stochastic Signal Refinement
This phase focused on increasing the acoustic fidelity and technical rigor of the agent's action space. The objective was to transition from simple additive approximations to physically-grounded audio transformations.

### 1. True Jitter Implementation (apply_jitter)
Implemented a high-precision temporal displacement mechanism to exploit GAT's dependency on periodic acoustic patterns:
* **Mechanism**: Replaced the previous additive noise approach with true temporal displacement $x[t + \delta]$.
* **Technical Implementation**: Utilizes `torch.nn.functional.grid_sample` with bilinear interpolation to shift audio samples stochastically on a sub-sample level. This creates authentic timing artifacts (jitter) without injecting unrelated amplitude noise.

### 2. Pivoting Spectral Tilt (apply_spectral_tilt)
Refined the spectral modification layer to allow for more nuanced frequency-domain "model blinding":
* **Mechanism**: Implemented a frequency-domain gain ramp that pivots specifically around 1 kHz.
* **Technical Implementation**: Uses `torch.fft.rfft` to calculate a precise gain curve based on a requested $dB/octave$ slope. Frequencies above 1 kHz are boosted/cut while frequencies below are inversely modified, ensuring overall energy is balanced around the pivot point.

### 3. Precision Signal Modeling & Cleanup
Standardized the synthesis pipeline for long-term optimization and academic clarity:
* **Code Refinement**: Updated `apply_jitter_shimmer` to utilize the new refined jitter logic, ensuring sub-sample precision during multi-step agent roll-outs.


---
## FEAT 6: Codec Optimization & DSP Pipeline Finalization **✓**
This phase resolved the primary performance bottleneck in the training pipeline by replacing file-based encoding with a high-performance differentiable simulation, while ensuring the persistence of all previous acoustic refinements.

### 1. Differentiable Pseudo-Codec (apply_codec_simulation)
To eliminate the serial latency of the `io.BytesIO` and `torchaudio.save/load` loop, a tensor-domain approximation was implemented:
* **Optimization**: Removed the expensive CPU/GPU data transfers and file-encoding overhead. The entire synthesis chain now resides within the differentiable PyTorch environment.
* **Bitrate Simulation Mechanism**:
    * **Adaptive LPF**: Dynamically scales the cutoff frequency of a low-pass filter to simulate the bandwidth limiting of low-bitrate codecs (e.g., MP3/Opus).
    * **Sub-band Quantization**: Mimics spectral compression noise and precision loss using a differentiable bit-depth reduction strategy.
* **Performance Impact**: Achieved a significant increase in Frames Per Second (FPS) and GPU utilization, allowing for faster RL convergence and large-scale architectural testing.

### 2. Algorithm Persistence & Technical Rigor
Ensured that technical advancements from previous phases remain central to the optimized pipeline:
* **Refined Jitter**: Re-validated the inclusion of "True Jitter" (stochastic temporal displacement via bilinear interpolation) within the new tensor-only architecture.
* **Spectral Tilt**: Confirmed the FFT-based pivoting gain ramp (centered at 1 kHz) remains mathematically valid and efficient.

---
## FEAT 7


---

▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
 > How could evaluation be improved? any relevant metrics could be added?                                       
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

  Researching Evaluation Improvements: 
  Analyze current evaluation metrics and research industry-standard improvements for audio spoofing
  detection.

  ✓  GoogleSearch  Searching the web for: "latest metrics for audio spoofing detection ASVspoof 2024 SASV t…
   → Search results for "latest metrics for audio spoofing detection ASVspoof 2024 SASV t-DCF EER EMD MOS" …

✦ To improve the evaluation of KON-Artist, we should transition from simple classification accuracy to metrics
  that measure security risk, perceptual quality, and distributional shifts.

  The ASVspoof 2024 challenge recently shifted the industry standard away from EER toward cost-based metrics
  like a-DCF. For KON-Artist, where we are actively attacking a model, we should also track how much the audio
  is "drifting" away from human speech.

  1. Recommended Security Metrics
   * Agnostic Detection Cost Function (a-DCF): The current ASVspoof 2024 primary metric. Unlike EER (which
     treats every error as equal), DCF assigns higher costs to security failures (letting a spoof through) vs.
     usability failures (blocking a real human).
   * Cost of Log-Likelihood Ratio (Cₗₗᵣ): This measures how well-calibrated the detector's scores are. An attack
     that doesn't just fool the model but makes it extremely confident in its wrong answer will result in a
     higher Cₗₗᵣ.
   * ASR @ Fixed FPR: Instead of just saying the attack succeeded, we should measure the "Attack Success Rate"
     at a very strict False Positive Rate (e.g., ASR at 1% FPR). This tells us if the attack is robust enough to
     bypass a detector set to high security.

  2. Recommended Perceptual & Quality Metrics
  One of the biggest risks in RL-based attacks is "reward hacking," where the agent creates noise that fools the
  model but sounds nothing like speech.
   * Virtual MOS (Mean Opinion Score): Use a pre-trained neural MOS estimator (like NISQA or DNSMOS) to ensure
     the adversarial audio maintains a high quality score (MOS > 3.0).
   * Character Error Rate (CER) / Word Error Rate (WER): Pass the modified audio through an ASR model (like
     Whisper). If the attack succeeds but the speech becomes unintelligible, the attack is practically useless
     for deepfaking.
   * NISQA-TTS: A specific model for estimating the naturalness of synthetic speech, which is highly effective
     at identifying the "robotic" artifacts RL agents often introduce.

  3. Distributional Metrics
   * Earth Mover's Distance (EMD) / Wasserstein Distance: Measure the distance between the embedding
     distribution of Bona Fide samples and Adversarial samples. A "perfect" attack would have an EMD near zero,
     meaning the adversarial samples are indistinguishable from real ones in the model's own latent space.
   * Maximum Mean Discrepancy (MMD): Similar to EMD, this helps detect if the RL agent has found a "manifold"
     that the detector hasn't seen during training, indicating a lack of generalization.

  Summary Table for Implementation

  ┌─────────────┬─────────────────┬──────────────────────────────────────────────────────────┐
  │ Metric      │ Category        │ Purpose                                                  │
  ├─────────────┼─────────────────┼──────────────────────────────────────────────────────────┤
  │ min a-DCF   │ Security        │ Measure the actual security risk of the attack.          │
  │ DNSMOS /    │ Quality         │ Ensure the attack remains "human-sounding."              │
  │ NISQA       │                 │                                                          │
  │ Whisper CER │ Intelligibility │ Verify that the "message" in the audio is preserved.     │
  │ Latent EMD  │ Distribution    │ Measure how "deeply" the attack penetrates the bona fide │
  │             │                 │ distribution.                                            │
  └─────────────┴─────────────────┴──────────────────────────────────────────────────────────┘

  Implementing a "Quality-Constrained Reward" (e.g., Reward = Score - λ · (MOSᵢₙᵢₜᵢₐₗ - MOS_{current})) would be
  the most significant improvement for the KON-Artist training loop, preventing the agent from "obliterating"
  the detector by simply introducing high-frequency noise.
