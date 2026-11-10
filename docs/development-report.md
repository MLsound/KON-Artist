# KON-Artist Development Report: Implemented Features

This development report documents all new features, optimizations, and infrastructure upgrades successfully implemented in the **KON-Artist** adversarial reinforcement learning framework. 

---

## 1. High-Throughput Vectorized Environment & GPU Batching Pipeline

To overcome CPU bottlenecks where the GPU remained idle during sequential audio processing, a vectorized execution pipeline was implemented to decouple environment simulation from detector inference:

*   **Centralized GPU Inference ([VecDetectorWrapper](file:///Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/RL/KON-Artist/src/env/wrappers.py)):** Intercepts raw waveforms from multiple parallel workers, stacks them into a single tensor batch of shape `[B, 1, L]`, and performs a single GPU forward pass. This isolates heavy deep learning overhead from the workers.
*   **Vectorized Worker Management (`SubprocVecEnv`):** Instantiates parallel CPU environments using Python multiprocessing, running audio loading and signal processing concurrently.
*   **Lazy Dataset Initialization ([AudioAttackEnv.reset](file:///Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/RL/KON-Artist/src/env/audio_attack.py#L155)):** Defers Hugging Face dataset loading until the individual worker process calls `reset()`, resolving pickling, serialization, and connection leaks in multiprocessing.
*   **Deterministic Seeding:** Implemented rank-based seed distribution (`seed + rank`) across worker processes to ensure non-overlapping, reproducible sample streams from the dataset.

---

## 2. Differentiable Pseudo-Codec Simulation

To eliminate the latency of file-based codec operations (which required disk writes, `io.BytesIO` buffering, and CPU/GPU memory copying), a tensor-domain codec approximation was implemented:

*   **Differentiable Low-Pass Filter (LPF):** Simulates bandwidth restriction in low-bitrate compression (e.g., MP3 or Opus) using an adaptive cutoff frequency that dynamically scales down with lower bitrates.
*   **Differentiable Sub-Band Quantization:** Mimics compression noise and precision loss using a Straight-Through Estimator (STE). The rounding operation is decoupled from the gradient graph (`x = x + (x_quant - x).detach()`), allowing backpropagation gradients to flow back to the policy network.
*   **Performance Impact:** Achieved a massive increase in frames-per-second (FPS) throughput and GPU utilization, enabling rapid RL policy exploration.

---

## 3. Advanced DSP Pipeline & Stochastic Signal Refinement

The action space of [AudioAttackEnv](file:///Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/RL/KON-Artist/src/env/audio_attack.py) maps continuous RL control outputs (normalized in `[-1, 1]`) to physically grounded audio transformations in [dsp.py](file:///Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/RL/KON-Artist/src/synthesis/dsp.py):

*   **True Jitter via Bilinear Interpolation (`apply_jitter`):** Implements temporal micro-fluctuations $x[t + \delta]$ by generating sample-specific stochastic offsets and applying `torch.nn.functional.grid_sample` to resample the audio grid, rather than using simple additive noise approximations.
*   **Pivoting Spectral Tilt (`apply_spectral_tilt`):** Implements a frequency-domain gain ramp pivoting around 1 kHz using PyTorch's Fast Fourier Transform (`torch.fft.rfft`). Frequencies above 1 kHz are boosted or cut while low frequencies are inversely scaled, preserving absolute signal energy.
*   **Precision Shimmer (`apply_jitter_shimmer`):** Simulates stochastic amplitude modulation by scaling the audio amplitude dynamically with normal noise vectors.

---

## 4. Policy Schedules & Exploration Tuning

To stabilize policy search and prevent the policy from collapsing in late-stage convergence, custom schedules were integrated with Stable-Baselines3 (SB3):

*   **Linear Learning Rate Annealing:** Progressively decays the learning rate from initial bounds (e.g., `1e-4` or `3e-4`) toward 0 over the course of training, ensuring weight updates stabilize as the policy matures.
*   **Entropy Decay Callback ([EntropyDecayCallback](file:///Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/RL/KON-Artist/src/utils/callbacks.py#L134)):** Dynamically scales down the entropy coefficient (exploration coefficient) from `0.05` to `0.005` to force the model to exploit high-confidence winning parameters in the final training deciles.
*   **Entropy Decay Modes:** Added support for:
    *   `global`: Decays entropy relative to the total training step target.
    *   `cycle`: Decays relative only to the current execution run/cycle (critical for fine-tuning resumed checkpoints).

---

## 5. Session-Aware Training Resumption & Logging

*   **Seamless Resumption:** The training orchestrator reads PPO checkpoints, extracts the genuine global steps completed, and scales the training loop to run only the remaining `pending_timesteps = TOTAL_TIMESTEPS - completed_steps`.
*   **Session-Aware Callbacks:** The callbacks (`RewardLoggerCallback`, `WandbAudioCallback`, etc.) compute metrics and track completion percentages (`session_progress_pct`) based on the active training cycle, preventing progress bars from resetting to 0% upon restart.
*   **Unified Hyperparameter Summary:** Prints a single, structured log at startup containing `TOTAL_TIMESTEPS`, `N_STEPS`, `TOTAL_UPDATES`, `EPOCHS`, and `BATCH_SIZE` for tracking.

---

## 6. Proportional Reward Shaping & Dataset Security

*   **Standardized Log-Probability Base:** Computes baseline reward using the log-probability of the detector classifying the sample as human: $R = \log(P + \epsilon) + 25.0$. The log scale prevents vanishing gradients when scores are very close to zero.
*   **Continuous Proportional Bonus:** Replaced the discrete milestone steps (+50 / +100) with a continuous linear scaling bonus ($P \times \text{bonus\_amount}$), creating a smooth gradient slope.
*   **Label Contamination Protection:** Implemented an validation check in `AudioAttackEnv.reset` that aborts training immediately if a true "Bonafide" human sample (label `1`) is loaded. This guarantees the attack pipeline only mutates "Spoof" audio (label `0`) to protect genuine voice assets.

---

## 7. Generalization Evaluation Framework

A complete evaluation pipeline was created to benchmark baseline robustness against adversarial changes on unseen splits:

*   **Validation Orchestrator ([evaluate.py](file:///Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/RL/KON-Artist/scripts/evaluate.py)):** Runs a trained PPO policy against ASVspoof 2019 test data.
*   **Standardized Security Metrics ([metrics.py](file:///Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/RL/KON-Artist/src/utils/metrics.py)):** Calculates Equal Error Rate (EER) and Minimum Detection Cost Function (min t-DCF).
*   **ASR Quantification:** Computes Attack Success Rate (ASR) mapping the percentage of spoofed files successfully flipped past the `0.5` decision threshold.
*   **Evaluation Data Streamer (`eval_generator`):** Unbiased data loader in [loader.py](file:///Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/RL/KON-Artist/src/data/loader.py) that streams both genuine (Bonafide) and synthetic (Spoof) classes for unbiased EER validation.

---

## 8. Robust Unit & Integration Test Suite

A complete test suite was established under [tests/](file:///Users/alejandrolloveras/Documents/ESTUDIO/UBA/Materias/RL/KON-Artist/tests/) to verify codebase functionality and protect against regressions:

1.  `test_aasist.py`: Validates batched forward passes and hook-based embedding extraction.
2.  `test_audio.py`: Checks raw audio preprocessing, cropping, and padding.
3.  `test_audio_attack_env.py`: Tests env state resets, step logic, and reward calculation.
4.  `test_callbacks.py`: Asserts reward recording, W&B serialization, and learning rate logging.
5.  `test_dsp.py`: Verifies mathematical correctness of LPF, quantization, FFT-tilt, and bilinear-jitter.
6.  `test_evaluate.py`: Tests EER/DCF calculation scripts and output saving.
7.  `test_integration.py`: Confirms end-to-end PPO step and DSP wave mutation cycles.
8.  `test_metrics.py`: Tests the SKLearn-based ROC curves and cost functions.
9.  `test_train.py`: Asserts training resumption, config parsing, and hyperparameter logs.
