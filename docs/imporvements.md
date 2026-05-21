# Implementation Roadmap

## To-Do
- [x] GPU Parallelization (Environment Vectorization)
- [x] SB3 Hyperparameter Tuning (Learning Rate & Entropy Decay)
- [x] Generalization Testing (Complete Evaluation Script)
- [ ] Noise and Reverb Implementation (Acoustic Simulation)
- [x] Jitter and Spectral Tilt Refinement
- [x] Performance Bottleneck (Codec Optimization)
- [ ] Robust Unit Testing (pytest)

### 1. GPU Parallelization (Environment Vectorization)
* **Priority:** High
* **Syllabus Connection:** Directly aligns with the study of **Multi-actor architectures** (Actor-learners or Workers), parallel execution environments in RLlib/PettingZoo, and advanced parameter handling in Stable-Baselines3 (SB3).
* **Task:** Replace the sequential environment with a vectorized environment using `SubprocVecEnv` in SB3 to instantiate multiple parallel CPU processes. This allows multiple workers to simulate audio Digital Signal Processing (DSP) simultaneously, while the AASIST3 critic model performs inferences in optimized batches on the GPU.
* **Why it is high priority:** The program emphasizes practical algorithm optimization and problem-solving through efficient Python source code. By increasing the number of environments, the data volume per update is adjusted, accelerating convergence and allowing the agent to more robustly explore the "adversarial pockets" of the detector's KAN layers.

### 2. SB3 Hyperparameter Tuning (Learning Rate & Entropy Decay) [COMPLETED]
* **Priority:** High
* **Syllabus Connection:** Addresses the understanding of **Total Loss in PPO**, clipping control, and parameter adjustment in SB3.
* **Task:** Implemented a **Linear Learning Rate Scheduler** and a custom **EntropyDecayCallback**.
* **Why it is high priority:** In policy gradient algorithms like PPO, abrupt changes in network weights can irreversibly ruin the agent's policy. Learning rate and entropy decay ensure that, after broad initial exploration (high entropy), the agent stabilizes its learning (linear decay) and consolidates the winning policy without overshooting local optima.

### 3. Generalization Testing (Complete Evaluation Script) [COMPLETED]
* **Priority:** High
* **Syllabus Connection:** Directly fulfills the objectives of practical problem solving and the final report submission format, which requires observations, comments, and conclusions based on relevant code execution.
* **Task:** Developed `scripts/evaluate.py` to contrast the model's original metrics (Baseline) against the performance degradation caused by the agent's transformations. The script calculates Attack Success Rate (ASR) and Adversarial Equal Error Rate (EER) using the ASVspoof test split.
* **Why it is high priority:** Validating the policy against data unseen by the agent during training is fundamental to certify that the attack is generalizable and not merely overfitting to training samples. This completes the deductive methodological cycle required by the course.

### 4. Noise and Reverb Implementation (Acoustic Simulation)
* **Priority:** Medium-High
* **Syllabus Connection:** Relies on the understanding of **convolutional operations** and the simulation of stochastic environments or games.
* **Task:** Implement impulse response (IR) convolution and background noise injection modules. The agent should learn to control the signal-to-noise ratio (SNR) level to mask spoofing artifacts without destroying the underlying phonetic content.
* **Why it is medium-high priority:** These modules were omitted in the initial version. Their inclusion allows for a more rigorous assessment of detector robustness in adversarial partially observable environments.

### 5. Jitter and Spectral Tilt Refinement [COMPLETED]
* **Priority:** Medium
* **Syllabus Connection:** Aligns with the "step-by-step" resolved examples with corresponding pseudocode required for the challenge.
* **Task:** Implemented true temporal displacement ($x[t + \delta]$) for jitter using bilinear interpolation and a frequency-domain pivoting filter at 1 kHz for spectral tilt.
* **Why it is medium priority:** These adjustments ensure the project meets the high technical standards expected for a Master's degree specialization.

### 6. Performance Bottleneck (Codec Optimization) [COMPLETED]
* **Priority:** Medium
* [cite_start]**Syllabus Connection:** Relies on the practical-theoretical modality and the efficient use of Python resources[cite: 145, 147].
* [cite_start]**Implementation Impact:** Replaced slow `io.BytesIO` simulations with a **Differentiable Pseudo-Codec** implementing adaptive LPF and sub-band quantization. This keeps audio processing entirely in the tensor domain, significantly increasing training throughput without losing the specific artifacts targeted by AASIST3.

### 7. Robust Unit Testing (pytest)
* **Priority:** Low
* **Syllabus Connection:** Indirectly linked to the repository presentation and code transcription.
* **Task:** Create a suite of automated tests to verify audio tensor alignment and reward calculation consistency.
* **Why it is low priority:** While excellent for general programming skill improvement, it is not a core conceptual pillar of the RL II theory evaluated in the syllabus.