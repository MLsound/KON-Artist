# Acoustic-Conditioned Policy (ACP) Development Report
**Framework:** KON-Artist (Adversarial Spoofing Attacks on AASIST3)
**Base Commit Reference:** 694f8cd4c8d7ccafabc37b77ac15cd3095a9986d

---

## 1. Executive Summary

This report evaluates the implementation of the Acoustics-Conditioned Policy (ACP) framework in the KON-Artist codebase. The transition from a context-blind policy to an ACP framework represents a paradigm shift. Rather than forcing the Proximal Policy Optimization (PPO) agent to locate a single global set of Digital Signal Processing (DSP) parameters, the policy conditional path is conditioned on unsupervised acoustic sub-manifolds (GMM clusters). 

This document compares the design directives found in `docs/notes/` against the actual Git commit logs. It maps code additions, identifies omissions or modifications of planned features, highlights experimental performance gains, and analyzes the methodological benefits of these changes for the final Master's thesis defense.

---

## 2. Chronological Git Commit History Analysis

Since the base commit `694f8cd` (Merge branch 'prior-knowledge' into dev), the codebase underwent rapid iterations to support acoustic clustering, action biasing, reward decoupling, boundary squashing, and telemetry tracking.

### Phase 1: Acoustic Clustering Foundations
* **39341f4 & 1bbc0c9 (Add acoustic clustering and GMM integration):** Introduced the initial unsupervised GMM pipeline. Calculated soft-assignment cluster probabilities based on PCA-reduced AASIST3 speech embeddings.
* **19ccb16 & f012bf8 (Add cluster-aware rewards):** Linked environment returns to cluster identities. Enabled translation biases to shift initial action domains.
* **ddc10c0 & 76276cf (Automated clustering pipeline):** Standardized optimal GMM components selection and visualization.

### Phase 2: Action Biasing & Metric Decoupling
* **4c958be & f336edf (Cluster-conditioned biasing & boundary penalties):** Intercepted actions to inject GMM centroid biases. Restricted PPO exploration grids using soft boundaries.
* **5e95e25 (Decouple cluster multipliers):** Refactored reward shaper to apply inverse-frequency multipliers exclusively to the success bonus vector, resolving artificial reward inflation for rare clusters (e.g., Cluster 9).

### Phase 3: Acoustic Drift Mitigation & Context Caching
* **3f312ea (Contextual cluster filtering & dynamic noise):** Added worker resetting for excluded clusters (Cluster 5) and environment-level action noise modulation based on per-cluster Attack Success Rate (ASR).
* **98278bc & adba176 (Invariant context caching & GMM exclusion):** Fixed a critical gradient leakage bug (Acoustic Drift). Implemented caching of the GMM probability vector at step 0 (`reset()`) and reuse of this static context throughout the remaining 10 steps of the trajectory.

### Phase 4: Mathematical Optimization & Boundary Corrections
* **1842dad & 4efcb60 (Logit-aware reward & stagnation handling):** Replaced linear rewards with raw detector logits to eliminate flat gradient zones. Established a piecewise reward function: Stage A (log-sigmoid for exploration) and Stage B (log-probability + capped tanh success bonus).
* **7147010 (Squashed Gaussian policy):** Replaced standard Gaussian distributions with a custom Squashed Gaussian policy layer applying a change-of-variables Jacobian determinant correction. This resolved vanishing gradients near boundary limits.

### Phase 5: Search Space Constraints & Telemetry
* **b7a09db (Adaptive Action-Space Clipping Wrapper):** Implemented a custom wrapper to sequentially compress search boundaries toward cluster biases when a cluster's empirical ASR breaches the compression threshold.
* **e228268 & be6489e (ClusterManifoldAlignmentCallback):** Integrated a callback to monitor action trajectory convergence toward the optimal vulnerability target plane `[-1.0, 1.0]`.

---

## 3. Comparison & Gap Identification

| Feature Planned in Notes | Actual Implementation Status | Codebase Location / Details |
| --- | --- | --- |
| **Cluster 5 Splitting & Exclusion** | **Fully Implemented** | Handled recursively during reset in `VecDetectorWrapper` and `AudioAttackEnv` to completely isolate the rollout buffer from Cluster 5 experiences. |
| **Static Trajectory Caching** | **Fully Implemented** | Caches probabilities at `reset()` and concatenates them with embeddings. Disables mid-trajectory GMM evaluations. |
| **Decoupling Multipliers** | **Fully Implemented** | `inverse_frequency_multipliers` scaled only during success checks in `src/env/reward_logic.py`. |
| **Beta Distribution Policy (Option A)** | **Omitted / Superseded** | Not implemented. The codebase opted for Option B (Squashed Gaussian policy) due to better integration compatibility with SB3 default networks. |
| **Squashed Gaussian Policy (Option B)** | **Fully Implemented** | Custom distribution class implemented in `src/models/custom_policies.py` with analytical log-likelihood Jacobian corrections. |
| **Log-Sigmoid Logit Rewards** | **Fully Implemented** | Sigmoid logit mapping applied for negative logits ($logit < 0$) in `src/env/reward_logic.py`. |
| **Dynamic Noise Scaling** | **Fully Implemented** | Contextual noise scale $\omega_k$ applied to PPO actions in `VecDetectorWrapper.step_async`. |
| **Adaptive Action-Space Clipping** | **Fully Implemented** | `AdaptiveActionSpaceClipsWrapper` dynamically contracts action space intervals towards `cluster_biases` as ASR rises. |
| **Manifold Shift Callback** | **Fully Implemented** | `ClusterManifoldAlignmentCallback` tracks distance to target plane `[-1.0, 1.0]` over rollout boundaries. |

### Key Differences & Design Decisions:
1. **Omission of the Beta Distribution Policy:** 
   Although a Beta distribution policy was proposed in `acp-features.md` as an alternative to prevent boundary gradient saturation, it was bypassed. The developers implemented a **Squashed Gaussian Policy with analytical log-prob correction**. This is mathematically equivalent in preventing vanishing gradients near boundary limits while maintaining structural compatibility with Stable-Baselines3's baseline networks.
2. **Two-Layer Exploration Controls:** 
   The notes present Dynamic Noise Scaling and Adaptive Action-Space Clipping as alternative strategies. However, the codebase implements both:
   * **Dynamic Noise Scaling** acts globally on the main process to modulate exploration entropy based on rolling ASR.
   * **Adaptive Action-Space Clipping** acts locally in the worker environments to compress search intervals around the GMM biases baseline.
   This two-layer design enforces stochastic exploration on resilient clusters while locking onto successful parameters in compromised clusters.

---

## 4. Codebase Implementation Architecture

### 1. Invariant Context Caching (Acoustic Drift Mitigation)
Acoustic Drift occurs when intermediate DSP transformations shift the acoustic features of a sample into a different GMM bucket mid-episode. If this bucket is Cluster 5, the policy network updates its shared weights on Cluster 5 features during backpropagation, causing **Gradient Leakage**.

To prevent this, `VecDetectorWrapper.reset()` and `VecDetectorWrapper.step_wait()` cache the initial cluster distribution vector and reuse it. The GMM pipeline is evaluated only on raw, unperturbed audio:
```python
# Caching during reset:
self.current_trajectory_cluster_probs = cond_obs[:, 160:].astype(np.float32)
for i in range(self.num_envs):
    self.venv.set_attr("current_trajectory_cluster_probs", self.current_trajectory_cluster_probs[i], indices=i)

# Concatenation during steps (suppressing intermediate GMM prediction):
new_cond_obs[i] = np.concatenate([embeddings_np[i], self.current_trajectory_cluster_probs[i]])
```

### 2. Piecewise Log-Sigmoid Evasion Rewards
To prevent flat zero-gradient spaces when classification probability drops to exactly 0.0000, `compute_attack_reward` implements logit-aware piecewise reward mapping:
* **Stage A (Exploration Phase, $logit < 0$):**
  $$R_{\text{base}} = \log(\text{sigmoid}(\text{logit}) + 10^{-9}) + 25.0$$
* **Stage B (Exploitation Phase, $logit \ge 0$):**
  $$R_{\text{base}} = \log(\text{score} + 10^{-9}) + 25.0$$

This configuration guarantees continuous, informative gradients across highly resistant populations.

### 3. Custom Squashed Gaussian Policy
The DiagGaussian policy assumes an infinite, unbounded action space. When continuous actions are clamped, the policy gradients vanish near the limits. The `SquashedGaussianDistribution` applies a `tanh` squash and corrects the log-probability calculation using the change-of-variables formula:
```python
clipped_actions = torch.clamp(actions, min=-1.0 + 1e-6, max=1.0 - 1e-6)
raw_actions = 0.5 * torch.log((1.0 + clipped_actions) / (1.0 - clipped_actions))
gaussian_log_prob = self.gaussian_dist.log_prob(raw_actions).sum(dim=-1)
jacobian_correction = torch.sum(torch.log(1.0 - clipped_actions.pow(2) + 1e-6), dim=-1)
return gaussian_log_prob - jacobian_correction
```

### 4. Adaptive Action-Space Clipping Wrapper
The `AdaptiveActionSpaceClipsWrapper` class restricts the continuous exploration grid:
$$a_{\text{compressed}, i} = b_{c,i} + \lambda_c \times (a_i - b_{c,i})$$
As the rolling Attack Success Rate (ASR) increases beyond `asr_compression_threshold` (0.35), the compression scalar $\lambda_c$ shrinks, forcing the policy to focus its updates on the uncompressed, resilient manifolds.

---

## 5. Experimental Results & Evasion Insights

The codebase includes training logs tracing experimental progress across multiple sessions:

1. **RUN 20260605_153945 (Baseline Context-Biased):**
   - Breakthrough on Cluster 5 at step 50,364 with a peak score of 0.6108.
   - Initial confirmation of high-fidelity masks (bitrate: 160k, ratio: 1.0, tilt: -1.0).
2. **RUN 20260611_143032 (Hybrid Stochastic):**
   - Tuned rollout buffers ($N_{\text{steps}}=4096, \text{Batch}=512$) resolved catastrophic forgetting.
   - Achieved 5 breaches on Cluster 5 with a peak score of 0.6884.
3. **RUN 20260612_135450 (Warm-Started Biases):**
   - Warm-starting using GMM centroids achieved a peak score of 0.9745 (97.45% deception) on Cluster 5.
4. **RUN 20260614_003526 (Logit Rewards & Squashed Policy):**
   - Widespread distribution penetration achieved.
   - Cluster 7 reached an ASR of 65.93%, Cluster 6 hit 59.12%, and Cluster 2 hit 41.59%.
5. **RUN 20260614_144009 (Gradient Focus on Resilient Manifolds):**
   - With Cluster 5 excluded, the shared policy successfully generalized to highly resistant clusters.
   - Cluster 3 (historically < 1.5% ASR) reached 2.55% ASR (peak score 0.7688).
   - Target vulnerabilities converged to a unified signature: minimal tilt (-1.00) and maxed harmonics (1.00).

---

## 6. Methodological Recommendations for Thesis Defense

If presenting this work for academic examination, justify the design decisions as follows:

* **Defense of Static Conditioning:**
  * **Argument:** The GMM cluster assignment is an invariant metadata tag representing the source architecture of the synthetic voice clone (e.g., generative family). Allowing the GMM vector to drift mid-episode means the policy network learns to bypass the detector by exploiting features of the excluded family. Caching at reset is the only way to enforce true out-of-distribution (OOD) generalization validation.
* **Defense of Decoupled Multipliers:**
  * **Argument:** In classical RL, rare samples are often scaled globally to prevent minority starvation. However, scaling base penalties (like fidelity constraints) along with the success bonus inflates rewards for flatline spoof signals. This causes the Critic network to evaluate failures on rare clusters as successes, stalling exploration. Decoupling ensures that baseline penalties are uniform, while the multiplier acts strictly on successful breaches.
* **Defense of the Squashed Gaussian Policy:**
  * **Argument:** Standard continuous Gaussian policies evaluate actions assuming a linear, unbounded range. Tanh clipping inside wrappers breaks PPO's log-probability calculations, leading to policy collapse. The Squashed Gaussian policy applies change-of-variables Jacobian determinant corrections, aligning policy expectations with the squashed physical search boundaries.
* **Defense of Adaptive Clipping:**
  * **Argument:** Rather than dropping data from highly compromised clusters (which reduces sample efficiency), compressing exploration boundaries towards the cluster's biases reduces exploration entropy. Because it is harder to find positive policy updates within a contracted search grid, the advantage function naturally shifts gradient allocation toward the wide, uncompressed search grids of resilient populations.


---
---
A critical review of the provided documentation, thesis notes, and training session histories reveals several structural, methodological, and evaluative gaps between the planned Acoustic-Conditioned Policy (ACP) architecture and the currently implemented codebase.

Here is the breakdown of missing components required to complete the Master's thesis methodology.

### 1. Scientific Rigor & MDP Integrity (Critical Methodological Flaw)

**Gap: Gradient Leakage Prevention Mechanism**
The documentation in `acp-notes.md` identifies a severe scientific flaw: simply discarding excluded sub-manifolds (like Cluster 5) triggers gradient leakage if the Proximal Policy Optimization (PPO) rollout buffer still registers the observation state vector containing the GMM probabilities for that cluster.

* **Current Status:** The active `AdaptiveActionSpaceClipsWrapper` discards the sample and resets the environment. However, there is no evidence in the pipeline that the underlying Markov Decision Process (MDP) observation array is explicitly scrubbed, masked, or cached *prior* to trajectory storage. If the PPO buffer captures the state transitions leading up to the reset, the weights are still training on Cluster 5 features, invalidating the "Zero-Shot Inter-Cluster Adversarial Transferability" hypothesis.
* **Required Implementation:** A state-masking intercept or a strict reset-caching mechanism within the Gym environment to ensure zero data traces of quarantined clusters enter the PPO `RolloutBuffer`.

### 2. Architectural Pipeline Transitions

**Gap: Temporal Latent Frame Embeddings**
Following the `20260614_003526` run, the development notes established a requirement to transition the acoustic partition model from static GMM cluster IDs to sequence-aware temporal latent frame embeddings. This was designed to mitigate the policy exploiting temporal shortcut gaps.

* **Current Status:** Unimplemented. The active `train_config.yaml` still explicitly relies on the static `models/gmm_registry.pkl` with discrete `n_components: 10`.

### 3. Out-of-Distribution (OOD) Validation Scripts

**Gap: Zero-Shot Transferability Evaluation (`evaluate.py`)**
The theoretical foundation detailed in the notes relies on freezing a monolithic checkpoint and evaluating it against an isolated Cluster 5 test partition. This proves whether the spectral perturbations learned on resilient clusters successfully transfer to unseen distributions.

* **Current Status:** The evaluation scripts and dataloader logic for parsing the test partition and filtering strictly for specific GMM IDs do not exist in the current workflow.

**Gap: Cross-Architecture Backbones**
Previous iterations identified the need to evaluate winner configurations (such as the 90 extracted arrays) against `RawNet2` and `XLS-R` backbones to separate AASIST3-specific artifacts from generalized deepfake vulnerabilities.

* **Current Status:** The pipeline is exclusively hardcoded to the `MTUCI/AASIST3` feature extractor.

### 4. Diagnostics & Visualization (Thesis Documentation)

**Gap: 3D Feature Importance Maps**
The `developing-notes-acp.md` document outlines a requirement to generate a 3D scatter plot of `dsp_tilt` versus `dsp_jitter` across all 10 clusters, mapping the color space to the final AASIST3 score.

* **Current Status:** Unimplemented. There is no automated telemetry hook or standalone plotting script dedicated to visualizing the multi-dimensional correlation beyond standard CSV dumps.

### 5. Pending Execution Tasks

**Gap: Callback and Slicing Scripts**
The architectural prompts for two critical diagnostic tools were formulated but remain unexecuted in the codebase:

* `ClusterManifoldAlignmentCallback`: Required to mathematically track the Mean Squared Error (MSE) distance between the actions in Clusters 3 and 9 against the established `[-1.0, 1.0]` vulnerability plane during the rollout boundary step.
* `Cluster 3 Downsampling Intercept`: A standalone diagnostic script intended to sweep bitrates and extract raw un-softmaxed logits to explain the necessity of extreme downsampling in specific manifolds.