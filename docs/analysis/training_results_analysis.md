# KON-Artist Training Results Analysis

## 1. Executive Summary
Based on the training logs from the final iterations (~77k to ~79k steps), the agent has developed a high-entropy exploration strategy but is struggling to consistently bypass the AASIST3 detector. While it has identified "weak spots" in specific acoustic profiles (primarily Cluster 5), the overall success rate remains low.

## 2. Key Observations

### 2.1 Score & Reward Distribution
- **Zero-Score Dominance:** The vast majority of steps result in a `Score: 0.0000`. This indicates that the current DSP manipulations are not reaching the detector's decision boundary for most samples.
- **Cluster 5 Sensitivity:** Most of the "spikes" in score (up to `0.1571` at Step 78801) occur in **Cluster 5**. This suggests Cluster 5 represents an acoustic profile that is inherently more vulnerable to the specific DSP manipulations available (or the agent has successfully specialized for it).
- **Reward Inflation:** Rewards are frequently in the `8.0` to `15.0` range even with a score of `0.0000`. This is due to the `vertical_shift` of 25 and the logarithmic nature of the reward ($log(1e-9) \approx -20.7 + 25 = 4.3$). The "floor" of the reward is around 4.3.

### 2.2 DSP Parameter Patterns
- **Boundary Hitting:** Many parameters are hitting their hard limits (`-1.0` or `1.0`). For example, `tilt` and `jitter` are frequently pushed to extremes. This suggests the agent wants to apply even stronger transformations, or it has concluded that only extreme values have any effect.
- **High Variance:** Within a single cluster (e.g., Cluster 7), the agent is still trying widely different parameters in successive steps. This confirms that exploration is still very active, which is good, but might indicate that the policy hasn't converged on a "reliable" attack vector.

### 2.3 Training Metrics
- **Explained Variance ($\approx 0$):** The summary table shows `explained_variance: -1.94e-05`. This is a critical red flag. It means the PPO Value Network is unable to predict the rewards at all. The reward signal might be too noisy or the environment too stochastic for the critic to learn a stable baseline.
- **FPS (5):** The training throughput is very low. Centralized GPU inference is likely waiting on the CPU-bound DSP transformations in the worker environments.

## 3. Potential Bottlenecks

1.  **Sparse Reward Gradient:** The AASIST3 detector is very robust. Moving from a score of `1e-7` to `1e-4` provides a mathematical gradient, but it may not be enough to guide the agent through the massive DSP parameter space.
2.  **DSP Chain Order:** The current environment likely applies DSP transformations in a fixed order. Some manipulations might be "undoing" the effects of others.
3.  **Cluster Overlap:** If the GMM clustering isn't distinct enough, the agent receives conflicting feedback for similar-sounding samples.

## 4. Proposed Improvements

### 4.1 Reward Shaping Refinement
- **Curriculum Learning:** Start with a "weaker" version of the detector or a lower `success_threshold` to give the agent more frequent positive reinforcement.
- **Differential Rewards:** Instead of rewarding the absolute score, reward the *improvement* in score relative to the previous step (delta-reward).
- **Penalize Boundary Hitting:** Add a small penalty for DSP parameters that stay at `-1.0` or `1.0` to encourage finding more nuanced solutions.

### 4.2 Architectural Changes
- **Increase `n_steps`:** With such a low `explained_variance`, increasing the rollout buffer (`n_steps`) might provide a more stable gradient for the critic to learn.
- **Value Function Input:** Ensure the critic network has enough capacity. If the observation space is `160 + K`, the default MLP might need to be wider to capture the relationship between embeddings and rewards.

### 4.3 Efficiency
- **DSP Parallelization:** If librosa/sox transformations are the bottleneck, move them to a dedicated thread pool or use a GPU-accelerated DSP library (like `torch-audiomentations`).

## 5. Conclusion for Deeper Analysis
The agent is "touching" the boundary of success in specific clusters (Cluster 5) but is largely lost in the "zero-score" wilderness for others. Future efforts should focus on **stabilizing the Critic (Value Network)** and implementing a **curriculum** that bridges the gap between raw spoofed audio and the `0.5` success threshold.
