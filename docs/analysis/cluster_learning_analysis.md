# Cluster-Aware Learning Analysis

## Overview
This document analyzes how the KON-Artist RL agent integrates acoustic clustering information into its learning process and evaluates its capacity to specialize DSP attack strategies per cluster.

## 1. Data Flow Verification

### Observation Ingestion
The PPO agent receives its state via the `VecDetectorWrapper` in `src/env/wrappers.py`. The observation vector is constructed as follows:
- **Base State (160 dimensions):** High-dimensional embeddings extracted from the AASIST3 detector.
- **Acoustic Context ($K$ dimensions):** Soft-assignment probabilities (from the GMM registry) indicating which acoustic cluster the current spoofed sample belongs to.

In `scripts/train.py`, the environment is initialized with this wrapper:
```python
env = VecDetectorWrapper(
    env, 
    detector, 
    # ...
    config=cfg
)
```
Since the `observation_space` of the wrapped environment includes the $K$ cluster dimensions, the PPO agent's policy network (MLP) explicitly "sees" the cluster identity during every forward pass.

### Learning Signal
The reward function in `src/env/reward_logic.py` provides the feedback loop:
- The agent attempts a DSP manipulation.
- It receives a reward based on the detector's classification score.
- Because the cluster ID is part of the input, the agent can correlate specific DSP parameter ranges (e.g., higher "jitter" for Cluster 0, lower "tilt" for Cluster 1) with higher rewards for that specific cluster.

## 2. Learning Capacity

### Architecture Support
The agent uses an `MlpPolicy`. A multi-layer perceptron is a universal function approximator. Given that the cluster probabilities are distinct features in the input vector, the hidden layers of the MLP can develop internal representations that act as "conditional logic." 

Mathematically, the policy $\pi(a|s, c)$ (where $a$ is action, $s$ is embedding, and $c$ is cluster) allows the agent to learn different mappings from $s \to a$ depending on the value of $c$.

### Specialization Evidence
To confirm if specialization is occurring, we analyze the output of `analyze_clusters.py`:
- **DSP Differentiation:** If Cluster A consistently shows a mean "jitter" of 0.8 while Cluster B shows 0.2, it confirms the agent has learned that different acoustic profiles require different levels of perturbation to bypass the detector.
- **Score Disparity:** Variations in average scores across clusters indicate that some acoustic profiles are harder to spoof than others, prompting the agent to further refine strategies for those specific clusters.

## 3. Potential Bottlenecks
- **Cluster Overlap:** If the GMM yields high entropy (e.g., 0.5 for Cluster 0 and 0.5 for Cluster 1), the agent may receive a "noisy" signal, making specialization harder.
- **Static Rewards:** Currently, the reward logic is cluster-agnostic (it only cares about the final score). While this allows for specialization, adding cluster-specific difficulty weighting could further accelerate learning on "tougher" acoustic signatures.

## 4. Conclusion
The current implementation of `scripts/train.py` and its associated wrappers correctly injects cluster information into the PPO agent's observation space. The architecture is fully capable of learning specialized DSP parameters per cluster, and the analysis tools (`get_winners.py` and `analyze_clusters.py`) provide the empirical evidence to verify this specialization.
