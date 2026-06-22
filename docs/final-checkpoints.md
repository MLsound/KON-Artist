# Final Checkpoints

---
# **Base Model:** `20260425_142549` (Legacy) — *Phase 1 - Protoyping*
> TOTAL_TIMESTEPS = 81920 | N_STEPS = 4096 | TOTAL_UPDATES = 20 | EPOCHS = 15 | BATCH = 4096

1.  **Confirmed Breach (Breakthrough):** Successfully bypassed the AASIST3 detector for the first time, achieving a **peak Bonafide score of 0.8402 (84%)**. The agent crossed the decision threshold in 3 distinct attack scenarios.
2.  **Additive Bonus Efficacy:** Replacing the multiplicative factor with a flat **additive bonus (+50)** provided the necessary "gradient gravity" to pull the policy across the 0.50 threshold.
3.  **Gradient Precision:** Increasing $N\_STEPS$ to **4096** significantly smoothed the convergence curve. The larger buffer allowed PPO to calculate more stable advantage estimates, which was critical for the final "squeeze" toward high-confidence scores.
4.  **Strategic Paradigm Shift:** The agent discovered a "High-Fidelity" attack strategy. Unlike early attempts using maximum distortion, the winning strategy utilized **minimum jitter/shimmer (-1.0)** and **high bitrates (~156k)**, suggesting that AASIST3 is more vulnerable to sutil spectral tilts than to heavy temporal noise.

---
# **PKI Model:** `20260617_172233` — *Phase 2 - Improvement*
> TOTAL_TIMESTEPS = 143360 | N_STEPS = 4096 | TOTAL_UPDATES = 35 | EPOCHS = 15 | BATCH = 256
* ***2 Checkpoints:** 20260602_003212 -> 20260617_172233*

1. **Successful Continuation and Reward Scale Elevation:** This session successfully resumed training, building directly upon previous model weights. A major architectural change was elevating the success bonus parameter to `bonus_amount: 75.0`. This adjustment lifted the global reward ceiling, with the absolute peak reward reaching an unprecedented **96.99**. The mean episodic reward began exceptionally high at **20.75** in the first decile and remained completely stable, hovering between **21.02 and 21.95** through to the tenth decile. This demonstrates that the model successfully locked onto a high-confidence exploit plateau without experiencing catastrophic forgetting or gradient regression.
2. **Extreme Exploit Density Over Extended Timesteps:** Over the course of the expanded 143,360-step horizon, the model demonstrated remarkable efficiency, logging an extraordinary **3,958 direct bypasses** ($Score > 0.50$). The model pushed its peak evasion confidence to an all-time elite ceiling of **0.9746**, demonstrating structural dominance over the AASIST3 Graph Attention Network.
3. **Consolidation of the Optimized Action Space Profile:** Statistical parameter analysis across the successful updates reveals that the policy distribution has completely consolidated around a high-fidelity signature. The agent locked the compression `ratio` cleanly to its absolute minimum baseline of **1.01** (representing near-zero dynamic degradation) and pinned the median `bitrate` at a highly pristine **151.94 kHz**. Concurrently, both `jitter` (mean: `-0.84`) and `shimmer` (mean: `-0.77`) were driven heavily toward their minimum boundaries to minimize destructive acoustic telling signs, proving that the agent has abandoned chaotic exploration in favor of a precise spectral masking attack.

---
# **ACP Model:** `20260425_142549` — *Phase 3 - Alternative*
> TOTAL_TIMESTEPS = 655360 | N_STEPS = 4096 | TOTAL_UPDATES = 160 | EPOCHS = 4 | BATCH = 512

* ***8 Checkpoints:** 20260605_234220 -> 20260612_135450 -> 20260613_021742 -> 20260613_151934 -> 20260614_003526 -> 20260614_144009 -> 20260616_134732 -> 20260618_130910*

1. **Empirical Validation of Latent Space Manifold Alignment:** Re-integrating Cluster 2 back into the active environment partition split has conclusively validated the latent transferability hypothesis. Without requiring standalone brute-force exploration loops, Cluster 2 registered an immediate **12.88% ASR** with a high mean deception profile of **0.8256**. The policy's internal shared hidden layers automatically mapped the learned structural parameters onto the new domain, pinning `dsp_tilt` to `-0.92` and `dsp_harmonics` to `0.92` to secure immediate breakthroughs.
2. **Cluster 1 Saturation Under Bound Compression Constraints:** Despite the adaptive clipping wrapper throttling exploration boundaries at the stricter `0.20` ASR compression threshold, Cluster 1 expanded its success footprint to **35.95% ASR** (up from 20.95%). Lowering the exploration variance budget to an initial coefficient of `0.08` forced the policy to abandon noisy parameters and hyper-optimize within the legal compressed volume, proving that constraint injection drives deterministic exploitation accuracy.
3. **Global Manifold Flattening and Multi-Dimensional Convergence:** The global acoustic space is now showing signs of complete optimization saturation. Cluster 4 stabilized at a robust **14.35% ASR**, Cluster 0 advanced to **5.43% ASR**, and both Clusters 3 and 9 maintained verified beachheads with localized bitrate downsampling. This uniform collapse across defensive sub-populations confirms the actor has mapped an invariant geometric vulnerability plane inside the AASIST3 extractor framework.
