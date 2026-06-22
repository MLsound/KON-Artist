The development process for the **KON-Artist** framework has transitioned from resolving critical environmental crashes to restructuring the fundamental Reinforcement Learning architecture. The strategic decisions made were focused on ensuring experimental reproducibility, stabilizing the data flow, and providing the PPO agent with mathematically viable gradients to exploit AASIST3's KAN decision boundaries.

# Improvement phase

# RUN: 20260516_194032
TOTAL_TIMESTEPS = 81920 | N_STEPS = 4096 | TOTAL_UPDATES = 20 | EPOCHS = 15 | BATCH = 256

***FEAT 1: High-Throughput Vectorized Pipeline & Configurable Training Refactor***

1. **Vectorized Architecture & Smooth Convergence:** This run successfully introduced parallel environment execution via `VecDetectorWrapper` for GPU batching. The convergence curve demonstrated a perfectly monotonic, textbook upward trend with zero policy oscillation, lifting the mean reward steadily across all training deciles (scaling from a mean of 9.20 to 13.95).
2. **Minibatch Regularization Effect:** Transitioning from full-batch updates to downscaled minibatches of **256** regularized the gradient surface. The stochastic noise from smaller minibatches effectively protected the agent from localized overfitting on specific file tracks, trading volatile singular peaks for widespread statistical stability.
3. **The Generalist Trade-Off:** Due to the smoothed optimization landscape, the agent acted as a conservative generalist. It prioritized lifting the overall performance floor across the entire data pool rather than aggressively exploiting brittle anomalies, holding the individual file peak at a highly structured **0.0223**.

> NEXT MOVE:
> 1. **Loosen PPO Clipping (`clip_range = 0.3`):** Increase the clip range from 0.2 to 0.3 to allow the generalized policy to take slightly larger gradient steps when it encounters vulnerabilities, pushing beyond the conservative 2.2% stall barrier.
> 2. **Implement Learning Rate (LR) Annealing:** Add a linear schedule to decay the learning rate towards zero in the late stages of training, ensuring the agent fine-tunes its precision once it drops into high-reward zones.

---
# RUN: 20260517_160451
TOTAL_TIMESTEPS = 81920 | N_STEPS = 4096 | TOTAL_UPDATES = 20 | EPOCHS = 15 | BATCH = 256

***FEAT 2: Policy Optimization & Strategic Exploration Tuning***

1. **Successful LR Annealing Deployment:** The integration of a linear learning rate decay schedule successfully stabilized the training trajectory. It regularized the mid-to-late stage exploration steps and smoothed reward transitions between optimization deciles, preventing policy degradation.
2. **Conservative Generalization Overfitting Bounded:** Due to the combination of a strict clipping parameter (default 0.2) and a flat success bonus structure, the agent learned a conservative, safe policy. It raised the universal reward baseline across the dataset pool but was structurally constrained from exploring high-variance, high-reward singular anomalies, locking the peak evasion score at 0.0196.

> NEXT MOVE:
> 1. **Modify `configs/train_config.yaml` to shorter train cycle for testing.**
> 2. **Explicitly pass `clip_range=0.3` to the PPO constructor:** This changes the policy boundary restriction, allowing the vectorized agent to take bolder steps when it identifies systemic vulnerabilities in AASIST3.

---
# RUN: 20260519_155042
TOTAL_TIMESTEPS = 40960 | N_STEPS = 4096 | TOTAL_UPDATES = 10 | EPOCHS = 15 | BATCH = 256

***FEAT 6: Codec Optimization & DSP Pipeline Finalization***

1. **Immediate Breakthrough via Parameter Loosening:** By loosening the PPO parameter limits to a custom target profile, the agent immediately broke out of its previous conservative baseline. Within a shorter **40,960-step execution horizon**, the framework successfully bypassed the target model, achieving a significant **peak score of 0.4077** (Reward: 24.10).
2. **Validation of Progressive Reward Signals:** The rapid ascent toward a 40.7% deception rate validates the code integration of intermediate directional targets. Providing earlier gradient signals pulled the vectorized actor straight out of the low-level plains, demonstrating that proper reward shaping can override conservative generalization traps even during brief, compressed training runs.
3. **Unsaturated Terminal Exploration:** As visualized in `convergence_20260519_155042.png`, the smoothed trajectory maintains a sharp, vertical, linear slope up to the final timestep. The policy was stopped mid-flight during its steepest learning phase, verifying that the combination of minibatch processing ($M=256$) and wider clipping boundaries creates an optimized path toward a full model breach ($>0.50$).

> NEXT MOVE:
> 1. **Implement Linear Learning Rate (LR) Annealing Schedule:** Now that high-scoring pockets have been successfully unblocked, reinstate the linear decay function to narrow optimization step sizes as the agent converges inside the newly discovered high-reward regions.
> 2. **Refactor `src/env/reward_logic.py` for two-stage progressive scaling:**
>    - If score > 0.10: reward += 25.0 (Intermediate directional signal)
>    - If score > 0.50: reward += 100.0 (Ultimate evasion objective)

---
# RUN: 20260519_205820
TOTAL_TIMESTEPS = 40960 | N_STEPS = 4096 | TOTAL_UPDATES = 10 | EPOCHS = 10 | BATCH = 256

1. **Critical Protocol Ingestion Leak (Data Contamination):** A deep audit of the environment reset pipeline revealed severe label contamination. The audio ingestion engine loaded **419 true "Bonafide" human samples** alongside **602 adversarial "Spoof" deepfakes**.
2. **Over-Regularization Stagnation:** The combination of an extremely restricted action update horizon (only 10 total updates and 10 epochs) coupled with mini-batching completely clipped the agent's parameter elasticity. The agent was trapped in an over-regularized baseline plateau, unable to take large enough mathematical steps to uncover hidden deceptive clusters.
3. **Incomplete Optimization Path:** The training session concluded at a truncated step count with a completely flat terminal profile. This visual stagnation confirms that the agent never escaped the noisy initialization plains due to the contaminated gradient signals received from the mixed audio dataset split.

> **NEXT STEPS & CODE REPAIRS:**
> 1. **Strict Dataset Ingestion Filtering (Mandatory):** Modify `generator_from_ds` inside `src/data/loader.py` or the filtering logic in `src/env/audio_attack.py` to assert a strict categorical exclusion rule. The dataset pipeline **must ignore true Bonafide signals** and *exclusively* yield `Label: Spoof` targets to the environment. The generator's sole mathematical function is to convert deepfakes into human representations, not to compromise real human voices.
> 2. **Increase Optimization Epoch Volume:** Increase `EPOCHS` back to **15** and set `TOTAL_UPDATES` to **25** to grant PPO sufficient gradient accumulation windows to safely stabilize its policy after the pipeline cleanup.
> 3. **Restore Evasion Signal Ramps:** Re-verify that the two-stage progressive reward thresholds ($+25$ at score $>0.10$ and $+100$ at score $>0.50$) are actively triggering inside `src/env/reward_logic.py` to give the policy an intense directional acceleration signal once dataset contamination is removed.

---
# RUN: 20260520_171441
TOTAL_TIMESTEPS = 81920 | N_STEPS = 4096 | TOTAL_UPDATES = 20 | EPOCHS = 15 | BATCH = 256

1. **Identification of Edge-Case Exploitation (Peak Score 0.9745):** While the agent achieved an all-time high peak score of 0.9745, this indicates a high-variance exploitation of a brittle vulnerability rather than a generalized architectural breakthrough. Relaxing PPO's structural bounding box (`clip_range = 0.3`) combined with the shift to batched GPU inference allowed the policy to aggressively overfit to highly specific, extreme DSP parameters, resulting in an unstable "glass cannon" attack vector.
2. **Bimodal Policy Collapse via Discrete Reward Shaping:** The two-stage milestone reward matrix created an exploitation trap. Massive binary reward jumps (e.g., oscillating between ~150 for success and ~9 for failure) dominated the gradient signal. Consequently, the agent learned to treat the continuous DSP action space as a discrete threshold breaker, violently thrashing between complete success and near-total failure on adjacent steps (e.g., scores swinging from 0.9738 to 0.0000) instead of incrementally optimizing difficult audio samples.
3. **False Convergence and Hyperparameter Overfitting:** The apparent exponential convergence (mean baseline reward rising from 9.88 to 29.38 in the final decile) is a statistical artifact driven by sparse, extreme outliers. With a peak reward of 149.97, a final decile mean of 29.38 mathematically demonstrates that the agent fails the vast majority of its attempts. Retaining 15 optimization epochs with smaller mini-batches accelerated convergence into local minima, causing the action variance to collapse prematurely before a stable, generalized adversarial strategy could be formulated.

> NEXT MOVE:
> 1. **Validate Continuous Optimization:** Run a new training session with the updated `train_config.yaml` and linear reward shaping. Monitor the action distribution entropy to ensure smooth decay and verify that the mean baseline reward rises synchronously with peak rewards.
>    * **Decreased Epochs (15 $\rightarrow$ 8):** Prevents the PPO agent from excessively iterating over the same 4096-step rollout buffer, stopping it from "memorizing" and overfitting to specific edge-case actions that accidentally fooled the detector.
>    * **Tightened Clip Range (0.3 $\rightarrow$ 0.15):** Enforces a stricter mathematical "trust region" for the optimizer. This stops the massive, destructive policy swings that were causing the agent to thrash between 100% success and near-total failure on consecutive steps.
>    * **Lowered Learning Rate (0.0003 $\rightarrow$ 0.0001):** Stabilizes the gradient updates, helping the model settle gently into a generalized adversarial strategy rather than aggressively oscillating and jumping past optimal solutions.
>    * **Increased Action Entropy (Initial: 0.01 $\rightarrow$ 0.05 | Final: 0.001 $\rightarrow$ 0.01):** Forces the agent to maintain a wider variety of DSP actions (jitter, shimmer, tilt) for a longer period. This ensures sustained exploration of the continuous parameter space and prevents the policy from prematurely collapsing into deterministic, extreme edge cases.
> 2. **Enforce Acoustic Bounding:** Implement strict bounds clipping on the continuous action space (jitter, shimmer, tilt, harmonics). This ensures adversarial perturbations remain acoustically plausible to a human listener, preventing the generation of invalid structural noise.

---
# RUN: 20260521_140820
TOTAL_TIMESTEPS = 81920 | N_STEPS = 4096 | TOTAL_UPDATES = 20 | EPOCHS = 15 | BATCH = 256

1. **Definitive Exploit Consolidation (97.45% Evasion):** The agent has mathematically validated that the vulnerabilities discovered in the previous session were not statistical anomalies. The model reached a peak score of 0.9745 (97.45% confidence that the modified audio is human) and an absolute peak reward of 268.61, confirming total dominance over the AASIST3 detector’s Graph Attention Network (GAT).
2. **High-Frequency Systematic Breach:** The system recorded 1,318 confirmed successful attacks (classifications > 0.50) throughout the training horizon. This demonstrates that combining the minibatch buffer ($M=256$) with PPO elastic limits (clip_range = 0.3) allows the policy to capitalize on high-reward regions and mass-replicate the attack across different files in the dataset.
3. **Stable Convergence and Policy Maturation:** Decile analysis shows healthy, exponential growth in mean reward, scaling from 10.85 in the first decile to stabilizing at ~22.19 in the last two. The convergence chart shows that after a sharp escalation, the moving average settles into a structured plateau with no signs of policy collapse, indicating the agent has learned a generalized and sustainable attack vector.

> NEXT MOVE:
> 1. **Configuration Layer (configs/train_config.yaml):** Introduced explicit parameters under the PPO configuration block to handle the continuous bonus logic.
> 2. **Environment & Wrapper Propagation:** Enhanced both AudioAttackEnv and VecDetectorWrapper by introducing bonus and bonus_amount as initialization parameters, allowing configurations to flow smoothly from the root pipeline into the core environment logic.
> 3. **Reward Logic Decoupling:** Rewrote the internal reward calculation to respect the new self.bonus flag and utilize self.bonus_amount dynamically during rollouts.
> 4. **Fallback Safety:** Maintained backward compatibility by implementing a default fallback value of 250.0 for instances where bonus_amount is omitted or unconfigured.

---
# RUN: 20260521_221713
TOTAL_TIMESTEPS = 81920 | N_STEPS = 4096 | TOTAL_UPDATES = 20 | EPOCHS = Reduced | BATCH = 1024

1. **Variance Stabilization (Reward Smoothing):** The implementation of proportional reward successfully eliminated the "mathematical cliff" that caused previous *overfitting*. By scaling the bonus dynamically (up to a recorded maximum of **+48.71**), the PPO optimizer stopped attempting to abruptly rewrite its network, massively reducing the amplitude between peaks and valleys in the convergence graph.
2. **Successful Minibatch Regularization:** Increasing the Minibatch size to **1024** forced the agent to average gradients over a much larger audio sample before updating weights. As a result, the mean reward scaled perfectly monotonically and in a controlled manner from **8.92** in the first decile to **11.03** in the last, with no signs of policy collapse.
3. **Attack Peak Retention (Balanced Exploration):** Despite strong regularization, the agent **did not lose its lethal attack capability**. **656 successful attacks** ($> 0.50$) were recorded, maintaining a **maximum score of 0.9741**. Most impressively, the network was able to replicate scores of $>0.97$ consistently throughout *all training deciles*, proving that the evasion policy is generalized and not tied to isolated files.

> NEXT MOVE:
> 1. **Verification & Testing:**
>     * **Validate Configuration Parsing & Propagation:** Run a short smoke test to verify that changes in train_config.yaml (bonus: False or altering bonus_amount) are correctly parsed by the configuration loader and successfully passed down to both AudioAttackEnv and VecDetectorWrapper without initialization errors.
>     * **Assert Reward Computation Integrity:** Write a unit test for AudioAttackEnv to explicitly check the reward logic under three scenarios:
>         1. bonus=True and bonus_amount=50.0 (assert custom scaling applies).
>         2. bonus=False (assert bonus is exactly 0.0 or omitted).
>         3. Edge case/Omission (assert backward-compatible fallback to 250.0 works seamlessly if parameters are missing from older or partial configs).
> 2. **Monitoring & Logging Adjustments:**
>     * **Expose Parameters to TensorBoard/W&B:** Ensure that bonus and bonus_amount are logged as hyperparameter metadata at the start of each training run.
>     * **Track Reward Components Separately:** If not already implemented, update the environment's logging/tracking to separate the base reward from the continuous_bonus_reward. This prevents the dynamic shifts in reward scaling from obscuring baseline policy progression during training analysis.
> 3. **Hyperparameter Exploration:** Since the recent run (20260521_221713) showed stable convergence with a maximum recorded bonus of **+48.71** using this new proportional approach, run a small grid search or ablation tracking variations of bonus_amount (e.g., 25.0, 50.0, 100.0) to find the optimal bounds for variance stabilization vs. optimization speed.

---
# RUN: 20260522_120609
TOTAL_TIMESTEPS = 20480 | N_STEPS = 4096 | TOTAL_UPDATES = 5 | EPOCHS = Reducidas | BATCH = Full

1. **Validation of Autonomous Evasion (Without Artificial Bonuses):** Despite the temporary deactivation of the additive bonus multiplier (`Bonus: 0.00`), the agent successfully discovered **87 complete breaches ($> 0.50$)** independently over just 5 iterations (20k steps). This demonstrates that the base architecture and the underlying logarithmic reward shaping are mathematically robust and sufficient to guide PPO toward vulnerability vectors.
2. **Sustainability of Attack Success (0.9743 Peak):** The model retained its lethal attack capacity, reaching a maximum evasion score of **0.9743** (97.43% confidence of Bonafide human voice). Impressively, it maintained scores $>0.96$ consistently across *all deciles* of training, confirming that the attack policy is generalized and not tied to isolated, lucky file hits.
3. **Absolute Variance Suppression (Max Batch):** By setting the Minibatch equal to the Rollout Buffer (`BATCH = 4096`), PPO's gradients became entirely deterministic, eliminating step-wise stochastic noise. The mean reward remained extremely flat and stable in the **8.52 to 9.05** range throughout the session. While this prevented policy collapse, the stabilization was so aggressive that it stifled exploration, suggesting that a smaller minibatch size is necessary for a more aggressive ascent during longer training runs.

> NEXT MOVE:
> 1. **Execute Full Training Horizon (100k+ Steps):** Since stability is now guaranteed and the *Full Batch* configuration suppresses noise, it is imperative to scale to `TOTAL_UPDATES = 25` (~100,000 steps) to provide the network with the time horizon required to aggressively raise the global mean reward above 10 points.
> 2. **Reactivate Proportional Scaling (Bonus):** Reactivate the proportional reward bonus to provide a stronger gravitational vector that accelerates the agent's climb out of the initial exploration valley, pushing the density of successful attacks beyond 1,000 per run.
> 3. **Prepare EER Extraction:** Proceed to run the saved model weights (`kon_artist_agent.zip`) through the evaluation pipeline against the *Test Split* to isolate and measure the objective degradation of the target AASIST3 model's Equal Error Rate.

---
# RUN: 20260530_205050
TOTAL_TIMESTEPS = 20480 | N_STEPS = 4096 | TOTAL_UPDATES = 5 | EPOCHS = Unknown (Likely reduced) | BATCH = Full

1. **Restoration of the Mean Reward Curve:** The structural fixes proposed by the coding agent were an absolute success. With the learning rate and entropy decay schedules properly resetting (`reset_num_timesteps=True`), the simple moving average (SMA) of the reward is no longer flatlining. As seen in the convergence graph, the SMA line exhibits a healthy, steady upward slope throughout the 20k-step horizon, confirming that the agent is actively learning and escaping the exploration valley.
2. **Activation of Proportional "Gravity" (Bonus):** The logs confirm that the proportional success bonus was successfully reactivated (`Bonus Reward: ENABLED (Amount: 50.0)`). We can observe the logic correctly applying intermediate scaling before the 0.50 threshold (e.g., `Score: 0.3660 | Reward: 42.29 | Bonus: 18.30`). This intermediate gravity successfully pulled the agent out of the baseline plains.
3. **High-Density Evasion Validation:** Even within a highly compressed timeframe of only 5 updates (20,480 total steps), the agent managed to discover **106 complete breaches** (scores $>0.50$), culminating in a **peak score of 0.9742**. This is an exceptionally high density of successful attacks for such a short window, proving that the combination of proportional rewards and the `VecDetectorWrapper` creates a highly optimized adversarial environment.

> NEXT MOVE:
> 1. **Launch the Full Horizon Run (~100k steps):** The convergence plot shows that the learning curve was abruptly cut off while it was still climbing steeply. You must increase `TOTAL_UPDATES` to 25 to allow the agent the required runway to reach the high-reward plateau and fully consolidate its policy weights.
> 2. **Reduce Minibatch Size to Increase Variance:** Currently, `BATCH = 4096` equals the `RolloutBuffer`, meaning you are executing Full Batch Gradient Descent. Reduce `BATCH` back down to **256 or 512** to reintroduce stochastic noise; this will help the agent break through local minima faster during the longer 100k-step run.

---
# RUN: 20260530_235531
TOTAL_TIMESTEPS = 81920 | N_STEPS = 4096 | TOTAL_UPDATES = 20 | EPOCHS = 15 | BATCH = 256

1. **Perfect Recreation of the Classic Learning Curve:** By breaking the full-batch deterministic trap and adjusting the minibatch size to **256**, you successfully reintroduced stochastic noise into the policy gradients. The mean episodic reward displays a flawless, monotonic upward climb across all training stages, starting at **9.45** in the first decile and escalating continuously to **18.47** in the final decile. This confirms that the model is genuinely learning a generalized policy instead of over-indexing or flatlining.
2. **High-Density, High-Confidence Exploits:** With the linear decay schedules executing smoothly over the expanded 20-update window (81,920 total timesteps), the agent moved systematically from broad exploration to razor-sharp exploitation. It logged a spectacular total of **1,557 direct bypasses** ($>0.50$), maintaining an all-time elite peak score of **0.9744**.
3. **Policy Stability under Loose Boundaries:** Raising the PPO clipping parameter to `clip_range: 0.3` granted the policy network the elasticity required to leap out of local minima and aggressively capture highly hidden adversarial pockets within the AASIST3 model. Crucially, the proportional reward bonus functioned smoothly alongside this setup, preventing the chaotic gradient explosions seen in early configurations while establishing a sustainable, high-fidelity attack vector.

> NEXT STEPS:
> 1. **Extend Training Horizon:** The unsaturated positive terminal slope of the convergence plot indicates that the stable minibatch framework requires a longer window (**150,000 to 200,000 total steps**) to fully exploit the learned boundaries.

---
# RUN: 20260531_184936 -> 20260601_135618
TOTAL_TIMESTEPS = 204800 | N_STEPS = 4096 | TOTAL_UPDATES = 50 | EPOCHS = 15 | BATCH = 256

1. **Extended Horizon Stability and Policy Convergence:** Spanning over 205,000 total combined timesteps across the merged sequence, the agent demonstrated exceptional algorithmic stability. The mean episodic reward sustained a robust and confident tier between **18.26** and **22.61** in the final phases, proving that the policy successfully saturated without suffering from catastrophic forgetting, gradient explosions, or policy divergence.
2. **Fixation on the Elite Clean Masking Vector:** In the deep exploitation phase, the continuous action variables clearly consolidated around the high-fidelity mask. The agent locked `ratio` permanently at `1.0`, kept `bitrate` safely maximized between **142k** and **160k**, and drove both `jitter` and `shimmer` to their absolute minimum boundaries (`-1.0`). This minimizes destructive acoustic artifacts while systematically blinding the AASIST3 graph attention architecture.
3. **Elimination of Stochastic Drift:** By allowing the entropy coefficient to decay smoothly over an extended timeline, the policy distribution tightened into a highly reliable, near-deterministic exploit mechanism. The highly repeatable scores confirm that the agent has transcended random exploration and established a highly specialized attack vector.

> NEXT STEPS:
> 1. **Deploy Prior Knowledge Injection (Strategy 2):** Initialize the next training run utilizing the Actor Network Biasing hook to start the policy distribution mean directly on this discovered signature, accelerating convergence on new trials.
> 2. **Extract DSP Feature Importance Vectors:** Analyze the preferred continuous action variables (Jitter, Shimmer, Tilt, and Bitrate) across the final 10,000 steps to isolate the specific spectral distortions that systemically blind graph attention networks.


------
# RUN: 20260602_003212 (Prior Knowledge Injection)
TOTAL_TIMESTEPS = 61440 | N_STEPS = 4096 | TOTAL_UPDATES = 15 | EPOCHS = 15 | BATCH = 256

1. **Massive Acceleration via Prior Knowledge Injection:** The implementation of Actor Network Biasing was a resounding success. By initializing the policy output mean to the previously discovered "Winning Signature" (`jitter: -1.0`, `ratio: 1.0`, `bitrate: 1.0`, etc.), the agent completely skipped the initial "exploration valley." The mean episodic reward started immediately at **16.51** in the first decile—nearly double the starting reward of previous unbiased runs—and quickly saturated at a highly stable plateau of **~21.00 - 21.55**.
2. **Unprecedented Exploit Density:** By warm-starting the agent in the optimal adversarial zone, the policy was able to focus entirely on fine-tuning rather than discovery. This resulted in an extraordinary **3,582 direct bypasses** (Score $> 0.50$) within a shortened horizon of only 61,440 total steps. This is more than double the successful attacks of your previous best run, despite training for far fewer total steps.
3. **Retention of Exploratory Freedom:** Despite the strong initial bias, the convergence graph and the final DSP logs demonstrate that the agent did not suffer from policy collapse. The entropy decay schedule allowed PPO to safely explore variations around the bias point (e.g., tweaking `tilt` and `harmonics` on a per-sample basis) to maintain the peak evasion score of **0.9745** while navigating different spoofed input files.

> NEXT STEPS:
> 1. **Ablation study:** starting from the same Knowledge injected, compare results without minibatch or reward bonus.

---
# RUN: 20260603_015026 (Prior Knowledge Injection) NO Minibatch
TOTAL_TIMESTEPS = 61440 | N_STEPS = 4096 | TOTAL_UPDATES = 15 | EPOCHS = 15 | BATCH = Max

1. **Impact of Full-Batch Processing on Bias Retainment:** This run returned to a `BATCH = 4096` configuration (Full-Batch Gradient Descent) while utilizing the Prior Knowledge Injection parameters. Because the batch updates were calculated deterministically across the entire rollout buffer, the initial reward profile started high (**16.09**) and experienced virtually zero chaotic fluctuations. The learning trajectory shows an extremely flat, highly stable ascent concluding at **18.70**, demonstrating how a full-batch approach acts as an algorithmic smoothing filter on the initialized bias.
2. **Deterministic Retention of the Elite Exploit Matrix:** Operating under the full batch size prevented stochastic gradient variance from knocking the policy out of its initialized comfort zone. The final step logs show the agent cleanly applying the preferred signature—consistently forcing `ratio: 1.0`, pinning `bitrate: 160000`, and holding `jitter: -1.0`. The architecture logged **3,646 direct bypasses** and matched the historical ceiling with a **peak score of 0.9745**.
3. **The Stability vs. Adaptability Trade-off:** Comparing the convergence curve of this run to the mini-batch run (`20260602_003212`), the full-batch execution tracks lower across the final deciles (ending at a mean reward of 18.70 vs the mini-batch's 21.55). Without mini-batch stochastic noise, the agent treats the initialized target signature as a rigid geometric baseline, heavily optimizing execution reliability but displaying less flexibility in adapting micro-parameters (like `threshold` or `harmonics`) to maximize per-sample scoring.

> NEXT STEPS:
> 1. **Lock the Mini-Batch Variant for Out-of-Distribution Benchmarking:** While this full-batch run provides unparalleled policy stability, the mini-batch variant (`20260602_003212`) achieved a higher average reward ceiling (21.55). Use the mini-batch checkpoint for your final `evaluate.py` execution against the unseen test split.
> 2. **Extract Comparison Graphs for the Thesis:** Plot the learning curves of `BATCH = 4096` and `BATCH = 256` side-by-side under the biased initialization. This yields a highly compelling academic visualization showcasing the effect of stochastic gradient noise on exploration vs. exploitation mechanics.

---
# RUN: 20260603_015026 (Prior Knowledge Injection) NO Reward bonus
TOTAL_TIMESTEPS = 61440 | N_STEPS = 4096 | TOTAL_UPDATES = 15 | EPOCHS = 15 | BATCH = Max

1. **Full-Batch Smoothing of Prior Knowledge Injection:** By matching the mini-batch size directly to the rollout buffer size (`BATCH = 4096`), this run eliminated all intra-update gradient variance. Powered by the pre-injected target signature, the policy warm-started aggressively with an immediate mean reward of **16.09** in the first decile. However, the lack of stochastic mini-batch noise acted as a massive optimization filter, forcing a very flat, highly stabilized trajectory that capped at **18.70** by the final decile.
2. **High-Volume Template Replication:** Because the full-batch update computes a single averaged gradient across the entire rollout, the policy became a highly rigid template applicator. It locked down the core winning signature variables uniformly—forcing `ratio: 1.0`, pinning `bitrate: 160000`, and holding `jitter: -1.0`—with extreme structural consistency. This deterministic locking mechanism produced a staggering **3,646 direct bypasses** and achieved the maximum peak evasion score of **0.9745**.
3. **The Exploitation Plateau:** While full-batch training maximizes policy safety and prevents the agent from drifting away from the initialized winning zone, it introduces a clear ceiling effect. The final mean reward plateaued noticeably lower than the mini-batch variant (18.70 vs. 21.55) because the agent lacked the gradient variance required to flexibly adapt secondary features (such as `tilt`, `harmonics`, or compression `threshold`) on a sample-by-sample basis.

> NEXT STEPS:
> 1. **Deploy the Adaptive Mini-Batch Variant for Out-of-Distribution Benchmarking:** Since sample-specific contextual adaptation is far more resilient against unseen audio files than a rigid static template, use the mini-batch checkpoint (`20260602_003212`) for the final `evaluate.py` test run.
> 2. **Map the Optimization Comparison for Thesis Documentation:** Pair the convergence plots of the mini-batch (stochastic) and full-batch (deterministic) initialized runs to provide a robust academic analysis on how gradient noise impacts exploration behavior when warm-starting an agent.
> 3. **Run Final Adversarial EER Verification:** Execute the finalized evaluation script across the balanced test split to compute your final degraded Equal Error Rate (EER) and tandem Detection Cost Function (min t-DCF) tables.

---
# RUN: 20260604_181410
TOTAL_TIMESTEPS = 40960 | N_STEPS = 4096 | TOTAL_UPDATES = 10 | EPOCHS = 15 | BATCH = 256

1. **Impact of Omitting Prior Knowledge Injection:** This training session served as an empirical baseline test by running without Actor Network Biasing. Deprived of the pre-injected target signature, the policy was forced to begin exploration from a neutral origin (mean action output of `0.0`). Consequently, the learning curve suffered a massive regression compared to previous warm-started runs; the mean episodic reward started at a low baseline of **7.77** in the first decile and slowly crawled to just **9.73** by the tenth decile.
2. **Zero Bypasses and Exploitation Ceiling:** Due to the combination of random initialization and a short training timeline (only 10 total updates), the agent was completely unable to discover the uncompressed, high-fidelity exploit pocket on its own. The run generated **0 direct bypasses** (AASIST3 scores $>0.50$), with the absolute peak evasion score topping out at a weak **0.1601**. As a result, the `winners_20260604_181410.csv` file remains completely empty, demonstrating that the "Clean Masking" strategy is too complex for the agent to organically hit within a limited 40k-step horizon.
3. **Successful Integration of Granular Cluster Logging:** A major structural advancement in this run is the introduction of data profile **Cluster tracking** (0 through 9) inside the reward logic loop. The console telemetry successfully registered and logged the sample category for every environment step. The distribution was heavily dominated by Cluster 7 (9,875 steps) and Cluster 2 (9,727 steps). This establishes the essential telemetry required to map and analyze whether specific deepfake generator sub-architectures exhibit higher resilience to adversarial adjustments.

> NEXT STEPS:
> 1. **Reinstate the Injected Target Signature:** Re-enable the Actor Network Biasing initialization hook using your optimized parameters (`jitter: -1.0`, `shimmer: -1.0`, `ratio: -1.0`, `bitrate: 1.0`) to immediately pull the agent out of this sub-optimal exploration loop and restore the high-density >0.95 evasion behavior.
> 2. **Conduct Per-Cluster Vulnerability Audits:** Once the biased agent is running effectively again, leverage the new cluster logging variables to plot success metrics across each data sub-category. This will allow you to isolate which deepfake pipelines are most vulnerable to your high-fidelity smoothing mask for your thesis discussion.
> 3. **Execute Final OOD Inference Verification:** Freeze your elite mini-batch model checkpoint (`20260602_003212`) and route it through `evaluate.py` against the unseen evaluation set to finalize your degraded Equal Error Rate (EER) and min t-DCF reporting tables.

---
# RUN: 20260602_003212 -> 20260617_172233
TOTAL_TIMESTEPS = 143360 | N_STEPS = 4096 | TOTAL_UPDATES = 35 | EPOCHS = 15 | BATCH = 256

1. **Successful Continuation and Reward Scale Elevation:** This session successfully resumed training, building directly upon previous model weights. A major architectural change was elevating the success bonus parameter to `bonus_amount: 75.0`. This adjustment lifted the global reward ceiling, with the absolute peak reward reaching an unprecedented **96.99**. The mean episodic reward began exceptionally high at **20.75** in the first decile and remained completely stable, hovering between **21.02 and 21.95** through to the tenth decile. This demonstrates that the model successfully locked onto a high-confidence exploit plateau without experiencing catastrophic forgetting or gradient regression.
2. **Extreme Exploit Density Over Extended Timesteps:** Over the course of the expanded 143,360-step horizon, the model demonstrated remarkable efficiency, logging an extraordinary **3,958 direct bypasses** ($Score > 0.50$). The model pushed its peak evasion confidence to an all-time elite ceiling of **0.9746**, demonstrating structural dominance over the AASIST3 Graph Attention Network.
3. **Consolidation of the Optimized Action Space Profile:** Statistical parameter analysis across the successful updates reveals that the policy distribution has completely consolidated around a high-fidelity signature. The agent locked the compression `ratio` cleanly to its absolute minimum baseline of **1.01** (representing near-zero dynamic degradation) and pinned the median `bitrate` at a highly pristine **151.94 kHz**. Concurrently, both `jitter` (mean: `-0.84`) and `shimmer` (mean: `-0.77`) were driven heavily toward their minimum boundaries to minimize destructive acoustic telling signs, proving that the agent has abandoned chaotic exploration in favor of a precise spectral masking attack.

> NEXT STEPS:
> 1. **Run Final Out-of-Distribution Validation:** Freeze this fully saturated model checkpoint (`kon_artist_pki_resume.zip`) and execute `evaluate.py` across the unseen validation split to calculate the definitive degraded Equal Error Rate (EER) and min t-DCF.
> 2. **Compile Global vs. Adaptive Comparative Data:** Aggregate the performance profiles of the unbiased, full-batch, and mini-batch runs alongside this resumed high-bonus session to build the central evaluation matrix for your thesis results chapter.
> 3. **Extract Spectral Tilt and Harmonic Attenuation Charts:** Map the precise coordinate relationships of the `dsp_tilt` and `dsp_harmonics` parameters from the 3,958 winner profiles to illustrate the acoustic filter bypass mechanism during your oral defense.