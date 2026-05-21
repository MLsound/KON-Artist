***Q: explain me how is reward managed between every core. every worker starts with the same settings and different mini-batch? best score wins? is it an average? how is the logic?***

1. Worker Setup & Data (The "Mini-Batches")
  Every worker (core) runs an identical copy of the environment settings, but they do not process the same data.
  In scripts/train.py (make_env function), each worker is assigned a unique rank which modifies its initialization seed (seed + rank). This
  ensures that each worker's data loader pulls a completely different sequence of audio samples from your dataset.

  During the rollout phase, each worker independently explores the environment for n_steps (100). With 7 workers, PPO collects a combined total
  of 700 steps (the TOTAL_ROLLOUT_BUFFER).

  2. Reward Management
  The rewards are managed completely independently for each worker during the rollout. 
  In src/env/wrappers.py (VecDetectorWrapper), the audio from all 7 workers is stacked into a single batch and sent to the AASIST3 model (on the
  CPU/GPU) simultaneously. 
  The model returns 7 independent scores. The wrapper then calculates the individual reward for each worker based on its specific score using
  compute_attack_reward.

  3. How PPO Uses the Rewards (No "Winner")
  It is neither a "best score wins" scenario nor a direct "average" of the rewards. Here is the RL logic:

   1. Collection: PPO collects the 700 independent experiences (State, Action, Reward, Next State) from all workers and pools them into a single
      massive buffer.
   2. Mini-batching: During the actual network update, PPO shuffles this massive buffer and grabs random mini-batches (size 512, as per your
      config) that contain mixed experiences from all 7 workers.
   3. Gradient Ascent (The "Average"): The neural network updates its weights to maximize the Expected Return (the average of the discounted
      future rewards) across all the experiences in that mini-batch. 

  Summary: The workers explore the audio space independently to provide diverse, parallel data. PPO pools all their experiences together and
  updates the agent's policy based on the average gradient of those combined, independent rewards.

  ---

  