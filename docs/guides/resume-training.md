# How to Resume Training
KON-Artist configuration (`configs/train_config.yaml`) has `load_last: true` under
logging.checkpoint, the `scripts/train.py` script is explicitly designed to look for the
latest `.zip` checkpoint and resume automatically.

1. **Zip the files back together:** Stable-Baselines3 natively expects these files to be bundled inside a single .zip file. You need to zip them together such that policy.pth, data, etc., are at the root of the zip file (not inside a subfolder).

You can do this via your terminal. Assuming you are in the directory containing those files:
```bash
zip -r kon_artist_ppo_resume.zip _stable_baselines3_version data policy.optimizer.pth policy.pth pytorch_variables.pth system_info.txt
```

2. **Place the `.zip` file in the checkpoints directory:** Move kon_artist_ppo_resume.zip into your project's checkpoint folder.
```bash
mv kon_artist_ppo_resume.zip ./outputs/checkpoints/
```

3. Run the training script:
Start training normally.
```bash
python -m scripts.train
```

The script will output something like:
```text
INFO [train:150] Checkpoint detected...
INFO [train:178] Resuming training from checkpoint:
outputs/checkpoints/kon_artist_ppo_resume.zip
```

It will seamlessly load the optimizer states (`policy.optimizer.pth`), the neural network
weights (`policy.pth`), and the training buffer/hyperparameters (data), and continue right
where it left off!