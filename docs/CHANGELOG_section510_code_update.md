# Section 5.10 code-aligned update

This update aligns the benchmark starter files with the uploaded Experiment 5.10 code.

Exact preset definitions were taken from `exp510_dataset.py`:
- split_A_current
- split_B_shifted
- split_C_shifted

The default runner (`run_exp510_all.sh`) executes all three presets with seeds:
42, 52, 62, 72, 82

The training script `exp510_train.py` uses:
- epochs = 60
- batch size = 32
- Adam(lr=2e-4, weight_decay=1e-4)
- HDA_CNN model
- n_train_aug = 10
- n_test_repeat = 10
