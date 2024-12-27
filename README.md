# STP Challenge 2025: Soccer Trajectory Prediction Competition

## Overview  
The STP Challenge 2025 aims to predict the movement trajectories of players and the ball from a few seconds before a goal is scored until the goal occurs during soccer matches

---

## Dataset  
We provide a dataset generated using the RoboCup Soccer Simulator. This dataset includes over 2,000 matches played by 10 teams from RoboCup 2024's round-robin tournament, totaling more than 15,000 hours of game time.  

**Prediction Target:** The evaluation will be conducted on newly played matches generated using teams included in this dataset.

**Data Source:** The complete dataset is available at: [https://github.com/hidehisa.akiyama/RoboCup2D-data/](https://github.com/hidehisa.akiyama/RoboCup2D-data/)

---

## Official Website  
[STP Challenge 2025 Official Site (in Japanese)](https://sites.google.com/view/stp-challenge/)

---

## How to Use  

### 1. Data Download  

To download the dataset, run the following command (if `debug = True`, only 5 games for each url will be downloaded):  
```bash
python download.py --subpaths rc2021-roundrobin/normal/alice2021-helios2021 rc2021-roundrobin/normal/alice2021-hfutengine2021
```

### 2. Training, Testing, and Evaluation
Run `main.py` for training, testing, and evaluation (you cannot run with fewer games in the current train/val/test splitting):

```bash
python main.py --n_epoch 10 --model RNN
```

You can use `--Sanity` and `--TEST` options for sanity check using velocity model and only test without training, respectively.

### 3. Challenge Data Inference
The pseudo-challenge data and corresponding ground truth are provided in `./test_samples/input` and `./test_samples/gt`, respectively.

Run the following command for inference and evaluation using the (pseudo) challenge dataset:

```bash
python main.py --n_epoch 10 --model RNN --challenge_data ./test_samples/input
```

### 4. Submission Evaluation
To evaluate your submission, run:

```bash
python evaluation.py --submit ./results/test/submission --gt ./test_samples/gt --input ./test_samples/input
```
