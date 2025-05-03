# STP Challenge 2025: Soccer Trajectory Prediction Competition 

## Overview 
The STP Challenge 2025 aims to predict the movement trajectories of players and the ball from a few seconds before a goal is scored until the goal occurs during soccer matches

## Dataset  
We provide a dataset generated using the RoboCup Soccer Simulator. This dataset includes over 2,000 matches played by 10 teams from RoboCup 2024's round-robin tournament, totaling more than 15,000 hours of game time. 

We host two distinct competitions:

- [STP-Challenge-Japan 2025](./STP-Challenge-Japan-2025.md): Completed in March 2025.
- [STP-World-Challenge 2025](./STP-World-Challenge-2025.md): (details TBD).

Please select the appropriate challenge for detailed instructions and resources.

## General Dataset Information
Both challenges utilize the same dataset, generated from RoboCup Soccer Simulator matches. It includes over 2,000 matches with 10 teams from RoboCup 2024, totaling over 15,000 hours of game data.

- [Google Drive Dataset (Recommended)](https://drive.google.com/drive/folders/1hiXe4Vyj79FQS8tS_fCvnhaYBM7ezEzy?usp=sharing)
- [Original Source](https://github.com/hidehisaakiyama/RoboCup2D-data/)

## How to Use 

### 0. Sample trajectory prediction code on Google Colab (Apr 8, 2025)

https://colab.research.google.com/drive/1kHYEbbcERr0MhEBGNENM2aqdTJg_27QU?usp=sharing

To use larger data, run below. 

### 1. Data Download  

(Recommended, added on Apr 8, 2025) To download data from [google drive](https://drive.google.com/drive/folders/1hiXe4Vyj79FQS8tS_fCvnhaYBM7ezEzy?usp=sharing), run the code such that
```bash
python download_from_gdrive.py --subpaths aeteam2024-cyrus2024 aeteam2024-fra2024
```

### 2. Training, Testing, and Evaluation
Run `main.py` for training, testing, and evaluation (you cannot run with fewer games in the current train/val/test splitting):

```bash
python main.py --n_epoch 10 --model RNN
```

You can use `--Sanity` and `--TEST` options for sanity check using velocity model and only test without training, respectively.

### 3. Challenge Data Inference (Initial version: Dec 9, 2024)
The pseudo-challenge data and corresponding ground truth are provided in `./test_samples/input` and `./test_samples/gt`, respectively.

Run the following command for inference and evaluation using the (pseudo) challenge dataset:

```bash
python main.py --n_epoch 10 --model RNN --challenge_data ./test_samples/input
```

### 4. Submission Evaluation (Initial version: Dec 9, 2024)
To evaluate your submission, run:

```bash
python evaluation.py --submit ./results/test/submission --gt ./test_samples/gt --input ./test_samples/input
```