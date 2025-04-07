# STP Challenge 2025: Soccer Trajectory Prediction Competition 

## Overview  
The STP Challenge 2025 aims to predict the movement trajectories of players and the ball from a few seconds before a goal is scored until the goal occurs during soccer matches

---

## Dataset  
We provide a dataset generated using the RoboCup Soccer Simulator. This dataset includes over 2,000 matches played by 10 teams from RoboCup 2024's round-robin tournament, totaling more than 15,000 hours of game time.  

**Prediction Target:** The evaluation will be conducted on newly played matches generated using teams included in this dataset.

**Data Source:** The complete dataset is available at: [https://github.com/hidehisaakiyama/RoboCup2D-data/](https://github.com/hidehisaakiyama/RoboCup2D-data/)

---

## Official Website  
[STP Challenge 2025 Official Site (in Japanese)](https://sites.google.com/view/stp-challenge/)

---

## How to Use 

### 0. Sample trajectory prediction code on Google Colab (Apr 8, 2025)

https://colab.research.google.com/drive/1kHYEbbcERr0MhEBGNENM2aqdTJg_27QU?usp=sharing

To use larger data, run below. 

### 1. Data Download  

(Recommended, added on Apr 8, 2025) To download data from google drive, run the code such that
```bash
python download_from_gdrive.py --subpaths aeteam2024-cyrus2024 aeteam2024-fra2024
```

(Not recommended, original)
To download the dataset from the original server, run the following command (if `debug = True`, only 5 games for each url will be downloaded):  
```bash
python download.py --baseurls \
  https://alab.ise.ous.ac.jp/robocupdata/rc2024-roundrobin/ \
  https://alab.ise.ous.ac.jp/robocupdata/rc2021-roundrobin/ \
  --subpaths normal/alice2021-helios2021 normal/alice2021-hfutengine2021
```
(Modified the code on Feb 12, 2025, to reduce server load)
(Modified the code on Mar 31, 2025, to modify dataset links)

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

### 5. Submission Evaluation (Second (official) version: Feb 1, 2025)
Another pseudo-challenge data is set to `./example`, and run:

```bash
python example/evaluation.py --gt ./example/ground-truth --input ./example/test-data --submit ./example/submission
```

### 6. The True Challenge Set (Feb 1, 2025)
See: 
[STP Challenge 2025 Official Compeititon Rule (in Japanese)](https://sites.google.com/view/stp-challenge/%E7%AB%B6%E6%8A%80%E3%83%AB%E3%83%BC%E3%83%AB)

### 7. Validation of Submission File (Feb 18, 2025)
To validate your submission file, run:

```bash
python example/validate_submission.py --input ./example/test-data --submit ./example/submission
```
