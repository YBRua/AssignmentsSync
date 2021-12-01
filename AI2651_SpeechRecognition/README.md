# AI2651 Project 1: Voice Activity Detection
Project 1 for AI2651 Intelligent Speech Recognition.
## File Structure
- `VAD`
  - `Classifiers`
    - `basic`: basic threshold classifier (task 1)
    - `dualGMM`: GMM-based classifier (task 2)
  - `features`
    - `short_time_features`: short time feature extractors (task 1)
    - `spectral_features`: spectral feature extractors (task 2)
  - `task1`: pipelines for task 1
  - `task2`: pipelines for task 2
  - `data_loader`: functions for loading training and test sets
  - `evaluate`: auc and eer metrics (provided by TA)
  - `short_time_analysis`: task 1
  - `vad_utils`: provided by TA
- `feature_engineering`: selecting features for task 2
- `model_selection`: parameter tuning for task 2
- `parameter_tune`: parameter tuning for task 1
- `task1`: task 1 main
- `task2`: task 2 main

## Task 1: Simple Classifier
Detect voice activity using short-time features of voice signals and basic linear classifiers.

### Progress
> Finished!

- [x] short-time energy
- [x] short-time magnitude
- [x] short-time zero-crossing rate
- [x] basic short-time feature extraction
- [x] short-time Fourier transform
- [x] feature extraction pipeline
- [x] estimate average values of features, to be used as thresholds
- [x] find and implement a proper classifier
- [x] Reference book: Bayes-based VAD
- [x] Parameter tuning
- [x] How to use ZCR and Low-freq Energy
- [x] doc strings for functions
- [x] Technical report.

## Task 2: ML Classifiers
Detect voice activity using spectral features of voice signals and statistic machine learning classifiers.

### Progress
> Done! No more VAD!

- [x] MFCC feature extraction
- [x] GMMHMM test run.
  - GMM: AUC 0.86 (trained on dev set)
  - AUC 0.9173 when using `predict_proba`
- [x] GMM test run.
  - GMMHMM: AUC 0.87 (trained on dev set)
  - AUC 0.96 when using `predict_proba`
- [x] DualGMM classifier.
  - 0.94 AUC (trained on dev set)
- [x] Optimize file structure.
- [x] Training and evaluation.
  - [x] Feature Engineering
    - Note: feature engineering is run on 1500 samples from training set.
- [x] Model selection.
- [x] Run on test set.
- [x] Technical Report.

### Memo
- Feature extraction on training set is SLOW.
- `n_mfcc` matters.
- `delta` matters. 1st- and 2nd-order or 2nd- and 3rd-order? Or all three orders?

### Feature Engineering
> feature engineering runned on 1500 samples of training set.

|     Features          |   AUC  |  EER   | Iters To Converge  |
| :-------------------: | :----: | :----: | :----------------: |
| 8_mfcc, Order 1,2     | 0.9519 | 0.0613 |      10 / 60       |
| 8_mfcc, Order 2,3     | 0.9527 | 0.0565 |      20 / 40       |
| 8_mfcc, Order 1,2,3   | 0.9545 | 0.0489 |      20 / 70       |
| 12_mfcc, Order 1,2    | 0.9513 | 0.0627 |      20 / 80       |
| 12_mfcc, Order 2,3    | 0.9545 | 0.0490 |      50 / 60       |
| 12_mfcc, Order 1,2,3  | 0.9532 | 0.0508 |      30 / 80       |
| 15_mfcc, Order 1,2    | 0.9536 | 0.0590 |      40 / 70       |
| 15_mfcc, Order 2,3    | 0.9434 | 0.0711 |      40 / 210      |
| 15_mfcc, Order 1,2,3  | 0.9388 | 0.0855 |      10 / 210      |
| 20_mfcc, Order 1,2    | 0.9380 | 0.0918 |      30 / 200      |
| 20_mfcc, Order 2,3    | 0.9419 | 0.0726 |      40 / 180      |
| 20_mfcc, Order 1,2,3  | 0.9384 | 0.0852 |      40 / 180      |

### Model Selection
> Benchmarked using 12_mfcc with 2nd and 3rd order deltas

|    Model    | AUC on Dev |
| :---------: | :--------: |
| 1 Gaussian  |   0.9371   |
| 2 Gaussian  |   0.9442   |
| 3 Gaussian  |   0.9452   |
| 5 Gaussian  |   0.9422   |
| 7 Gaussian  |   0.9441   |
| 10 Gaussian |   0.9488   |
| 12 Gaussian |   0.9492   |

Other attempts failed due to not enough memory. I'm poor at producing quality code :cry:.
