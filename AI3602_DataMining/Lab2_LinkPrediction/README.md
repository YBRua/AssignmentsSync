# AI3602 Data Mining Lab2 Link Prediction

> PyTorch implementation of deep walk

## Requirements

```txt
torch
numpy==1.21.2
scipy==1.5.2
pandas==1.1.3
```

## Quick Start

To run the algorithm with default config

```sh
python src/main.py
```

It should dump a trained model to current working directory, and write results to `prediction.csv`.

## Commandline Args

- `--epoch` `-e`
  - Number of training epoches. Default `50`
- `--batchsize` `-b`
  - Batch size. Default `128`
- `--neg_samples` `-n`
  - Number of negative samples when computing loss by negative sampling. Default `12`
- `--walker_return_param`
  - Return parameter for the biased random walker. Determines the likelihood it returns to previous node. Default `1.0`
- `--walker_io_param`
  - In/Out parameter for the biased random walker. Determines the likelihood it moves further away. Default `1.0`
- `--walk_length`
  - Length for each random walk. Default `15`
- `window_size`
  - Window size for creating a neighbourhood of nodes in the walk trajectory. Default `5`
- `--lr`
  - Learning rate for SGD. Default `0.1`
- `--mmt`
  - Momentum for SGD. Default `0.9`
- `--device` `-d`
  - Device for torch. Default `cuda:0`
- `--dataset_path`
  - Path to csv file containing edges. Default `./data/lab2_edge.csv`
- `--testset_path`
  - Path to test set containing edges. Default `./data/lab2_test.csv`
- `--file_output` `-o`
  - Path to save the results. Default `./prediction.csv`
- `--model_save` `-s`
  - Path to save model. Default `./baseline.pt`
- `--pretrained_path` `-p`
  - Path to a pretrained model (if exists)
- `--split_dataset`
  - If enabled, will split the dataset for training and evaluation
