# AI3607 Deep Learning and Its Applications Assignment 2

> MNIST

## Quick Start

### Requirements

- `paddle` 2.1.3
- `matplotlib`

### How to Run

To run training with vanilla setting, use

```sh
> python main.py -m vanilla
```

To run training with 90% 01234's dropped, use

```sh
> python main.py -m dropout
```

To run training with 90% 01234's dropped, and 10% 56789 sampled for balance, use

```sh
> python main.py -m tanking
```

## Commandline Arguments

- `--mode` `-m` Mode for training
  - `vanilla`: Standard training on the entire MNIST dataset
  - `dropout`: Drops 90% 01234 at random and run training
  - `tanking`: Drops 90% 01234 at random, then sample 10% 56789 at the beginning of each epoch to mitigate imbalanced training data
- `--epoch` `-e`: Epoches for training. Default `10`
- `--batch_size` `-b`: Batch size for training. Default `64`
- `--fancy`: If enabled, will plot curves for losses, training accs and test accs
