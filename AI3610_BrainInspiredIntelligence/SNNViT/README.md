# MNIST-ViT Spiking Neural Network Implementation

> Powered by [PyTorch](https://pytorch.org/) and [SnnTorch](https://snntorch.readthedocs.io/en/latest/index.html)

This repository implements a simple Visual-Transformer (ViT) and its SNN counterpart. The models are able to be trained on MNIST and CIFAR10 dataset.

## Quick Start

### Run ANN ViT

```sh
>>> python main.py --dataset <dataset>
```

- `<dataset>` should be one of `mnist` or `cifar10`
  
### Run SNN ViT

```sh
>>> python main.py --dataset <dataset> -snn
```

## Commandline Args

For detailed documentation on commandline arguments, please check `args.py`, or use

```sh
>>> python main.py --help
```

## Performance

> Models are trained for 10 epochs
> We report the performance of *the final epoch on test set*.

Note that the CIFAR10 dataset is used for DEMO only. The architecture of the ViT is tuned for MNIST, not CIFAR, and therefore the performance on CIFAR10 is relatively poor. Despite this, however, the performance of ANN and SNN are similar.

| Dataset | Model | Test Acc |
| :-----: | :---: | :------: |
|  MNIST  |  ANN  |  0.9407  |
|  MNIST  |  SNN  |  0.9291  |
|  CIFAR  |  ANN  |  0.5070  |
|  CIFAR  |  SNN  |  0.4037  |
