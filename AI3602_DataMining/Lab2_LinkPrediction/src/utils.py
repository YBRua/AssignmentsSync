import numpy as np
import pandas as pd
from argparse import ArgumentParser


def load_edges(path: str) -> np.ndarray:
    return pd.read_csv(path).to_numpy()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--epoch', '-e', type=int, default=50,
        help='Number of training epoches')
    parser.add_argument(
        '--batchsize', '-b', type=int, default=128,
        help='Batch size')
    parser.add_argument(
        '--neg_samples', '-n', type=int, default=12,
        help='Number of negative samples when computing loss')
    parser.add_argument(
        '--walker_return_param', type=float, default=1,
        help='Return parameter for the biased random walker.'
        + ' Determines the likelihood it returns to previous node.')
    parser.add_argument(
        '--walker_io_param', type=float, default=1,
        help='In/Out parameter for the biased random walker.'
        + ' Determines the likelihood it moves further away.')
    parser.add_argument(
        '--walk_length', type=int, default=15,
        help='Length for each random walk')
    parser.add_argument(
        '--window_size', type=int, default=5,
        help='Window size for a neighbourhood of nodes in the walk trajectory')
    parser.add_argument(
        '--lr', type=float, default=0.1,
        help='Learning rate for SGD')
    parser.add_argument(
        '--mmt', type=float, default=0.9,
        help='Momentum for SGD')
    parser.add_argument(
        '--device', '-d', type=str, default='cuda:0',
        help='PyTorch style device to run the model')
    parser.add_argument(
        '--dataset_path', type=str,
        default='./data/lab2_edge.csv',
        help='Path to dataset (csv file)')
    parser.add_argument(
        '--testset_path', type=str,
        default='./data/lab2_test.csv',
        help='Path to test set (csv file)')
    parser.add_argument(
        '--file_output', '-o', type=str,
        default='./prediction.csv',
        help='Path of output prediction result')
    parser.add_argument(
        '--model_save', '-s', type=str,
        default='./baseline.pt',
        help='Path to save the model')
    parser.add_argument(
        '--pretrained_path', '-p', type=str,
        default='',
        help='Path to a pretrained embedding model (.pt file)')
    parser.add_argument(
        '--split_dataset', dest='split_dataset',
        action='store_true',
        help='Whether to split the dataset into training and evaluation.')
    parser.add_argument(
        '--fancy_graphics', '-f', dest='fancy',
        action='store_true',
        help='Whether to plot fancy loss and auc curves.')

    return parser.parse_args()
