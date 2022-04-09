from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset_path', type=str, default='./data/',
        help='Path to the dataset.')
    parser.add_argument(
        '--batch_size', '-b', type=int, default=64,
        help='Batch size.')
    parser.add_argument(
        '--epochs', '-e', type=int, default=10,
        help='Number of epochs.')
    parser.add_argument(
        '--is_snn', '-snn', action='store_true',
        help='Use SNN verison ViT if enabled.')
    parser.add_argument(
        '--beta', type=float, default=0.9,
        help='SNN Exclusive. Decay parameter in LIF (snn.Leaky) model')
    parser.add_argument(
        '--steps', type=int, default=5,
        help='SNN exclusive. Number of time steps')
    parser.add_argument(
        '--device', '-d', type=str, default='cuda',
        help='Device to use.')

    return parser.parse_args()
