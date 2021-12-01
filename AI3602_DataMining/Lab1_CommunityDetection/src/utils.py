from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset_path', '-d', type=str,
        default='./data/edges_update.csv',
        help='Path to dataset (csv file)')
    parser.add_argument(
        '--output', '-o', type=str,
        default='./src/res.csv',
        help='Path of output file (csv format)')

    return parser.parse_args()
