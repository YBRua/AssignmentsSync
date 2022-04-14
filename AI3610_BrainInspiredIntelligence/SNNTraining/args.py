from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--model', choices=['Linear', 'MLP', 'Conv'],
        default='Linear',
        help='Model architecture')
    parser.add_argument(
        '--data_path', type=str, default='./data/',
        help='Path to dataset')
    parser.add_argument(
        '--model_save', type=str, default='model.pt',
        help='File name for saved model')
    parser.add_argument(
        '--trainer', choices=['ADAM', 'RANDOM', 'SPSA'],
        default='ADAM', help='Training method for model')
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of epochs for training')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size for training')
    parser.add_argument(
        '--snn', action='store_true',
        help='Whether to use SNN architecture')
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use.')
    parser.add_argument(
        '--beta', type=float, default=0.9,
        help='SNN exclusive. Decay rate for LIF layers')
    parser.add_argument(
        '--time_steps', type=int, default=5,
        help='SNN exclusive. Number of time steps')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate')
    parser.add_argument(
        '--perturb_step', type=float, default=0.005,
        help='Blackbox exclusive. Scaling factor for perturbations')
    parser.add_argument(
        '--rate_encoding', action='store_true',
        help='SNN exclusive. Whether to use rate encoding')
    parser.add_argument(
        '--test_batch_size', type=int, default=128,
        help='Batch size for testing')
    parser.add_argument(
        '--seed', type=int, default=1337,
        help='Random seed.')
    parser.add_argument(
        '--random_search_steps', type=int, default=10,
        help='RANDOM trainer exclusive. Number of random search steps.')
    parser.add_argument(
        '--spsa_use_adam', action='store_true',
        help='Whether to use ADAM as the optimizer for SPSA.')
    parser.add_argument(
        '--pretrained_init', type=str, default='',
        help='Path to pretrained model. Initialize with pretrained ANN')
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Whether to skip training and run tests directly.')
    parser.add_argument('--loss', choices=['mse', 'ce'], default='ce')

    return parser.parse_args()


def print_args(args):
    print(f'Seed: {args.seed}')
    print(f'Epochs: {args.epochs}')
    print(f'Model: {args.model}')

    print(f'Trainer: {args.trainer}')
    if args.trainer in ['RANDOM', 'SPSA']:
        print(f'  Perturb step: {args.perturb_step}')

    print(f'Learning rate: {args.lr}')

    print(f'SNN: {args.snn}')
    if args.snn:
        print(f'  Rate encoding: {args.rate_encoding}')
        print(f'  Beta: {args.beta}')
        print(f'  Time steps: {args.time_steps}')
