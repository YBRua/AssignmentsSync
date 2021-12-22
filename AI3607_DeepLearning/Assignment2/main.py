import sys
import random
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
import data_loaders as ldrs
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from tqdm import tqdm
from model import LeNet


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--mode', '-m', type=str,
        choices=['vanilla', 'dropout', 'tanking'],
        default='vanilla',
        help='Training mode. Vanilla: standard training.'
        + ' Dropout: Drops 90% 01234.'
        + ' Tanking: Drops 90% 01234, samples 10% 56789.')
    parser.add_argument(
        '--epoch', '-e', type=int, default=10,
        help='Training epoches. Default 10.')
    parser.add_argument(
        '--batch_size', '-b', type=int, default=64,
        help='Batch size. Default 64.')
    parser.add_argument(
        '--seed', '-s', type=int, default=416,
        help='Random seed.')
    parser.add_argument(
        '--fancy', '-f', dest='fancy', action='store_true',
        help='Whether to plot fancy grapics for loss and acc.')

    return parser.parse_args()


def train_one_epoch(
        epoch: int,
        model: nn.Layer,
        optimizer: optim.Optimizer,
        loader: ldrs.BaseLoaderHelper):
    train_loader, test_loader = loader()
    batch_counter = 0
    trn_acc = 0
    tst_acc = 0
    tot_loss = 0
    batch_progress = tqdm(train_loader)
    for x, y in batch_progress:
        batch_counter += 1
        pred = model.forward(x)
        loss = F.cross_entropy(pred, y)
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        acc = paddle.metric.accuracy(pred, y)
        loss_val = loss.numpy().item()
        acc_val = acc.numpy().item()
        tot_loss += loss_val
        trn_acc += acc_val
        batch_progress.set_description(
            f'E: {epoch:2d}; B: {batch_counter:3d}; '
            + f'Loss: {tot_loss / batch_counter:.4f}; '
            + f'Acc: {trn_acc / batch_counter:.4f}')
    tot_loss /= batch_counter
    trn_acc /= batch_counter

    model.eval()
    batch_counter = 0
    for x, y in test_loader:
        batch_counter += 1
        pred = model.forward(x)
        test_acc = paddle.metric.accuracy(pred, y)
        tst_acc += test_acc.numpy().item()
    print(f'Test Acc: {tst_acc / batch_counter:.4f}', file=sys.stderr)
    tst_acc /= batch_counter
    model.train()

    return tot_loss, trn_acc, tst_acc


def main():
    args = parse_args()
    SEED = args.seed
    EPOCHES = args.epoch
    BATCH_SIZE = args.batch_size
    MODE = args.mode

    model = LeNet()
    optimizer = optim.Adam(parameters=model.parameters())
    paddle.seed(SEED)
    random.seed(SEED)

    if MODE == 'vanilla':
        loader = ldrs.VanillaLoader(batch_size=BATCH_SIZE)
    elif MODE == 'dropout':
        loader = ldrs.DropoutLoader(batch_size=BATCH_SIZE)
    else:
        loader_2 = ldrs.TankingLoader(batch_size=BATCH_SIZE)
        loader_1 = ldrs.DropoutLoader(batch_size=BATCH_SIZE)
        # ensure the two loaders are using the same 10% 01234
        loader_1.train_loader = loader_2.train_loader

    model.train()
    losses = []
    trn_accs = []
    tst_accs = []
    for e in range(EPOCHES):
        if MODE == 'tanking':
            loader = loader_1 if e < 7 else loader_2
        loss, trn_acc, tst_acc = train_one_epoch(e, model, optimizer, loader)
        losses.append(loss)
        trn_accs.append(trn_acc)
        tst_accs.append(tst_acc)
    if args.fancy:
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        ax[0].plot(losses)
        ax[0].set_title('Train Losses')
        ax[1].plot(trn_accs)
        ax[1].set_title('Train Accs')
        ax[2].plot(tst_accs)
        ax[2].set_title('Test Accs')
        for i in range(3):
            ax[i].set_xlabel('Epoches')
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
