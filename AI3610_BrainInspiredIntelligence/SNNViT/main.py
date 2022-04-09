import torch
import torch.nn as nn
import torch.optim as optim

import snntorch.spikegen as spikegen
import snntorch.functional as SF
import snntorch.surrogate as surrogate

from tqdm import tqdm
from torch.utils.data import DataLoader

import data_prep
from args import parse_args
from models import ViT, SNNViT


def train(
        epoch: int,
        model: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        device: torch.device):

    model.train()

    tot_loss = 0
    tot_acc = 0
    prog = tqdm(train_loader)
    for bid, (x, y) in enumerate(prog):
        bid += 1
        x = x.to(device)

        if IS_SNN:
            x = spikegen.rate(x, STEPS)

        y = y.to(device)

        optimizer.zero_grad()
        if IS_SNN:
            out, _ = model(x)
        else:
            out = model(x)
        loss_value = loss(out, y)
        loss_value.backward()
        optimizer.step()

        if IS_SNN:
            acc = SF.accuracy_rate(out, y)
        else:
            pred = out.argmax(dim=1)
            acc = (pred == y).float().mean()

        tot_acc += acc.item()
        tot_loss += loss_value.item()

        avg_acc = tot_acc / bid
        avg_loss = tot_loss / bid

        prog.set_description(f'| Epoch {epoch} | Loss {avg_loss:.4f} | Acc {avg_acc:.4f} |')


def evaluate(
        epoch: int,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device):
    print(f'| Epoch {epoch} | Running Evaluation |')
    model.eval()

    with torch.no_grad():
        tot_acc = 0
        tot_size = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            if IS_SNN:
                x = spikegen.rate(x, STEPS)

            if IS_SNN:
                out, _ = model(x)
            else:
                out = model(x)

            if IS_SNN:
                acc = SF.accuracy_rate(out, y) * y.shape[0]
            else:
                pred = out.argmax(dim=1)
                acc = (pred == y).float().sum()

            tot_acc += acc.item()
            tot_size += y.shape[0]

        print(f'| Epoch {epoch} | Test Acc {tot_acc / tot_size:.4f} |')


if __name__ == '__main__':
    args = parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DATASET_PATH = args.dataset_path

    IS_SNN = args.is_snn

    BETA = args.beta
    STEPS = args.steps
    DEVICE = torch.device(args.device)

    spike_grad = surrogate.fast_sigmoid()

    # ViT Architecture parameters
    IMG_SIZE = 28
    N_CHANNELS = 1
    PATCH_SIZE = 4
    EMBEDDING_DIM = 64
    N_CLASSES = 10
    N_HEADS = 4

    if IS_SNN:
        print('Initializing SNN ViT')
        model = SNNViT(
            IMG_SIZE,
            N_CHANNELS,
            PATCH_SIZE,
            EMBEDDING_DIM,
            N_CLASSES,
            N_HEADS,
            BETA,
            spike_grad,
            STEPS)
    else:
        print('Initializing ANN ViT')
        model = ViT(
            img_size=28,
            n_channels=1,
            patch_size=4,
            embedding_dim=64,
            n_classes=10,
            nhead=4,)
    model.to(DEVICE)

    loss = SF.ce_count_loss() if IS_SNN else nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_set, test_set = data_prep.load_mnist(DATASET_PATH)
    train_loader, test_loader = data_prep.wrap_dataloaders(train_set, test_set, batch_size=BATCH_SIZE)

    for e in range(EPOCHS):
        train(e, model, loss, optimizer, train_loader, DEVICE)
        evaluate(e, model, test_loader, DEVICE)
