import sys
import random
import paddle
import paddle.optimizer as optim
import paddle.nn.functional as F
import data_loaders as ldrs

from tqdm import tqdm

from model import LeNet

SEED = 10374
EPOCHES = 15
BATCH_SIZE = 64

model = LeNet()
optimizer = optim.Adam(parameters=model.parameters())

paddle.seed(SEED)
random.seed(SEED)

loader_2 = ldrs.TankingLoader(batch_size=BATCH_SIZE)
loader_1 = ldrs.DropoutLoader(batch_size=BATCH_SIZE)
loader_1.train_loader = loader_2.train_loader

model.train()
for e in range(EPOCHES):
    loader = loader_1 if e < 5 else loader_2
    train_loader, test_loader = loader()
    batch_counter = 0
    tot_acc = 0
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
        tot_acc += acc_val
        batch_progress.set_description(
            f'E: {e:2d}; B: {batch_counter:3d}; '
            + f'Loss: {tot_loss / batch_counter:.4f}; '
            + f'Acc: {tot_acc / batch_counter:.4f}')

    model.eval()
    tot_acc = 0
    batch_counter = 0
    for x, y in test_loader:
        batch_counter += 1
        pred = model.forward(x)
        test_acc = paddle.metric.accuracy(pred, y)
        tot_acc += test_acc.numpy().item()
    print(f'Test Acc: {tot_acc / batch_counter:.4f}', file=sys.stderr)
    model.train()
