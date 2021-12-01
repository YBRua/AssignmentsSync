import paddle
import numpy as np
import seaborn as sns
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm


class Prototype(nn.Layer):
    def __init__(self):
        super(Prototype, self).__init__()
        self.input = nn.Linear(1, 512)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, inputs):
        x = self.input(inputs)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        y = self.out(x)

        return y


def target_func(x):
    """x^3 + 2x^2 - 8x + 1"""
    return x ** 3 + 2 * x ** 2 - 8 * x + 1


def generate_data(seed=114514, low=-3.0, high=3.0, size=1000, train_prop=0.8):
    np.random.seed(seed)
    assert train_prop < 1, f'Training set proportion {train_prop} should < 1.'
    train_size = int(size * train_prop)
    test_size = size - train_size
    train_set = np.random.uniform(
        low, high, (train_size, 1)).astype(np.float32)
    test_set = np.random.uniform(
        low, high, (test_size, 1)).astype(np.float32)

    return train_set, test_set


def main():
    N_EPOCHS = 300
    BATCH_SIZE = 50

    train_set, test_set = generate_data()

    model = Prototype()
    model.train()
    optimizer = optim.SGD(learning_rate=0.01, parameters=model.parameters())

    t = tqdm(range(N_EPOCHS))
    for e in t:
        np.random.shuffle(train_set)
        batches = [
            train_set[i:i+BATCH_SIZE]
            for i in range(0, len(train_set), BATCH_SIZE)]

        model.train()
        train_loss = []
        for b, x in enumerate(batches):
            y = target_func(x)
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)

            pred = model(x)
            loss = F.loss.square_error_cost(pred, y)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            train_loss.append(avg_loss.numpy()[0])
        avgloss_train = np.mean(train_loss)

        model.eval()
        test_y = paddle.to_tensor(target_func(test_set))
        test_x = paddle.to_tensor(test_set)
        pred_test = model(test_x)
        test_loss = paddle.mean(
            F.loss.square_error_cost(pred_test, test_y)).numpy()
        t.set_description(
            'Epoch {:3d} Train {:.4f} Test {:.4f}'.format(
                e+1, avgloss_train, test_loss[0]))


    X = paddle.linspace(-3, 3, 1000)
    model.eval()
    Y = model(paddle.unsqueeze(X, -1))
    TRU = target_func(X.numpy())

    sns.set_theme(style='white', palette='pastel')

    fig, ax = plt.subplots()
    ax.plot(X.numpy(), Y.numpy(), label='Model Pred')
    ax.plot(X.numpy(), TRU, label='Ground Truth', ls=':')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    sns.despine(fig=fig, ax=ax)
    fig.savefig('./510930910374_杨博睿_result.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
