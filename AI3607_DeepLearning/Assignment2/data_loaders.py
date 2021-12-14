import random
import paddle.vision.transforms as trans
from paddle.vision.datasets import MNIST

from paddle.io import DataLoader, Dataset


class ThinkSet(list, Dataset):
    pass


class BaseLoaderHelper():
    def __init__(self, batch_size=128, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _get_paddle_mnist(self):
        transform = trans.Compose([
            trans.Normalize(mean=[127.5], std=[127.5], data_format='CHW')])

        train_dataset = MNIST(mode='train', transform=transform)
        test_dataset = MNIST(mode='test', transform=transform)

        return train_dataset, test_dataset

    def _wrap_dataloaders(self, train, test):
        train_loader = DataLoader(
            train, batch_size=self.batch_size, shuffle=self.shuffle)
        test_loader = DataLoader(
            test, batch_size=self.batch_size, shuffle=self.shuffle)

        return train_loader, test_loader

    def __call__(self):
        return None, None


class VanillaLoader(BaseLoaderHelper):
    def __init__(self, batch_size=128, shuffle=True):
        super().__init__(batch_size=batch_size, shuffle=shuffle)

    def __call__(self):
        train, test = self._get_paddle_mnist()
        return self._wrap_dataloaders(train, test)


class DropoutLoader(BaseLoaderHelper):
    def __init__(
            self,
            batch_size=128,
            shuffle=True,
            proportion=0.9,
            drops=set(range(5))):
        super().__init__(batch_size=batch_size, shuffle=shuffle)
        self.proportion = proportion
        self.drops = drops

    def _data_dropout(self, dataset):
        dropped = [
            (x, y) for (x, y) in dataset
            if y not in self.drops or random.random() >= self.proportion]

        return ThinkSet(dropped)

    def __call__(self):
        train, test = self._get_paddle_mnist()
        train = self._data_dropout(train)
        return self._wrap_dataloaders(train, test)


class RandomLabeledLoader(DropoutLoader):
    def __init__(
            self,
            batch_size=128,
            shuffle=True,
            proportion=0.9,
            drops=set(range(5))):
        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            proportion=proportion,
            drops=drops)

    def _shuffle_labels(self, dataset):
        dataset = [(x, random.randint(0, 9)) for (x, _) in dataset]
        return ThinkSet(dataset)

    def __call__(self):
        train, test = self._get_paddle_mnist()
        train = self._data_dropout(train)
        train = self._shuffle_labels(train)

        return self._wrap_dataloaders(train, test)
