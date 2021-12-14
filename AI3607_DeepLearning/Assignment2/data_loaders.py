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
        self.train, self.test = self._get_paddle_mnist()

    def _get_paddle_mnist(self):
        transform = trans.Compose([
            trans.Normalize(mean=[127.5], std=[127.5], data_format='CHW')])

        train_dataset = MNIST(mode='train', transform=transform)
        test_dataset = MNIST(mode='test', transform=transform)

        return train_dataset, test_dataset

    def _wrap_dataloaders(self, dataset):
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return loader

    def __call__(self):
        return None, None


class VanillaLoader(BaseLoaderHelper):
    def __init__(self, batch_size=128, shuffle=True):
        super().__init__(batch_size=batch_size, shuffle=shuffle)
        self.train_loader = self._wrap_dataloaders(self.train)
        self.test_loader = self._wrap_dataloaders(self.test)
            

    def __call__(self):
        return self.train_loader, self.test_loader


class DropoutLoader(BaseLoaderHelper):
    def __init__(
            self,
            batch_size=128,
            shuffle=True,
            proportion=0.9,
            drops=list(range(5))):
        super().__init__(batch_size=batch_size, shuffle=shuffle)
        self.proportion = proportion
        self.drops = drops
        self.dropped_train = self._data_dropout(self.train)
        self.train_loader = self._wrap_dataloaders(
            ThinkSet(self.dropped_train))
        self.test_loader = self._wrap_dataloaders(self.test)

    def _data_dropout(self, dataset):
        dropped = [
            (x, y) for (x, y) in dataset
            if y not in self.drops or random.random() >= self.proportion]

        return dropped

    def __call__(self):
        return self.train_loader, self.test_loader


class TankingLoader(DropoutLoader):
    def __init__(
            self,
            batch_size=128,
            shuffle=True,
            proportion=0.9,
            drops=list(range(5))):
        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            proportion=proportion,
            drops=drops)

    def _sample_training_set(self):
        resampled = [
            (x, y) for (x, y) in self.train
            if y not in self.drops and random.random() >= self.proportion]

        return resampled + self.dropped_train

    def __call__(self):
        resampled_train = ThinkSet(self._sample_training_set())
        return self._wrap_dataloaders(resampled_train), self.test_loader


class RandomLabelLoader(DropoutLoader):
    def __init__(
            self,
            batch_size=128,
            shuffle=True,
            proportion=0.9,
            drops=list(range(5))):
        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            proportion=proportion,
            drops=drops)

    def _shuffle_labels(self, dataset):
        dataset = [(x, random.randint(0, 9)) for (x, _) in dataset]
        return ThinkSet(dataset)

    def __call__(self):
        pass
