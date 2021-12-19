import random
import paddle.vision.transforms as trans
from paddle.vision.datasets import MNIST

from paddle.io import DataLoader, Dataset


class ThinkSet(list, Dataset):
    pass


class BaseLoaderHelper():
    def __init__(self, batch_size=128, shuffle=True):
        """The base for all data loaders. Should not be instantiated or used.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train, self.test = self._get_paddle_mnist()
        raise TypeError('This cls should not be instantiated!')

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
        """Standard dataloader that loads the entire MNIST when called

        Args:
            batch_size (int): Defaults to 128.
            shuffle (bool): Defaults to True.
        """
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
        """Dropout loader that randomly drops `proportion` samples
        of given labels from the original dataset.

        Will drop samples with labels indicated by `drops`

        Args:
            batch_size (int): Defaults to 128.
            shuffle (bool): Defaults to True.
            proportion (float): Proportion of samples to drop. Defaults to 0.9.
            drops (List): A list of ints, indicates the classes to drop.
                Defaults to [0, 1, 2, 3, 4].
        """
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
        """Randomly drops `proportion` samples of given labels
        from the original dataset.
        Also samples (1 - `proportion`) samples
        from classes that are not in `drops` to mitigate imbalance

        Will drop samples with labels indicated by `drops`

        Args:
            batch_size (int): Defaults to 128.
            shuffle (bool): Defaults to True.
            proportion (float): Proportion of samples to drop. Defaults to 0.9.
            drops (List): A list of ints, indicates the classes to drop.
                Defaults to [0, 1, 2, 3, 4].
        """
        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            proportion=proportion,
            drops=drops)
        self.dropped_train = [
            (x, y) for (x, y) in self.dropped_train if y in self.drops]

    def _sample_training_set(self):
        resampled = [
            (x, y) for (x, y) in self.train
            if y not in self.drops and random.random() >= self.proportion]

        train = resampled + self.dropped_train
        return train

    def __call__(self):
        resampled_train = ThinkSet(self._sample_training_set())
        return self._wrap_dataloaders(resampled_train), self.test_loader
