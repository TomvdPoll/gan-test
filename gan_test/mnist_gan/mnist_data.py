from typing import Tuple
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda, Normalize


def get_mnist() -> Tuple[Dataset, Dataset]:
    transform = Compose([
        ToTensor(),
        Normalize(mean=0.5, std=0.5),
        Lambda(lambda image: image.view(784))
    ])
    data_train = MNIST(root="dataset/", download=True, train=True, transform=transform)
    data_test = MNIST(root="dataset/", download=True, train=False, transform=transform)

    return data_train, data_test


if __name__ == "__main__":
    train, test = get_mnist()
    a=2
