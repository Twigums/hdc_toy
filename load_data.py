import torch

from torchvision import datasets
from torchvision.transforms import ToTensor

def load_MNIST(batch_size, device):
    print("Loading MNIST data...")

    kwargs = {"num_workers": 1,
              "pin_memory": True} if device == "cuda" else {}

    transforms = ToTensor()

    train_dataset = datasets.MNIST(root = "MNIST-data",
                                   train = True,
                                   download = True,
                                   transform = transforms)

    test_dataset = datasets.MNIST(root = "MNIST-data",
                                   train = False,
                                   transform = transforms)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = batch_size,
                                              shuffle = False,
                                              **kwargs)

    print("Loading complete.")

    return train_dataset, test_dataset, train_loader, test_loader
