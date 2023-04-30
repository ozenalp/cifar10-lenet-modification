import os
import copy
import torch
import torchvision
from torchvision import transforms

def get_transforms():
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    return train_transform, test_transform

def load_cifar10_datasets(root='../data', train_transform=None, test_transform=None):
    if not os.path.exists(root):
        os.makedirs(root)

    train_data = torchvision.datasets.CIFAR10(root=root,
                                              train=True,
                                              download=True,
                                              transform=train_transform)

    test_data = torchvision.datasets.CIFAR10(root=root,
                                             train=False,
                                             download=True,
                                             transform=test_transform)
    
    return train_data, test_data

def split_train_validation(train_data, valid_ratio=0.9, test_transform=None):
    n_train_examples = int(len(train_data) * valid_ratio)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples])

    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transform

    return train_data, valid_data


def create_data_loaders(train_data, valid_data, test_data, batch_size=64):
    train_iterator = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    test_iterator = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_iterator, valid_iterator, test_iterator
