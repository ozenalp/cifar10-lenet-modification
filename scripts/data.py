import os
import copy
import torch
import torchvision
from torchvision import transforms

def get_transforms():
    # Define the image transformations for the training set
    # ToTensor() will convert the PIL image to a PyTorch tensor
    # Normalize() will standardize the pixel values (mean, std for each of the 3 channels (R, G, B))
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Define the image transformations for the testing set
    # ToTensor() will convert the PIL image to a PyTorch tensor
    # Normalize() will standardize the pixel values (mean, std for each of the 3 channels (R, G, B))
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    # Return both the train and test transformations
    return train_transform, test_transform


def load_cifar10_datasets(root='../data', train_transform=None, test_transform=None):
    # Check if the root directory exists, if not, create it
    if not os.path.exists(root):
        os.makedirs(root)

    # Load the CIFAR10 training data from torchvision datasets
    # If the data is not available at the root location, download it
    # Apply the specified training transformations to the data
    train_data = torchvision.datasets.CIFAR10(root=root,
                                              train=True,
                                              download=True,
                                              transform=train_transform)

    # Load the CIFAR10 testing data from torchvision datasets
    # If the data is not available at the root location, download it
    # Apply the specified testing transformations to the data
    test_data = torchvision.datasets.CIFAR10(root=root,
                                             train=False,
                                             download=True,
                                             transform=test_transform)
    
    # Return the train and test data
    return train_data, test_data


def split_train_validation(train_data, valid_ratio=0.9, test_transform=None):
    # Calculate the number of training examples as per the validation ratio
    n_train_examples = int(len(train_data) * valid_ratio)
    # Remaining examples are used for validation
    n_valid_examples = len(train_data) - n_train_examples

    # Randomly split the train_data into training and validation sets
    train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples])

    # Deepcopy valid_data so as not to alter the original data when applying transformations
    valid_data = copy.deepcopy(valid_data)
    # Apply transformations specific to validation data, typically these are the same as test set transformations
    valid_data.dataset.transform = test_transform

    return train_data, valid_data



def create_data_loaders(train_data, valid_data, test_data, batch_size=64):
    # Create a data loader for the training data
    # DataLoader allows us to efficiently load data in batches
    # Shuffle=True ensures that the data is shuffled at every epoch
    train_iterator = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Create data loaders for validation and test data
    # For these, shuffling of data is typically not necessary
    valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    test_iterator = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Return the data loaders
    return train_iterator, valid_iterator, test_iterator



classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')