from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torch import Generator
from modules.dataset import BAF, UNSW_NB15

def load_dataset(dataset: str, batch_size: int, root='./data', seed=None):
    """
    Load the dataset and return the train and test loaders.\n
    ---
    DATASETS
    ---

    BAF - Bank Account Fraud:
    - Variant: Base, TypeI, TypeII, TypeIII, TypeIV, TypeV
    - Dataset size: 1,000,000
    - Features: 31 or 33
    - Classes: 2

    MNIST:
    - Dataset size: 60,000 / 10,000
    - Image size: 28 x 28
    - Classes: 10
    
    Fashion MNIST:
    - Dataset size: 60,000 / 10,000
    - Image size: 28 x 28
    - Classes: 10

    CIFAR10:
    - Dataset size: 50,000 / 10,000
    - Image size: 32 x 32
    - Classes: 10

    CIFAR100:
    - Dataset size: 50,000 / 10,000
    - Image size: 32 x 32
    - Classes: 100

    UNSW-NB15:
    - Dataset size: 175,341
    - Features: 49
    - Classes: 10

    IRIS:
    - Dataset size: 150
    - Features: 4
    - Classes: 3

    ---
    INPUTS/OUTPUTS
    ---

    Args:
        dataset (str): Name of the dataset (baf-{variant}, mnist, fashion_mnist, cifar10, cifar100).
        batch_size (int): Batch size.
        root (str): Path to the dataset.
        seed (int): Seed for the random

    Returns:
        DataLoader: Train data loader.
        DataLoader: Test data loader.
    """
    transform_mnist = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if dataset == 'baf':
        train = BAF(variant='Base', root=f"{root}/BAF", train=True)
        test = BAF(variant='Base', root=f"{root}/BAF", train=False)
    elif 'baf' in dataset.lower():
        train = BAF(variant=dataset.split('-')[-1], root=f"{root}/BAF", train=True)
        test = BAF(variant=dataset.split('-')[-1], root=f"{root}/BAF", train=False)
    elif dataset == 'mnist':
        train = MNIST(root=root, train=True, download=True, transform=transform_mnist)
        test = MNIST(root=root, train=False, download=True, transform=transform_mnist)
    elif dataset == 'fashion_mnist':
        train = FashionMNIST(root=root, train=True, download=True, transform=transform_mnist)
        test = FashionMNIST(root=root, train=False, download=True, transform=transform_mnist)
    elif dataset == 'cifar10':
        train = CIFAR10(root=root, train=True, download=True, transform=transform_cifar)
        test = CIFAR10(root=root, train=False, download=True, transform=transform_cifar)
    elif dataset == 'cifar100':
        train = CIFAR100(root=root, train=True, download=True, transform=transform_cifar)
        test = CIFAR100(root=root, train=False, download=True, transform=transform_cifar)
    elif dataset == 'unsw_nb15':
        train = UNSW_NB15(root=f"{root}/UNSW-NB15", train=True, multiclass=True)
        test = UNSW_NB15(root=f"{root}/UNSW-NB15", train=False, multiclass=True)
    else:
        raise ValueError("Invalid dataset")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, generator=Generator().manual_seed(seed))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0, generator=Generator().manual_seed(seed))
    return train_loader, test_loader