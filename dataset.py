import math

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from utils import cache_object

@cache_object(filename='dataset.pkl')
def load_dataset(opt, return_set=False, return_loader=True, noise=(False, False)):
    if not return_set and not return_loader:
        raise ValueError("One of return_set and return_loader should be true")

    common_transformers = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    noise_transformers = [
        transforms.GaussianBlur(math.ceil(4*opt.gaussion_std)//2*2+1, sigma=opt.gaussion_std)
    ]

    train_transformers = (noise_transformers if noise[0] else []) + common_transformers
    trainset = torchvision.datasets.CIFAR10(
        root=opt.data_dir, train=True, download=True,
        transform=transforms.Compose(train_transformers)
    )
    trainset, valset = train_test_split(trainset, test_size=1./50, random_state=2021, stratify=trainset.targets)
    test_transformers = (noise_transformers if noise[1] else []) + common_transformers
    testset = torchvision.datasets.CIFAR10(
        root=opt.data_dir, train=False, download=True,
        transform=transforms.Compose(test_transformers)
    )
    if return_set and not return_loader:
        return (trainset, valset, testset), (None, None, None)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=opt.batch_size, shuffle=False, num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=opt.batch_size, shuffle=False, num_workers=2
    )
    if not return_set and return_loader:
        return (None, None, None), (trainloader, valloader, testloader)

    return (trainset, valset, testset), (trainloader, valloader, testloader)

