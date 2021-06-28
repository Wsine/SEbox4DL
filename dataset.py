import math
import random

import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from utils import cache_object


class RandomApplyOne(object):
    """Randomly apply one of the tranformers to the input"""

    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        t = random.choice(self.trans)
        return t(x)


class PostTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


@cache_object(filename='dataset.pkl')
def load_dataset(
        opt,
        return_set=False, return_loader=True,
        noise=(False, False), prob=0.5):
    if not return_set and not return_loader:
        raise ValueError("One of return_set and return_loader should be true")

    if opt.dataset == "cifar10":
        cifar = torchvision.datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == "cifar100":
        cifar = torchvision.datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError("Invalid dataset value")

    common_transformers = [
        T.ToTensor(),
        T.Normalize(mean, std)
    ]

    train_noise, test_noise = noise
    train_transformers = []
    if train_noise:
        train_transformers.append(
            T.RandomApply(
                [RandomApplyOne([
                    T.GaussianBlur(math.ceil(4*std)//2*2+1, sigma=std)
                    for std in [0.5, 1., 1.5, 2., 2.5, 3.]
                ])],
                p=prob
            )
        )
    base_trainset = cifar(root=opt.data_dir, train=True, download=True)
    pretrainset, prevalset = train_test_split(
        base_trainset, test_size=1./50, random_state=2021, stratify=base_trainset.targets)
    trainset = PostTransformDataset(
        pretrainset,
        transform=T.Compose(train_transformers + common_transformers)
    )
    valset = PostTransformDataset(
        prevalset,
        transform=T.Compose(train_transformers + common_transformers)
    )

    base_testset = cifar(
        root=opt.data_dir, train=False, download=True,
        transform=T.Compose(common_transformers)
    )
    increase_testset = []
    if test_noise:
        test_noise_transformers = [
            T.GaussianBlur(math.ceil(4*std)//2*2+1, sigma=std)
            for std in [0.5, 1., 1.5, 2., 2.5, 3.]
        ]
        for t in test_noise_transformers:
            incset = cifar(
                root=opt.data_dir, train=False, download=False,
                transform=T.Compose([t] + common_transformers)
            )
            increase_testset.append(incset)
    testset = torch.utils.data.ConcatDataset([base_testset] + increase_testset)

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

