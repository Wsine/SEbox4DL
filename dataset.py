import math
import random

import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from utils import cache_object


class RandomApply(object):
    def __init__(self, tran, p=0.5):
        self.tran = tran
        self.prob = p
        self.apply = False

    def __call__(self, x):
        p = random.random()
        if p < self.prob:
            self.apply = True
            x = self.tran(x)
        else:
            self.apply = False
        return x


class RandomApplyOne(object):
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        t = random.choice(self.trans)
        return t(x)


class MaskNoiseLabel(object):
    def __init__(self, label=-1):
        self.noise_label = label

    def __call__(self, y, apply):
        return self.noise_label if apply is True else y


class PostTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        apply = False
        if self.transform is not None:
            x = self.transform(x)
            if isinstance(self.transform, T.Compose):
                for t in self.transform.transforms:
                    if isinstance(t, RandomApply):
                        apply = t.apply
        if self.target_transform is not None:
            y = self.target_transform(y, apply)
        return x, y

    def __len__(self):
        return len(self.dataset)


@cache_object(filename='dataset.pkl')
def load_dataset(
        opt, split,
        noise=False, noise_type=None,
        gblur_std=None, target_trsf=False):

    if opt.dataset == 'cifar10':
        cifar = torchvision.datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        cifar = torchvision.datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('Invalid dataset value')

    common_transformers = [
        T.ToTensor(),
        T.Normalize(mean, std)
    ]
    target_transform = None if target_trsf is False else MaskNoiseLabel()

    if split == 'test':
        base_dataset = cifar(root=opt.data_dir, train=False, download=True)
    elif split == 'val':
        base_largeset = cifar(root=opt.data_dir, train=True, download=True)
        _, base_dataset = train_test_split(
            base_largeset, test_size=1./50, random_state=2021, stratify=base_largeset.targets)
    elif split == 'train':
        base_largeset = cifar(root=opt.data_dir, train=True, download=True)
        base_dataset, _ = train_test_split(
            base_largeset, test_size=1./50, random_state=2021, stratify=base_largeset.targets)
    else:
        raise ValueError('Invalid parameter of split')

    if noise is True:
        if noise_type == 'random':
            #  trsf = T.RandomApply([
            #      RandomApplyOne([
            #          T.GaussianBlur(math.ceil(4*std)//2*2+1, sigma=std)
            #          for std in [0.5, 1., 1.5, 2., 2.5, 3.]
            #      ])
            #  ], p=0.5)
            trsf = RandomApply(RandomApplyOne([
                T.GaussianBlur(math.ceil(4*std)//2*2+1, sigma=std)
                for std in [0.5, 1., 1.5, 2., 2.5, 3.]
            ]), p=0.5)
            dataset = PostTransformDataset(
                base_dataset,
                transform=T.Compose([trsf] + common_transformers),
                target_transform=target_transform
            )
        elif noise_type == 'replace':
            assert gblur_std is not None, 'gblur_std should be a floating number'
            trsf = T.GaussianBlur(math.ceil(4*gblur_std)//2*2+1, sigma=gblur_std)
            dataset = PostTransformDataset(
                base_dataset,
                transform=T.Compose([trsf] + common_transformers),
                target_transform=target_transform
            )
        elif noise_type == 'expand' or noise_type == 'append':
            incset = [PostTransformDataset(
                base_dataset,
                transform=T.Compose(common_transformers),
                target_transform=target_transform
            )] if noise_type == 'append' else []
            for std in [0.5, 1., 1.5, 2., 2.5, 3.]:
                trsf = T.GaussianBlur(math.ceil(4*std)//2*2+1, sigma=std)
                incset.append(PostTransformDataset(
                    base_dataset,
                    transform=T.Compose([trsf] + common_transformers),
                    target_transform=target_transform
                ))
            dataset = torch.utils.data.ConcatDataset(incset)
        else:
            raise ValueError('Invalid noise_type parameter')

        if target_trsf is True:
            black_img = torch.zeros(dataset[0][0].size())
            dataset.black_values = black_img.flatten(1).mean(1)
    else:
        dataset = PostTransformDataset(
            base_dataset,
            transform=T.Compose(common_transformers)
        )

    shuffle = True if split == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=2
    )

    return dataset, dataloader

