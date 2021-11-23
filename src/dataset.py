import math
import random
from datetime import date

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split


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
    def __init__(self, persistent=None):
        self.persistent = persistent

    def set_persistent(self, p):
        self.persistent = p

    def __call__(self, apply):
        if self.persistent is not None:
            return self.persistent
        return 1 if apply is True else 0


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
            y = self.target_transform(apply)
        return x, y

    def __len__(self):
        return len(self.dataset)


def compute_mean_std(ctx, entry):
    @ctx.cache
    def _compute_wrapper():
        dataset = entry(root=ctx.data_dir, train=True, download=True, transform=T.ToTensor())
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=ctx.opt.batch_size, shuffle=False, num_workers=2
        )
        mean, std = 0., 0.
        nb_samples = 0.
        for data, _ in loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

        return mean, std
    return _compute_wrapper()


def load_dataset(ctx, split,
        noise=False, noise_type=None,
        gblur_std=None, target_trsf=False):

    dataset_entry = eval(f'torchvision.datasets.{ctx.opt.dataset}')
    mean, std = compute_mean_std(ctx, dataset_entry)
    random_state = date.today().year

    common_transformers = [
        T.ToTensor(),
        T.Normalize(mean, std)
    ]
    target_transform = None if target_trsf is False else MaskNoiseLabel()

    if split == 'test':
        base_dataset = dataset_entry(root=ctx.data_dir, train=False, download=True)
    elif split == 'val':
        base_largeset = dataset_entry(root=ctx.data_dir, train=True, download=True)
        _, base_dataset = train_test_split(
            base_largeset, test_size=1./50, random_state=random_state, stratify=base_largeset.targets)
    elif split == 'train':
        base_largeset = dataset_entry(root=ctx.data_dir, train=True, download=True)
        base_dataset, _ = train_test_split(
            base_largeset, test_size=1./50, random_state=random_state, stratify=base_largeset.targets)
    else:
        raise ValueError('Invalid parameter of split')

    if noise is True:
        if noise_type == 'random':
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
            if target_transform is not None:
                target_transform.set_persistent(1)
            dataset = PostTransformDataset(
                base_dataset,
                transform=T.Compose([trsf] + common_transformers),
                target_transform=target_transform
            )
        elif noise_type == 'expand' or noise_type == 'append':
            if target_transform is not None:
                target_transform.set_persistent(0)
            incset = [PostTransformDataset(
                base_dataset,
                transform=T.Compose(common_transformers),
                target_transform=target_transform
            )] if noise_type == 'append' else []
            if target_transform is not None:
                target_transform.set_persistent(1)
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
    else:
        dataset = PostTransformDataset(
            base_dataset,
            transform=T.Compose(common_transformers)
        )

    shuffle = True if split == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=ctx.opt.batch_size, shuffle=shuffle, num_workers=2
    )

    return dataset, dataloader

