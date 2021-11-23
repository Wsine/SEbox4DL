import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.runners.train import train, test
from src.utils import *


repair_dispatcher = AttrDispatcher('repair_runner')


class CorrectionUnit(nn.Module):
    def __init__(self, num_filters, Di, k):
        super(CorrectionUnit, self).__init__()
        self.conv1 = nn.Conv2d(
            num_filters, Di, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(Di)
        self.conv2 = nn.Conv2d(
            Di, Di, kernel_size=k, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(Di)
        self.conv3 = nn.Conv2d(
            Di, Di, kernel_size=k, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(Di)
        self.conv4 = nn.Conv2d(
            Di, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        out += self.shortcut(x)
        return out


class ConcatCorrect(nn.Module):
    def __init__(self, conv_layer, indices):
        super(ConcatCorrect, self).__init__()
        self.indices = indices
        self.others = [i for i in range(conv_layer.out_channels)
                       if i not in indices]
        self.conv = conv_layer
        num_filters = len(indices)
        self.cru = CorrectionUnit(num_filters, num_filters, 3)

    def forward(self, x):
        out = self.conv(x)
        out_lower = out[:, self.others]
        out_upper = self.cru(out[:, self.indices])
        out = torch.cat([out_lower, out_upper], dim=1)
        #  out[:, self.indices] = self.cru(out[:, self.indices])
        return out


class ReplaceCorrect(nn.Module):
    def __init__(self, conv_layer, indices):
        super(ReplaceCorrect, self).__init__()
        self.indices = indices
        self.conv = conv_layer
        self.cru = nn.Conv2d(
            conv_layer.in_channels,
            len(indices),
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            groups=conv_layer.groups,
            bias=False)

    def forward(self, x):
        out = self.conv(x)
        out[:, self.indices] = self.cru(x)
        return out


@repair_dispatcher.register('deepcorrect')
def deepcorrect_method(_, model, key_dict=None):
    new_model = model
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    new_key_dict = {}
    for layer_name in conv_names:
        module = rgetattr(model, layer_name)
        if module.groups != 1:
            continue

        num_susp = int(module.out_channels * 0.75)
        if key_dict is None:
            indices = random.sample(range(module.out_channels), num_susp)
        else:
            indices = key_dict[layer_name]
        new_key_dict[layer_name] = indices

        correct_module = ConcatCorrect(module, indices)
        rsetattr(new_model, layer_name, correct_module)
    return new_model, new_key_dict


@repair_dispatcher.register('deeppatch')
def deeppatch_method(_, model, key_dict=None):
    new_model = model
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    new_key_dict = {}
    for layer_name in conv_names:
        module = rgetattr(model, layer_name)
        if module.groups != 1:
            continue

        num_susp = int(module.out_channels * 0.25)
        if key_dict is None:
            indices = random.sample(range(module.out_channels), num_susp)
        else:
            indices = key_dict[layer_name]
        new_key_dict[layer_name] = indices

        correct_module = ReplaceCorrect(module, indices)
        rsetattr(new_model, layer_name, correct_module)
    return new_model, new_key_dict


def repair_model(ctx, model, trainloader, valloader):
    new_model, key_dict = repair_dispatcher(ctx.opt, model)
    new_model = new_model.to(ctx.device)

    for name, module in new_model.named_modules():
        if 'cru' in name:
            for param in module.parameters():
                param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = False

    criterion = eval(f'torch.nn.{ctx.opt.criterion}()')
    optimizer = eval(f'torch.optim.{ctx.opt.optimizer}')(
        filter(lambda p: p.requires_grad, new_model.parameters()),
        lr=ctx.opt.lr, momentum=ctx.opt.momentum, weight_decay=ctx.opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc, *_ = test(ctx, new_model, valloader, criterion, ctx.device)
    for epoch in ctx.tqdm(range(0, ctx.opt.max_epoch), desc='Epochs'):
        train(ctx, new_model, trainloader, optimizer, criterion, ctx.device)
        acc, loss = test(ctx, new_model, valloader, criterion, ctx.device)
        if acc > best_acc:
            state = {
                'epoch': epoch,
                'net': new_model.state_dict(),
                'key_dict': key_dict,
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc,
                'loss': loss
            }
            torch.save(state, get_model_path(ctx, state='repair'))
            best_acc = acc
        scheduler.step()

    best_model_path = get_model_path(ctx, state='repair')
    ckpt = torch.load(best_model_path, map_location=torch.device('cpu'))
    new_model.load_state_dict(ckpt['net'])
    repair_model = new_model.to(ctx.device)

    return repair_model, best_model_path

