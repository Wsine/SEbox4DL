import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from model import load_model
from dataset import load_dataset
from arguments import corparser as parser
from train import test
from correct import construct_model, ReplaceCorrect
from utils import *


class DecisionUnit(nn.Module):
    def __init__(self, module: ReplaceCorrect):
        super(DecisionUnit, self).__init__()
        self.indices = module.indices
        self.conv = module.conv
        self.cru = module.cru
        self.deconv = nn.ConvTranspose2d(
            module.conv.out_channels,
            module.conv.in_channels,
            kernel_size=module.conv.kernel_size,
            stride=module.conv.stride,
            padding=module.conv.padding,
            bias=module.conv.bias
        )
        self.dropout = nn.Dropout(p=0.25)
        self.revert = False
        self.boundary = 0

    def train(self, mode=True):
        self.conv.eval()
        self.cru.eval()
        self.deconv.train(mode)

    def forward(self, x):
        out = self.conv(x)
        inn = self.deconv(self.dropout(out))
        if self.revert is True:
            return inn

        global indicator
        indicator = (inn - x).abs().view(x.size(0), -1).sum(1) \
                    .gt(self.boundary).sum().div(x.size(0)).gt(0.5).item()

        if indicator is True:
            out[:, self.module.indices] = self.cru(x)

        return out


class AdvWrapper(nn.Module):
    def __init__(self, module):
        super(AdvWrapper, self).__init__()
        self.module = module

    def avg_diam_distance(self, x):
        v = x.view(-1).reshape(-1, 1)
        m = torch.triu(torch.matmul(v, v.t()), diagonal=1)
        d = m.sum().div(v.size(0) * (v.size(0) - 1))
        return d

    def forward(self, x):
        global indicator
        if isinstance(self.module, ReplaceCorrect):
            out = self.module.conv(x)
            repl = self.module.cru(x)

            if indicator is None:
                #  global diff
                num_repl = 0
                for i in range(x.size(0)):
                    o1 = self.avg_diam_distance(out[i, self.module.indices])
                    o2 = self.avg_diam_distance(repl[i])
                    #  diff.append((o2 - o1).item())
                    if o2 - o1 > 0.035:
                        num_repl += 1
                indicator = True if num_repl > x.size(0) / 2 else False

            if indicator is True:
                out[:, self.module.indices] = repl

        elif isinstance(self.module, nn.BatchNorm2d):
            if indicator is True:
                out = self.module(x)
            else:
                out = self.std_bn(x)
        else:
            out = self.module(x)

        return out


def square_error(x, y):
    e = x.flatten(1).sub(y.flatten(1)).square().sum(1)
    return e


def train_decision_unit(opt, model, device):
    model = construct_model(opt, model, patch=True)

    # Resume
    ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}_g{opt.gpu}'))
    model.load_state_dict(ckp['net'])

    # Prepare
    for m in model.modules():
        if isinstance(m, ReplaceCorrect):
            unit = DecisionUnit(m)
            unit.revert = True
            for param in unit.deconv.parameters():
                param.requires_grad = True
            break
    unit = unit.to(device)

    # Train
    trainset, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random', target_trsf=True)
    valset, valloader = load_dataset(opt, split='val', noise=True, noise_type='append', target_trsf=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_loss = 1e10
    for epoch in range(0, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))

        unit.train()
        train_loss = 0
        with tqdm(trainloader, desc='Train') as tepoch:
            for batch_idx, (imgs, labels) in enumerate(tepoch):
                noise_index = labels.eq(-1).nonzero().flatten()
                targets = imgs.clone()
                for i in range(imgs.size(1)):
                    targets[noise_index, i] = trainset.black_values[i]

                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                fimgs = unit(imgs)
                loss = criterion(fimgs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                avg_loss = train_loss / (batch_idx + 1)
                tepoch.set_postfix(loss=avg_loss)

        unit.eval()
        test_loss = 0
        with tqdm(valloader, desc='Evaluate') as tepoch:
            for batch_idx, (imgs, labels) in enumerate(tepoch):
                noise_index = labels.eq(-1).nonzero().flatten()
                targets = imgs.clone()
                for i in range(imgs.size(1)):
                    targets[noise_index, i] = valset.black_values[i]

                imgs, targets = imgs.to(device), targets.to(device)
                fimgs = unit(imgs)
                loss = criterion(fimgs, targets)
                test_loss += loss.item()
                avg_loss = test_loss / (batch_idx + 1)
                tepoch.set_postfix(loss=avg_loss)

        if avg_loss < best_loss:
            print('Record...')
            best_state = {
                'repoch': epoch,
                'net': unit.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'loss': avg_loss
            }
            best_loss = avg_loss
        scheduler.step()
    print('[info] the best revert loss is {:.4f}'.format(best_loss))

    # Calibrate
    unit.load_state_dict(best_state['net'])

    unit.eval()
    _, valloader = load_dataset(opt, split='val', noise=False)
    std_err = []
    for imgs, _ in valloader:
        imgs = imgs.to(device)
        fimgs = unit(imgs)
        err = square_error(fimgs, imgs)
        std_err.append(err)
    std_err = torch.cat(std_err)

    print('stat min:', std_err.min())
    print('stat median:', std_err.median())
    print('stat mean:', std_err.mean())
    print('stat max:', std_err.max())

    _, noiseloader = load_dataset(opt, split='val', noise=True, noise_type='expand')
    noise_err = []
    for imgs, _ in noiseloader:
        imgs = imgs.to(device)
        fimgs = unit(imgs)
        err = square_error(fimgs, imgs)
        noise_err.append(err)
    noise_err = torch.cat(noise_err)

    print('stat min:', noise_err.min())
    print('stat median:', noise_err.median())
    print('stat mean:', noise_err.mean())
    print('stat max:', noise_err.max())

    boundary = max(std_err.mean(), std_err.mean()).add(
        min(noise_err.mean(), noise_err.mean())
    ).mean()
    unit.boundary = boundary

    # Save
    print('Saving unit...')
    best_state['net'] = unit.state_dict()
    torch.save(best_state, get_model_path(opt, state=f'switch_{opt.fs_method}_g{opt.gpu}'))


def switch_on_the_fly(opt, model, device):
    backup = copy.deepcopy(model)
    # TODO: add here

    def _hook(module, pinput):
        global indicator
        indicator = None

    # Patch
    first_repl = True
    for name, module in model.named_modules():
        if isinstance(module, ReplaceCorrect):
            new_module = AdvWrapper(module)
            if first_repl is True:
                new_module.register_forward_pre_hook(_hook)
                first_repl = False
            rsetattr(model, name, new_module)
        elif isinstance(module, nn.BatchNorm2d):
            old_module = rgetattr(backup, name)
            new_module = AdvWrapper(module)
            new_module.std_bn = old_module
            rsetattr(model, name, new_module)

    model = model.to(device)

    # Evaluate
    criterion = torch.nn.CrossEntropyLoss()

    _, testloader = load_dataset(opt, split='test')
    acc, _ = test(model, testloader, criterion, device)
    print('[info] the normal accuracy is {:.4f}%'.format(acc))
    _, testloader = load_dataset(opt, split='test', noise=True, noise_type='append')
    acc, _ = test(model, testloader, criterion, device)
    print('[info] the robustness accuracy is {:.4f}%'.format(acc))


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = load_model(opt, pretrained=True)
    #  backbone = copy.deepcopy(model)


    train_decision_unit(opt, model, device)
    #  switch_on_the_fly(opt, model, device)


if __name__ == '__main__':
    #  global diff
    #  diff = []
    main()

    #  import statistics
    #  print('min:', min(diff))
    #  print('median:', statistics.median(diff))
    #  print('mean:', statistics.mean(diff))
    #  print('max:', max(diff))
    #  print('var:', statistics.variance(diff))

