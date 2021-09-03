import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import load_model
from dataset import load_dataset
from arguments import Args, advparser as parser
from train import train, test
from correct import construct_model, ReplaceCorrect
from utils import *


class DecisionUnit(nn.Module):
    def __init__(self, module: ReplaceCorrect, img_size):
        super(DecisionUnit, self).__init__()
        self.indices = module.indices
        self.conv = module.conv
        self.cru = module.cru
        self.clf = nn.Linear(
            self.cru.out_channels*img_size*img_size,
            2
        )
        self.shortpre = False
        self.boundary = 0

    def train(self, mode=True):
        self.conv.eval()
        self.cru.eval()
        self.clf.train(mode)

    def corrective_matrix(self, x, y):
        return torch.logical_xor(F.relu(x).gt(0), F.relu(y).gt(0)).float()

    def forward(self, x):
        out = self.conv(x)
        repl = self.cru(x)
        diff = self.corrective_matrix(out[:, self.indices], repl)
        pred = self.clf(diff.view(diff.size(0), -1))
        if self.shortpre is True:
            return pred

        global indicator
        indicator = (1 - pred.softmax(1).square().sum(1)) \
                    .gt(self.boundary).sum().div(x.size(0)).gt(0.5).item()

        if indicator is True:
            out[:, self.indices] = self.cru(x)

        return out


class AdvWrapper(nn.Module):
    def __init__(self, module):
        super(AdvWrapper, self).__init__()
        self.module = module
        self.boundary = 0

    def avg_diam_distance(self, x):
        v = x.view(-1).reshape(-1, 1)
        m = torch.triu(torch.matmul(v, v.t()), diagonal=1)
        d = m.sum().div(v.size(0) * (v.size(0) - 1))
        return d

    def neuron_coverage(self, x):
        nc = F.relu(x).gt(1e-6).sum().div(x.numel())
        return nc

    def diffenentiate_activation(self, x, y):
        batch, numel = x.view(x.size(0), -1).size()
        #  d = torch.logical_and(F.relu(x).gt(0), F.relu(y).gt(0))
        #  d = torch.logical_or(F.relu(x).gt(0), F.relu(y).gt(0))
        d = torch.logical_xor(F.relu(x).gt(0), F.relu(y).gt(0))
        nc = d.view(batch, numel).sum(1).div(numel)
        return nc

    def forward(self, x):
        global indicator
        if isinstance(self.module, ReplaceCorrect):
            out = self.module.conv(x)
            repl = self.module.cru(x)

            if indicator is None:
                out = (out, repl, self.module.indices)
            elif indicator is True:
                out[:, self.module.indices] = repl

        elif isinstance(self.module, nn.BatchNorm2d):
            if indicator is None:
                p_out, p_repl, p_indices = x
                out1 = self.std_bn(p_out)
                p_out[:, p_indices] = p_repl
                out2 = self.module(p_out)
                self.dnc = self.diffenentiate_activation(out1[:, p_indices], out2[:, p_indices])
                if self.dnc.lt(self.boundary).sum().div(p_out.size(0)).gt(0.5):
                    indicator = True
                    out = out2
                else:
                    indicator = False
                    out = out1
            elif indicator is True:
                out = self.module(x)
            else:
                out = self.std_bn(x)
        else:
            out = self.module(x)

        return out


def train_decision_unit(opt, model, device):
    model = construct_model(opt, model, patch=True)

    # Resume
    ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}_g{opt.gpu}'))
    model.load_state_dict(ckp['net'])

    # Prepare
    img_size = Args.get_img_size(opt.dataset)
    for m in model.modules():
        if isinstance(m, ReplaceCorrect):
            unit = DecisionUnit(m, img_size)
            break
    for param in unit.clf.parameters():
        param.requires_grad = True
    for param in unit.conv.parameters():
        param.requires_grad = False
    for param in unit.cru.parameters():
        param.requires_grad = False
    unit.shortpre = True
    unit = unit.to(device)

    # Train
    _, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random', target_trsf=True)
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append', target_trsf=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, unit.parameters()),
        lr=opt.lr*0.01, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0
    train_acc, train_loss = [], []
    for epoch in range(0, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        tacc, tloss = train(unit, trainloader, optimizer, criterion, device)
        train_acc.append(tacc)
        train_loss.append(tloss)
        acc, _ = test(unit, valloader, criterion, device)
        if acc > best_acc:
            print('Record...')
            best_state = {
                'repoch': epoch,
                'net': unit.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            best_acc = acc
        scheduler.step()
    print('[info] the best short prediction accuracy is {:.4f}'.format(best_acc))
    print('train_acc:', train_acc)
    print('train_loss:', train_loss)

    # Evaluate
    unit.load_state_dict(best_state['net'])
    _, testloader = load_dataset(opt, split='test', noise=True, noise_type='append', target_trsf=True)
    acc, _ = test(unit, testloader, criterion, device)
    print('[info] decision accuracy is {:.4f}%'.format(acc))


def switch_on_the_fly(opt, model, device):
    backbone = copy.deepcopy(model)
    model = construct_model(opt, model, patch=True)

    # Resume
    ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}_g{opt.gpu}'))
    model.load_state_dict(ckp['net'])

    def _clean_indicator_hook(module, pinput):
        global indicator
        indicator = None

    # Patch
    first_repl = True
    for name, module in model.named_modules():
        if isinstance(module, ReplaceCorrect):
            new_module = AdvWrapper(module)
            if first_repl is True:
                new_module.register_forward_pre_hook(_clean_indicator_hook)
                first_repl = False
            rsetattr(model, name, new_module)
        elif isinstance(module, nn.BatchNorm2d):
            old_module = rgetattr(backbone, name)
            new_module = AdvWrapper(module)
            new_module.std_bn = old_module
            rsetattr(model, name, new_module)

    model = model.to(device)

    # Calibrate
    diff_list = []
    def _calib_hook(module, finput, foutput):
        diff_list.append(module.dnc)

    first_bn_name = None
    for n, m in model.named_modules():
        if isinstance(m, AdvWrapper) and isinstance(m.module, nn.BatchNorm2d):
            first_bn_name = n
            handle = m.register_forward_hook(_calib_hook)
            break

    criterion = torch.nn.CrossEntropyLoss()
    _, valloader = load_dataset(opt, split='val', noise=False)
    test(model, valloader, criterion, device, desc='Calibrate')
    std_mean = torch.cat(diff_list).mean()

    diff_list.clear()
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='expand')
    test(model, valloader, criterion, device, desc='Calibrate')
    noise_mean = torch.cat(diff_list).mean()

    boundary = (std_mean + noise_mean) / 2
    rgetattr(model, first_bn_name).boundary = boundary


    # Evaluate decision accuracy
    diff_list.clear()
    _, testloader = load_dataset(opt, split='test', noise=False)
    test(model, testloader, criterion, device)
    total = torch.cat(diff_list).size(0)
    correct = torch.cat(diff_list).gt(boundary).sum().item()

    diff_list.clear()
    _, testloader = load_dataset(opt, split='test', noise=True, noise_type='expand')
    test(model, testloader, criterion, device)
    total += torch.cat(diff_list).size(0)
    correct += torch.cat(diff_list).lt(boundary).sum().item()

    print('[info] decision accuracy is {:.4f}%'.format(100. * correct / total))
    handle.remove()

    # Evaluate
    _, testloader = load_dataset(opt, split='test', noise=False)
    std_acc, _ = test(model, testloader, criterion, device)
    print('[info] the normal accuracy is {:.4f}%'.format(std_acc))

    partial_noisy_acc = []
    for std in [0.5, 1., 1.5, 2., 2.5, 3.]:
        _, testloader = load_dataset(opt, split='test', noise=True, noise_type='replace', gblur_std=std)
        acc, _ = test(model, testloader, criterion, device)
        print('[info] the robustness accuracy for std {:.1f} is {:.4f}%'.format(std, acc))
        partial_noisy_acc.append(acc)

    _, testloader = load_dataset(opt, split='test', noise=True, noise_type='append')
    noisy_acc, _ = test(model, testloader, criterion, device)
    print('[info] the robustness accuracy is {:.4f}%'.format(noisy_acc))

    #  print('Saving model...')
    #  state = {
    #      'net': model.state_dict(),
    #      'std_acc': std_acc,
    #      'noisy_acc': noisy_acc,
    #      'partial_noisy_acc': partial_noisy_acc
    #  }
    #  torch.save(state, get_model_path(opt, state=f'switch_{opt.fs_method}_g{opt.gpu}'))


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = load_model(opt, pretrained=True)

    train_decision_unit(opt, model, device)
    #  switch_on_the_fly(opt, model, device)


if __name__ == '__main__':
    main()

