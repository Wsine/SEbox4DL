import copy

import torch
import torch.nn as nn

from model import load_model
from dataset import load_dataset
from arguments import corparser as parser
from train import test
from correct import construct_model, ReplaceCorrect
from utils import *


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


def auto_encoder(opt, model, device):
    pass


def switch_on_the_fly(opt, model, device):
    backup = copy.deepcopy(model)
    model = construct_model(opt, model, patch=True)

    # Resume
    ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}_g{opt.gpu}'))
    model.load_state_dict(ckp['net'])

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


    #  auto_encoder(opt, model, device)
    switch_on_the_fly(opt, model, device)


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

