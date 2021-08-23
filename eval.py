import sys

import torch

from dataset import load_dataset
from model import load_model
from arguments import commparser, corparser
from train import test
from correct import construct_model
from utils import *


def main():
    if '-c' in sys.argv or '--crt_method' in sys.argv:
        opt = corparser.parse_args()
    else:
        opt = commparser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if hasattr(opt, 'fs_method') and opt.fs_method is not None:
        model = load_model(opt, pretrained=False)
        model = construct_model(opt, model).to(device)
        ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}_g{opt.gpu}'))
        model.load_state_dict(ckp['net'])
    elif hasattr(opt, 'crt_method') and opt.crt_method == 'finetune':
        model = load_model(opt, pretrained=False).to(device)
        ckp = torch.load(get_model_path(opt, state=f'finetune_g{opt.gpu}'))
        model.load_state_dict(ckp['net'])
    else:
        model = load_model(opt, pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    _, testloader = load_dataset(opt, split='test', noise=False)
    acc, _ = test(model, testloader, criterion, device)
    print('[info] the base accuracy is {:.4f}%'.format(acc))

    for std in [0.5, 1., 1.5, 2., 2.5, 3.]:
        _, noiseloader = load_dataset(opt, split='test', noise=True, noise_type='replace', gblur_std=std)
        acc, _ = test(model, noiseloader, criterion, device)
        print('[info] the robustness accuracy for std {:.1f} is {:.4f}%'.format(std, acc))

    _, noiseloader = load_dataset(opt, split='test', noise=True, noise_type='append')
    acc, _ = test(model, noiseloader, criterion, device)
    print('[info] the robustness accuracy is {:.4f}%'.format(acc))


if __name__ == '__main__':
    main()

