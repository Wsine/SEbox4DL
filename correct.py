import os
import json
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from model import load_model
from dataset import load_dataset
from arguments import corparser as parser
from train import train, test
from utils import *


dispatcher = AttrDispatcher('crt_method')


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


class NoneCorrect(nn.Module):
    def __init__(self, conv_layer, indices):
        super(NoneCorrect, self).__init__()
        self.indices = indices
        self.conv = conv_layer

    def forward(self, x):
        out = self.conv(x)
        out[:, self.indices] = 0
        return out


def construct_model(opt, model, patch=True):
    sus_filters = json.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, f'susp_filters_{opt.fs_method}.json'
    )))
    for name, info in sus_filters.items():
        layer_name = name.rstrip('.weight')
        module = rgetattr(model, layer_name)

        ranking = info['indices']
        num_susp = int(len(ranking) * opt.susp_ratio)
        indices = ranking[:num_susp] if opt.susp_side == 'front' else ranking[-num_susp:]
        #  while len(indices) % module.groups != 0:
        #      num_susp += 1
        #      indices = ranking[:num_susp]

        if patch is False:
            correct_module = NoneCorrect(module, indices)
        elif opt.crt_type == 'crtunit':
            correct_module = ConcatCorrect(module, indices)
        elif opt.crt_type == 'replace':
            correct_module = ReplaceCorrect(module, indices)
        else:
            raise ValueError('Invalid correct type')
        rsetattr(model, layer_name, correct_module)
    return model


@torch.no_grad()
def coordinate_filters(_, model):
    correct_rank = []
    for name, module in model.named_modules():
        if 'cru' in name:
            conv_name = name.rstrip('cru') + 'conv'
            conv_module = rgetattr(model, conv_name)
            parent_name = name.rstrip('.cru')
            parent_module = rgetattr(model, parent_name)

            chn_out, chn_in, kh, kw = conv_module.weight.size()
            indices = parent_module.indices
            non_indices = [i for i in range(chn_out) if i not in indices]

            #  precision = 1000
            conv_weight = conv_module.weight[non_indices].view(len(non_indices), -1)
            #  conv_weight = (conv_weight * precision).int().float()
            conv_rank = torch.matrix_rank(conv_weight)
            cru_weight = module.weight.view(module.weight.size(0), -1)

            unrank_idx = []
            unrank_mean = []
            for i, w in enumerate(cru_weight):
                w = w.view(-1).view(1, -1)
                #  w = (w * precision).int().float()
                cat_weight = torch.cat([conv_weight, w])
                new_rank = torch.matrix_rank(cat_weight)
                if new_rank <= conv_rank:
                    unrank_idx.append(i)
                    unrank_mean.append(cat_weight.mean(dim=0).view(chn_in, kh, kw))

            if unrank_idx:
                #  torch.nn.init.xavier_uniform_(module.weight[unrank_idx])
                for i, m in zip(unrank_idx, unrank_mean):
                    module.weight[i] = m

            correct_rank.append(len(unrank_idx))
    print('Correct Rank: [{}]'.format(' '.join(map(str, correct_rank))))


@dispatcher.register('patch')
def patch(opt, model, device):
    _, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random')
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')

    model = construct_model(opt, model)
    model = model.to(device)

    for name, module in model.named_modules():
        if 'cru' in name:
            for param in module.parameters():
                param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start_epoch = -1
    if opt.resume:
        ckp = torch.load(get_model_path(opt, state=f'patch_{opt.fs_method}_g{opt.gpu}'))
        model.load_state_dict(ckp['net'])
        optimizer.load_state_dict(ckp['optim'])
        scheduler.load_state_dict(ckp['sched'])
        start_epoch = ckp['cepoch']
        best_acc = ckp['acc']
    else:
        best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')

    for epoch in range(start_epoch + 1, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state=f'patch_{opt.fs_method}_g{opt.gpu}'))
            best_acc = acc
        scheduler.step()
        #  if opt.crt_type == 'replace':
        #      coordinate_filters(opt, model)
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


@dispatcher.register('finetune')
def finetune(opt, model, device):
    _, trainloader = load_dataset(opt, split='train', noise=True, noise_type='random')
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='append')

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc, *_ = test(model, valloader, criterion, device, desc='Baseline')
    for epoch in range(0, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = test(model, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'cepoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state=f'finetune_g{opt.gpu}'))
            best_acc = acc
        scheduler.step()
    print('[info] the best retrain accuracy is {:.4f}%'.format(best_acc))


@dispatcher.register('calibrate')
@torch.no_grad()
def calibrate(opt, model, device):
    _, (_, _, testloader) = load_dataset(opt)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    base_acc, (preds, trgs) = test(model, testloader, criterion, device, return_label=True)
    print('Base accuracy is {:.4f}%'.format(base_acc))
    base_corrs = preds.eq(trgs)

    cls_err_matrix = pickle.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, 'feature_error_probability.pkl'
    ), 'rb'))
    cls_err_matrix = cls_err_matrix.to(device)
    num_classes = cls_err_matrix.size(0)

    least_err_index = []
    def _hook(module, finput, foutput):
        batch_size = foutput.size(0)
        features = foutput[:, None, :, :, :].repeat(1, num_classes, 1, 1, 1)
        err_prob = cls_err_matrix.repeat(batch_size, 1, 1, 1, 1)
        reluout = torch.nn.functional.relu(features)
        mask_prob = torch.where(reluout > 0, err_prob, torch.zeros_like(err_prob))
        _, least_err_idx = mask_prob.abs().sum(dim=[2, 3, 4]).topk(1, largest=False)
        least_err_index.append(least_err_idx.cpu())

    m = None
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            m = module
    m.register_forward_hook(_hook)

    _, (preds, _) = test(model, testloader, criterion, device, return_label=True)

    least_err_index = torch.cat(least_err_index, dim=0)
    decision_corrs = preds.view(-1, 1).eq(least_err_index).sum(dim=1)
    cf_mat = confusion_matrix(base_corrs, decision_corrs)
    df = pd.DataFrame(cf_mat/cf_mat.sum(), index=[0, 1], columns=[0, 1])
    print('Decison Confusion Matrix:\n{}'.format(df))


# Borrow from: https://stackoverflow.com/questions/55681502
#   /label-smoothing-in-pytorch/66773267#66773267
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction='mean', weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else \
               loss.sum()  if self.reduction == 'sum'  else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


@torch.no_grad()
def dualeval(model1, model2, valloader, device):
    def _gini(x):
        x = F.softmax(x, dim=-1)
        x = 1 - x.square().sum(dim=-1)
        return x

    model1.eval()
    model2.eval()

    correct, total = 0, 0
    with tqdm(valloader, desc='DualEval', leave=True) as tepoch:
        for inputs, targets in tepoch:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)

            gini1 = _gini(outputs1)
            gini2 = _gini(outputs2)
            #  moutputs = F.softmax(outputs1, dim=-1) + F.softmax(outputs2, dim=-1)
            #  gini2 = _gini(moutputs)
            #  diff_gini = (gini2 - gini1) < 0
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)

            predicted = torch.where(gini1 < gini2, predicted1, predicted2)
            #  predicted = torch.where(diff_gini, predicted1, predicted2)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            tepoch.set_postfix(acc=acc)

    acc = 100. * correct / total
    return acc


@dispatcher.register('dual')
def dual(opt, model, device):
    # state1: finetune classifier1 only
    _, (trainloader, _, testloader) = load_dataset(opt, noise=(False, False))
    model2 = copy.deepcopy(model)
    #  model2 = construct_model(opt, model2, patch=False)
    #  model2 = model2.to(device)
    #
    #  for name, module in model2.named_modules():
    #      if isinstance(module, nn.Linear):
    #          for param in module.parameters():
    #              param.requires_grad = True
    #      else:
    #          for param in module.parameters():
    #              param.requires_grad = False
    #
    #  criterion = LabelSmoothingLoss(smoothing=0.025)
    #  optimizer = torch.optim.SGD(
    #      filter(lambda p: p.requires_grad, model2.parameters()),
    #      lr=opt.lr*0.1, momentum=opt.momentum, weight_decay=opt.weight_decay
    #  )
    #  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    #
    #  best_acc = 0
    #  for epoch in range(0, opt.crt_epoch//3):
    #      print('Epoch: {}'.format(epoch))
    #      train(model2, trainloader, optimizer, criterion, device)
    #      acc, *_ = eval(model2, testloader, criterion, device)
    #      if acc > best_acc:
    #          print('Saving...')
    #          state = {
    #              'fepoch': epoch,
    #              'net': model2.state_dict(),
    #              'optim': optimizer.state_dict(),
    #              'sched': scheduler.state_dict(),
    #              'acc': acc
    #          }
    #          torch.save(state, get_model_path(opt, state=f'correct_{opt.fs_method}_none'))
    #          best_acc = acc
    #      scheduler.step()
    #  print('[info] the finetune accuracy is {:.4f}%'.format(best_acc))
    #  model2 = model2.cpu()

    # state2: train classifier2 and patch module
    _, (trainloader, valloader, _) = load_dataset(opt, noise=(True, False), prob=1)
    model3 = copy.deepcopy(model)
    model3 = construct_model(opt, model3, patch=True)
    model3 = model3.to(device)

    for name, module in model3.named_modules():
        #  if isinstance(module, nn.Linear) or 'cru' in name:
        if 'cru' in name:
            if 'cru' in name and opt.crt_type == 'replace':
                conv_module_name = name.rstrip('.cru') + '.conv'
                conv_module = rgetattr(model3, conv_module_name)
                indices = rgetattr(model3, name.rstrip('.cru')).indices
                state_dict = conv_module.state_dict()
                for k in state_dict:
                    state_dict[k] = state_dict[k][indices]
                module.load_state_dict(state_dict)
            for param in module.parameters():
                param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model3.parameters()),
        lr=opt.lr*0.5, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0
    for epoch in range(0, opt.crt_epoch):
        print('Epoch: {}'.format(epoch))
        train(model3, trainloader, optimizer, criterion, device)
        acc, *_ = test(model3, valloader, criterion, device)
        if acc > best_acc:
            print('Saving...')
            state = {
                'pepoch': epoch,
                'net': model3.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'acc': acc
            }
            torch.save(state, get_model_path(opt, state=f'correct_{opt.fs_method}_patch'))
            best_acc = acc
        scheduler.step()
    print('[info] the validated robust accuracy is {:.4f}%'.format(best_acc))

    # Evaluate
    ckp = torch.load(get_model_path(opt, state=f'correct_{opt.fs_method}_none'))
    model2.load_state_dict(ckp['net'])
    model2 = model2.to(device)
    ckp = torch.load(get_model_path(opt, state=f'correct_{opt.fs_method}_patch'))
    model3.load_state_dict(ckp['net'])
    _, (_, _, testloader) = load_dataset(opt, noise=(False, False))
    std_acc = dualeval(model2, model3, testloader, device)
    #  std_acc, *_ = test(model2, testloader, criterion, device)
    print('[info] the normal accuracy is {:.4f}%'.format(std_acc))
    _, (_, _, testloader) = load_dataset(opt, noise=(False, True))
    rob_acc = dualeval(model2, model3, testloader, device)
    #  rob_acc, *_ = test(model3, testloader, criterion, device)
    print('[info] the robust accuracy is {:.4f}%'.format(rob_acc))


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = load_model(opt, pretrained=True)
    dispatcher(opt, model, device)


if __name__ == '__main__':
    main()

