import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix

from model import resume_model
from dataset import load_dataset
from arguments import parser
from train import train, eval
from utils import *


class CorrectUnit(nn.Module):
    def __init__(self, num_filters, Di, k):
        super(CorrectUnit, self).__init__()
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

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        out += residual
        return out


class ConvCorrect(nn.Module):
    def __init__(self, conv_layer, indices):
        super(ConvCorrect, self).__init__()
        self.indices = indices
        self.conv = conv_layer
        num_filters = len(indices)
        self.cru = CorrectUnit(num_filters, num_filters, 3)

    def forward(self, x):
        out = self.conv(x)
        out[:, self.indices] = self.cru(out[:, self.indices])
        return out


def construct_model(opt, model):
    sus_filters = json.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, "suspicious_filters.json"
    )))
    for weights_name, indices in sus_filters.items():
        layer_name = '.'.join(weights_name.split('.')[:-1])
        correct_unit = ConvCorrect(rgetattr(model, layer_name), indices)
        rsetattr(model, layer_name, correct_unit)
    return model


def retrain(opt, model, ckp, dataloaders, device):
    trainloader, _, testloader = dataloaders

    model = construct_model(opt, model)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = ckp["acc"]
    for epoch in range(0, opt.correct_epoch):
        print("Epoch: {}".format(epoch))
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = eval(model, testloader, criterion, device)
        if acc > best_acc:
            print("Saving...")
            state = {
                "epoch": ckp["epoch"],
                "cepoch": epoch,
                "net": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "acc": acc
            }
            torch.save(state, get_model_path(opt, state="correct"))
            best_acc = acc
        scheduler.step()


def patch(opt, model, dataloaders, device):
    _, _, testloader = dataloaders
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    base_acc, (preds, trgs) = eval(model, testloader, criterion, device, return_label=True)
    print("Base accuracy is {:.4f}%".format(base_acc))
    base_corrs = preds.eq(trgs)

    feat_err_prob = pickle.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, "feature_error_probability.pkl"
    ), "rb"))

    decision_wgt = []

    def _adjust_features(layer_name):
        def __hook(module, finput, foutput):
            zero_output = torch.zeros(foutput.size(), device=device)
            ones_output = torch.ones(foutput.size(), device=device)

            layer_eprob = feat_err_prob[layer_name].to(device)
            err_prob = torch.stack(foutput.size(0) * [layer_eprob])
            relu_out = torch.nn.functional.relu(foutput)

            act_wgt = torch.where(relu_out > 0, err_prob, zero_output) \
                           .view(relu_out.size(0), -1).mean(dim=1)
            nonact_wgt = torch.where(relu_out <= 0, err_prob, zero_output) \
                           .view(relu_out.size(0), -1).mean(dim=1)

            decision_wgt.append((act_wgt > nonact_wgt).cpu())
            wgt_mask = torch.stack([ones_output[0] if larger else zero_output[0]
                                    for larger in (act_wgt > nonact_wgt)])
            prune_mask = torch.bernoulli(torch.clamp(err_prob, min=0., max=1.))
            act_mask = (relu_out > 0)
            comb_mask = torch.logical_and(prune_mask, torch.logical_and(wgt_mask, act_mask))
            return torch.where(comb_mask, zero_output, foutput)
        return __hook

    #  for lname, module in model.named_modules():
    #      if isinstance(module, nn.BatchNorm2d):
    #          module.register_forward_hook(_adjust_features(lname))
    #          break

    n, m = None, None
    for lname, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            n, m = lname, module
    m.register_forward_hook(_adjust_features(n))


    acc, (preds, trgs) = eval(model, testloader, criterion, device, return_label=True)
    print("Patch accuracy is {:.4f}%".format(acc))
    corrs = preds.eq(trgs)

    decision_wgt = torch.cat(decision_wgt, dim=0)
    cf_matrix = confusion_matrix(base_corrs, decision_wgt)
    df = pd.DataFrame(cf_matrix/cf_matrix.sum(), index=[0, 1], columns=[0, 1])
    print("Decison Confusion Matrix:\n", df)

    cf_matrix = confusion_matrix(base_corrs, corrs)
    df = pd.DataFrame(cf_matrix/cf_matrix.sum(), index=[0, 1], columns=[0, 1])
    print("Prediction Confusion Matrix:\n", df)


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model, ckp = resume_model(opt)
    _, dataloaders = load_dataset(opt)

    if opt.fs_method in ("bpindiret", "perfswap"):
        retrain(opt, model, ckp, dataloaders, device)
    elif opt.fs_method == "featweight":
        patch(opt, model, dataloaders, device)
    else:
        raise ValueError("Invalid method")


if __name__ == "__main__":
    main()

