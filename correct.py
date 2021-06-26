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


class ConvCorrect(nn.Module):
    def __init__(self, conv_layer, indices):
        super(ConvCorrect, self).__init__()
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


class ConvCorrect2(nn.Module):
    def __init__(self, conv_layer, indices):
        super(ConvCorrect2, self).__init__()
        self.indices = indices
        self.conv = conv_layer
        self.cru = nn.Conv2d(
            conv_layer.in_channels,
            len(indices),
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=False)

    def forward(self, x):
        out = self.conv(x)
        out[:, self.indices] = self.cru(x)
        return out


def construct_model(opt, model):
    sus_filters = json.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, f"susp_filters_{opt.fs_method}.json"
    )))
    for idx, (name, ranking) in enumerate(sus_filters.items()):
        layer_name = name.rstrip(".weight")
        r = 1.5 * opt.suspicious_ratio if idx == 0 else opt.suspicious_ratio
        num_susp = (int)(len(ranking) * r)
        indices = ranking[:num_susp]
        if opt.correct_type == "crtunit":
            correct_unit = ConvCorrect(rgetattr(model, layer_name), indices)
        elif opt.correct_type == "replace":
            correct_unit = ConvCorrect2(rgetattr(model, layer_name), indices)
        else:
            raise ValueError("Invalid correct type")
        rsetattr(model, layer_name, correct_unit)
    return model


def coordinate_filters(opt, model):
    for name, module in model.named_modules():
        if "cru" in name:
            conv_name = name.rstrip("cru") + "conv"
            conv_module = rgetattr(model, conv_name)

            precision = 1000
            conv_weight = conv_module.weight.view(conv_module.weight.size(0), -1)
            conv_weight = (conv_weight * precision).int().float()
            conv_rank = torch.matrix_rank(conv_weight)
            cru_weight = module.weight.view(module.weight.size(0), -1)

            unrank_idx = []
            for i, w in enumerate(cru_weight):
                w = (w.view(-1).view(1, -1) * precision).int().float()
                cat_weight = torch.cat([conv_weight, w])
                new_rank = torch.matrix_rank(cat_weight)
                if new_rank <= conv_rank:
                    unrank_idx.append(i)

            if unrank_idx:
                with torch.no_grad():
                    torch.nn.init.xavier_uniform_(module.weight[unrank_idx])


def retrain(opt, model, ckp, device):
    _, (trainloader, valloader, _) = load_dataset(opt, noise=(True, True))

    model = construct_model(opt, model)
    model = model.to(device)

    for name, module in model.named_modules():
        if "cru" in name:
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

    best_acc, *_ = eval(model, valloader, criterion, device, desc="Baseline")
    for epoch in range(0, opt.correct_epoch):
        print("Epoch: {}".format(epoch))
        train(model, trainloader, optimizer, criterion, device)
        acc, *_ = eval(model, valloader, criterion, device)
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
            torch.save(state, get_model_path(opt, state=f"correct_{opt.fs_method}"))
            best_acc = acc
        scheduler.step()
        if opt.correct_type == "replace":
            coordinate_filters(opt, model)
    print("[info] the best retrain accuracy is {:.4f}%".format(best_acc))

    del trainloader, valloader
    state = torch.load(get_model_path(opt, state=f"correct_{opt.fs_method}"))
    model.load_state_dict(state["net"])
    _, (_, _, testloader) = load_dataset(opt, noise=(False, False))
    normal_acc, *_ = eval(model, testloader, criterion, device, desc="Normal")
    print("[info] the normal accuracy is {:.4f}%".format(normal_acc))
    _, (_, _, testloader) = load_dataset(opt, noise=(False, True), prob=1)
    robust_acc, *_ = eval(model, testloader, criterion, device, desc="Robustness")
    print("[info] the robustness accuracy is {:.4f}%".format(robust_acc))


@torch.no_grad()
def patch(opt, model, device):
    _, (_, _, testloader) = load_dataset(opt)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    base_acc, (preds, trgs) = eval(model, testloader, criterion, device, return_label=True)
    print("Base accuracy is {:.4f}%".format(base_acc))
    base_corrs = preds.eq(trgs)

    cls_err_matrix = pickle.load(open(os.path.join(
        opt.output_dir, opt.dataset, opt.model, "feature_error_probability.pkl"
    ), "rb"))
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

    _, (preds, _) = eval(model, testloader, criterion, device, return_label=True)

    least_err_index = torch.cat(least_err_index, dim=0)
    decision_corrs = preds.view(-1, 1).eq(least_err_index).sum(dim=1)
    cf_mat = confusion_matrix(base_corrs, decision_corrs)
    df = pd.DataFrame(cf_mat/cf_mat.sum(), index=[0, 1], columns=[0, 1])
    print("Decison Confusion Matrix:\n{}".format(df))


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model, ckp = resume_model(opt, state="primeval")

    if opt.fs_method in ("bpindiret", "featswap", "wgtchange", "lowrank"):
        retrain(opt, model, ckp, device)
    elif opt.fs_method == "featwgting":
        patch(opt, model, device)
    else:
        raise ValueError("Invalid method")


if __name__ == "__main__":
    main()

