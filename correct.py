import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model, ckp = resume_model(opt)
    model = construct_model(opt, model)
    model = model.to(device)

    trainloader, _, testloader = load_dataset(opt)
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
        acc = eval(model, testloader, criterion, device)
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


if __name__ == "__main__":
    main()

