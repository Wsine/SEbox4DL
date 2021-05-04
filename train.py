import os

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from arguments import parser
from utils import *


def load_dataset(opt):
    trainset = torchvision.datasets.CIFAR10(
        root=opt.data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    )
    trainloader = torch.utils.data.DataLoader(  # type: ignore
        trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root=opt.data_dir, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    )
    testloader = torch.utils.data.DataLoader(  # type: ignore
        testset, batch_size=opt.batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def load_model(opt):
    from models.cifar.resnet import ResNet34
    model = ResNet34()
    return model


def train(model, trainloader, optimizer, criterion, device):
    model.train()
    train_loss, correct, total = 0, 0, 0
    with tqdm(trainloader, desc="   Train") as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            avg_loss = train_loss / (batch_idx + 1)
            acc = 100. * correct / total
            tepoch.set_postfix(loss=avg_loss, acc=acc)


@torch.no_grad()
def eval(model, valloader, optimizer, criterion, device):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with tqdm(valloader, desc="Evaluate") as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            avg_loss = test_loss / (batch_idx + 1)
            acc = 100. * correct / total
            tepoch.set_postfix(loss=avg_loss, acc=acc)

    acc = 100. * correct / total
    return acc


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder([opt.output_dir])

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    trainloader, testloader = load_dataset(opt)
    model = load_model(opt).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start_epoch = -1
    best_acc = 0
    if opt.resume:
        ckp = torch.load(get_model_path(opt))
        model.load_state_dict(ckp["net"])
        optimizer.load_state_dict(ckp["optim"])
        scheduler.load_state_dict(ckp["sched"])
        start_epoch = ckp["epoch"]
        best_acc = ckp["acc"]

    for epoch in range(start_epoch + 1, opt.max_epoch):
        print("Epoch: {}".format(epoch))
        train(model, trainloader, optimizer, criterion, device)
        acc = eval(model, testloader, optimizer, criterion, device)
        if acc > best_acc:
            print("Saving...")
            state = {
                "epoch": epoch,
                "net": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "acc": acc
            }
            torch.save(state, get_model_path(opt))
            best_acc = acc
        scheduler.step()


if __name__ == "__main__":
    main()

