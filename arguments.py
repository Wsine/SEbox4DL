import argparse


devices = ["cpu", "cuda"]
datasets = ["cifar10", "cifar100"]
models = ["resnet34"]

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data")
parser.add_argument("--output_dir", default="output")
parser.add_argument("--device", default="cuda", choices=devices)
parser.add_argument("-r", "--resume", action="store_true")
parser.add_argument("-b", "--batch_size", type=int, default=256)
parser.add_argument("-e", "--max_epoch", type=int, default=20)
parser.add_argument("-d", "--dataset", type=str, default="cifar10", choices=datasets)
parser.add_argument("-m", "--model", type=str, default="resnet34", choices=models)
optim_group = parser.add_argument_group('optimizer')
optim_group.add_argument("--lr", type=float, default=0.1, help="learning rate")
optim_group.add_argument("--momentum", type=float, default=0.9)
optim_group.add_argument("--weight_decay", type=float, default=5e-4)

