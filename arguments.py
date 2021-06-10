import argparse


devices = ["cpu", "cuda"]
datasets = ["cifar10", "cifar100"]
models = ["resnet34", "resnet50"]
noises = ["gaussion", "awgn"]
fsmethods = ["bpindiret", "perfswap", "featweight"]

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data")
parser.add_argument("--output_dir", default="output")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--device", default="cuda", choices=devices)
parser.add_argument("-r", "--resume", action="store_true")
parser.add_argument("-b", "--batch_size", type=int, default=256)
parser.add_argument("-e", "--max_epoch", type=int, default=50)
parser.add_argument("-m", "--model", type=str, default="resnet34", choices=models)
data_group = parser.add_argument_group('dataset')
data_group.add_argument("-d", "--dataset", type=str, default="cifar10", choices=datasets)
data_group.add_argument("-n", "--noise_type", type=str, default="gaussion", choices=noises)
data_group.add_argument("--gaussion_std", type=float, default=1.5)
optim_group = parser.add_argument_group('optimizer')
optim_group.add_argument("--lr", type=float, default=0.01, help="learning rate")
optim_group.add_argument("--momentum", type=float, default=0.9)
optim_group.add_argument("--weight_decay", type=float, default=5e-4)
select_group = parser.add_argument_group('selection')
select_group.add_argument("--fs_method", type=str, default="featweight", choices=fsmethods)
select_group.add_argument("--mask_smallest_ratio", type=float, default=0.1)
select_group.add_argument("--suspicious_ratio", type=float, default=0.05)
correct_group = parser.add_argument_group('correct')
correct_group.add_argument("--correct_epoch", type=int, default=20)
