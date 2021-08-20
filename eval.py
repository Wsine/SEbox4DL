import torch

from dataset import load_dataset
from model import load_model
from arguments import commparser as parser
from train import test
from utils import *


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    _, testloader = load_dataset(opt, split='test')
    model = load_model(opt, pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    acc, _ = test(model, testloader, criterion, device)
    print("[info] the base accuracy is {:.4f}%".format(acc))
    _, perturbloader = load_dataset(opt, split='test', noise=True, noise_type='append')
    acc, _ = test(model, perturbloader, criterion, device)
    print("[info] the robustness accuracy is {:.4f}%".format(acc))


if __name__ == "__main__":
    main()

