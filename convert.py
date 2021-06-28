import os
import urllib

import numpy as np
import torch

from model import load_model
from arguments import parser
from utils import *


def convert_dealexnet(model):
    weights_path = "weights/deepcorrect_alexnet_weights.npy"
    if not os.path.exists(weights_path):
        path_dir, _ = weights_path.split('/')
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)
        url = "https://github.com/tsborkar/DeepCorrect/blob/master/cifar_models/CIFAR_100_fine_best_model.npy?raw=true"
        urllib.request.urlretrieve(url, weights_path)
    weights = np.load(weights_path, allow_pickle=True, encoding='bytes')


    with torch.no_grad():
        m1, m2 = len(model.basic1), len(model.basic2)
        for i in range(m1 + m2):
            a, b, c, d, e, f = weights[i*6:(i+1)*6]
            if i < m1:
                module = model.basic1[i]
            else:
                module = model.basic2[i - m1]
            module.conv.weight.copy_(torch.from_numpy(a).float())
            module.conv.bias.copy_(torch.from_numpy(b).float())
            module.bn.weight.copy_(torch.from_numpy(c).float())
            module.bn.bias.copy_(torch.from_numpy(d).float())
            module.bn.running_mean.copy_(torch.from_numpy(e).float())
            module.bn.running_var.copy_(torch.from_numpy(f).float())

    return model


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    model = load_model(opt)
    model = convert_dealexnet(model)
    state = {
        "net": model.state_dict()
    }
    torch.save(state, get_model_path(opt, state="test"))


if __name__ == "__main__":
    main()

