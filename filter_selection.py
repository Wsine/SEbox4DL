import copy
from utils import get_model_path

import torch

from model import load_model
from dataset import load_dataset
from arguments import parser
from train import train, eval
from utils import *


def resume_model(opt):
    model = load_model()
    ckp = torch.load(
        get_model_path(opt),
        map_location=torch.device("cpu")
    )
    model.load_state_dict(ckp["net"])
    return model, ckp


def overall_improve(opt, model, ckp, dataloaders, device):
    print("[info] Overall improvement")
    model1 = copy.deepcopy(model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model1.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    optimizer.load_state_dict(ckp["optim"])
    scheduler.load_state_dict(ckp["sched"])

    trainloader, _, testloader = dataloaders
    train(model1, trainloader, optimizer, criterion, device)
    eval(model1, testloader, criterion, device)

    state = model.state_dict()
    state1 = model1.to("cpu").state_dict()
    dstate1 = {k: state1[k] - state[k] for k in state.keys()}

    return dstate1


def error_bias_improve(opt, model, ckp, datasets, dataloaders, device):
    print("[info] Error bias improvement")
    trainset, valset, testset = datasets
    _, valloader, testloader = dataloaders

    model2 = copy.deepcopy(model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    _, pred_result = eval(model2, valloader, criterion, device, desc=" Extract")

    class_weights = [0 for _ in range(len(testset.classes))]
    for sample, pred in zip(valset, pred_result):
        if not pred:
            _, label = sample
            class_weights[label] += 1
    sum_weight = sum(class_weights)
    class_weights = [w/sum_weight for w in class_weights]

    class_sampler = torch.utils.data.WeightedRandomSampler(
        weights=[class_weights[c] for _, c in trainset],
        num_samples=len(trainset),
        replacement=True
    )
    classloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, sampler=class_sampler, num_workers=2
    )

    optimizer = torch.optim.SGD(
        model2.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    optimizer.load_state_dict(ckp["optim"])
    scheduler.load_state_dict(ckp["sched"])

    train(model2, classloader, optimizer, criterion, device)
    eval(model2, testloader, criterion, device)

    state = model.state_dict()
    state2 = model2.to("cpu").state_dict()
    dstate2 = {k: state2[k] - state[k] for k in state.keys()}

    return dstate2


def differentiate_weights(opt, oa_dstate, oe_dstate):
    def _mask_out_smallest(z):
        r = opt.mask_smallest_ratio
        t = z.abs().view(-1).topk((int)(z.numel()*r), largest=False, sorted=True).values[-1]
        return z.masked_fill(z.abs() < t, 1e-6)

    suspicious = {}

    state_keys = [k for k in oa_dstate.keys() if "conv" in k]
    for k in state_keys:
        num_filters = oa_dstate[k].size(0)
        oak_dstate = _mask_out_smallest(oa_dstate[k])
        oek_dstate = _mask_out_smallest(oe_dstate[k])
        ind_map = (oak_dstate > 0) != (oek_dstate > 0)
        sum_indirection = ind_map.view((num_filters, -1)).sum(dim=1)
        r = opt.suspicious_ratio
        indices = sum_indirection.topk((int)(num_filters*r)).indices
        suspicious[k] = indices.tolist()

    return suspicious


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model, ckp = resume_model(opt)
    datasets, dataloaders = load_dataset(opt, return_set=True)

    oa_dstate = overall_improve(opt, model, ckp, dataloaders, device)
    oe_dstate = error_bias_improve(opt, model, ckp, datasets, dataloaders, device)

    suspicious_filters = differentiate_weights(opt, oa_dstate, oe_dstate)
    preview_object(suspicious_filters)
    export_object(opt, "suspicious.json", suspicious_filters)


if __name__ == "__main__":
    main()

