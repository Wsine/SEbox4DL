import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from model import resume_model
from dataset import load_dataset
from arguments import parser
from train import train, eval
from utils import *


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


def construct_error_bias_dataloader(opt, model, datasets, dataloaders, device):
    trainset, valset, _ = datasets
    _, valloader, _ = dataloaders

    model2 = copy.deepcopy(model).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    _, (pred_labels, trg_labels) = eval(
        model2, valloader, criterion, device, desc=" Extract", return_label=True)
    pred_result = pred_labels.eq(trg_labels).tolist()

    class_weights = [0 for _ in range(len(opt.classes))]
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

    return classloader


def error_bias_improve(opt, model, ckp, datasets, dataloaders, device):
    print("[info] Error bias improvement")

    classloader = construct_error_bias_dataloader(opt, model, datasets, dataloaders, device)
    _, _, testloader = dataloaders

    model2 = copy.deepcopy(model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
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


def backprob_indirection(opt, model, ckp, datasets, dataloaders, device):
    oa_dstate = overall_improve(opt, model, ckp, dataloaders, device)
    oe_dstate = error_bias_improve(opt, model, ckp, datasets, dataloaders, device)

    susp_filters = differentiate_weights(opt, oa_dstate, oe_dstate)
    return susp_filters


def extract_feature_map(lname, model, dataloader, device):
    feature_map = []

    def _hook(module, finput, foutput):
        feature_map.append(foutput.detach().cpu())

    module = rgetattr(model, lname)
    handle = module.register_forward_hook(_hook)

    criterion = torch.nn.CrossEntropyLoss()
    base_acc, (pred_labels, trg_labels) = eval(
        model, dataloader, criterion, device,
        desc="Evaluate", return_label=True, tqdm_leave=False)
    feature_map = torch.cat(feature_map, dim=0)

    handle.remove()

    return feature_map, (pred_labels, trg_labels), base_acc


# TODO: unfinished
def performance_swap(opt, model, datasets, dataloaders, device):
    trainloader, _, _, = dataloaders
    classloader = construct_error_bias_dataloader(opt, model, datasets, dataloaders, device)

    model2 = copy.deepcopy(model).to(device)

    for lname, module in tqdm(model2.named_modules(), desc=" Modules"):
        if isinstance(module, nn.Conv2d):
            print("[info] Extracting feature map of {} layer".format(lname))
            fmap, (_, trgs), base_acc = extract_feature_map(lname, model2, trainloader, device)

            def _substitute_feature(filter_index):
                def __hook(module, finput, foutput):
                    foutput[filter_index] = fmap[filter_index]
                return __hook

            for fidx in tqdm(range(module.out_channels), desc=" Filters", leave=False):
                module.register_forward_hook(_substitute_feature(fidx))

                correct, total = 0, 0
                with tqdm(classloader, desc="    Swap") as tepoch:
                    for inputs, targets in tepoch:
                        proc_trgs = targets
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model2(inputs)

                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                        acc = 100. * correct / total
                        tepoch.set_postfix(acc=acc)
                acc = 100. * correct / total

        break

    return None


def feature_weighted(opt, model, dataloaders, device):
    _, valloader, _, = dataloaders
    num_classes = len(opt.classes)

    model2 = copy.deepcopy(model).to(device)

    feat_err_prob = {}
    num_modules = sum(1 for _ in model2.modules())
    for lname, module in tqdm(model2.named_modules(), total=num_modules, desc="Weighting"):
        if isinstance(module, nn.BatchNorm2d):
            fmaps, (preds, trgs), _ = extract_feature_map(lname, model2, valloader, device)
            fmaps = torch.nn.functional.relu(fmaps)

            cls_err_matrix = []
            for cls_i in range(num_classes):
                cfmaps = fmaps[torch.logical_and(trgs == cls_i, preds == trgs)]
                efmaps = fmaps[torch.logical_and(trgs == cls_i, preds != trgs)]
                cweights = (cfmaps > 0).sum(dim=0).div(cfmaps.size(0))
                eweights = (efmaps > 0).sum(dim=0).div(efmaps.size(0))
                clsi_err_mat = torch.mul(cweights, eweights)
                cls_err_matrix.append(clsi_err_mat)
            cls_err_matrix = torch.stack(cls_err_matrix, dim=0)

            cls_weighted = torch.stack(
                [(trgs == i).sum() for i in range(len(opt.classes))]
            ).div(trgs.numel())
            err_matrix = cls_err_matrix \
                .view(num_classes, -1) \
                .mul(cls_weighted.view(num_classes, 1)) \
                .view(cls_err_matrix.size()) \
                .sum(dim=0)

            feat_err_prob[lname] = err_matrix

    return feat_err_prob


def main():
    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model, ckp = resume_model(opt)
    datasets, dataloaders = load_dataset(opt, return_set=True, return_loader=True)
    opt.classes = datasets[2].classes

    if opt.fs_method == "bpindiret":
        result = backprob_indirection(
            opt, model, ckp, datasets, dataloaders, device
        )
        result_name = "susp_filters.json"
        preview_object(result)
    elif opt.fs_method == "perfswap":
        raise RuntimeError("unfinished implementation")
        #  result = performance_swap(
        #      opt, model, datasets, dataloaders, device
        #  )
        result_name = "susp_filters.json"
    elif opt.fs_method == "featweight":
        result = feature_weighted(
            opt, model, dataloaders, device
        )
        result_name = "feature_error_probability.pkl"
    else:
        raise ValueError("Invalid method")

    export_object(opt, result_name, result)


if __name__ == "__main__":
    main()

