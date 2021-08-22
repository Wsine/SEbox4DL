import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from model import load_checkpoint, load_model
from dataset import load_dataset
from arguments import selparser as parser, Args
from train import train, test
from utils import *


dispatcher = AttrDispatcher('fs_method')


def overall_improve(opt, model, ckp, dataloaders, device):
    print('[info] Overall improvement')
    model1 = copy.deepcopy(model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model1.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    optimizer.load_state_dict(ckp['optim'])
    scheduler.load_state_dict(ckp['sched'])

    trainloader, _, testloader = dataloaders
    train(model1, trainloader, optimizer, criterion, device)
    test(model1, testloader, criterion, device)

    state = model.state_dict()
    state1 = model1.to('cpu').state_dict()
    dstate1 = {k: state1[k] - state[k] for k in state.keys()}

    return dstate1


def construct_error_bias_dataloader(opt, model, datasets, dataloaders, device):
    trainset, valset, _ = datasets
    _, valloader, _ = dataloaders
    num_classes = Args.get_num_class(opt.dataset)

    model2 = copy.deepcopy(model).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    _, (pred_labels, trg_labels) = test(
        model2, valloader, criterion, device, desc=' Extract', return_label=True)
    pred_result = pred_labels.eq(trg_labels).tolist()

    class_weights = [0 for _ in range(num_classes)]
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
    print('[info] Error bias improvement')

    classloader = construct_error_bias_dataloader(opt, model, datasets, dataloaders, device)
    _, _, testloader = dataloaders

    model2 = copy.deepcopy(model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model2.parameters(),
        lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    optimizer.load_state_dict(ckp['optim'])
    scheduler.load_state_dict(ckp['sched'])

    train(model2, classloader, optimizer, criterion, device)
    test(model2, testloader, criterion, device)

    state = model.state_dict()
    state2 = model2.to('cpu').state_dict()
    dstate2 = {k: state2[k] - state[k] for k in state.keys()}

    return dstate2


def differentiate_weights(opt, oa_dstate, oe_dstate):
    def _mask_out_smallest(z):
        r = opt.mask_smallest_ratio
        t = z.abs().view(-1).topk((int)(z.numel()*r), largest=False, sorted=True).values[-1]
        return z.masked_fill(z.abs() < t, 1e-6)

    suspicious = {}

    state_keys = [k for k in oa_dstate.keys() if 'conv' in k]
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


@dispatcher.register('bpindiret')
def backprob_indirection(opt, model, ckp, device):
    datasets, dataloaders = load_dataset(opt, return_set=True, return_loader=True)

    oa_dstate = overall_improve(opt, model, ckp, dataloaders, device)
    oe_dstate = error_bias_improve(opt, model, ckp, datasets, dataloaders, device)

    susp_filters = differentiate_weights(opt, oa_dstate, oe_dstate)
    return susp_filters


def extract_feature_map(lname, model, dataloader, device):
    feature_map = []

    def _hook(module, finput, foutput):
        feature_map.append(foutput.detach())

    module = rgetattr(model, lname)
    handle = module.register_forward_hook(_hook)

    criterion = torch.nn.CrossEntropyLoss()
    base_acc, (pred_labels, trg_labels) = test(
        model, dataloader, criterion, device,
        desc='Extract', return_label=True, tqdm_leave=False)
    feature_map = torch.cat(feature_map, dim=0)

    handle.remove()

    return feature_map, (pred_labels, trg_labels), base_acc


@dispatcher.register('featwgting')
def feature_weighting(opt, model, _, device):
    _, (_, valloader, _) = load_dataset(opt)
    num_classes = Args.get_num_class(opt.dataset)

    model2 = copy.deepcopy(model).to(device)

    layer_name = None
    for lname, module in model2.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            layer_name = lname

    fmaps, (preds, trgs), _ = extract_feature_map(layer_name, model2, valloader, device)
    fmaps = torch.nn.functional.relu(fmaps).cpu()

    cls_err_matrix = []
    for cls_i in range(num_classes):
        cfmaps = fmaps[torch.logical_and(trgs == cls_i, preds == trgs)]
        efmaps = fmaps[torch.logical_and(trgs == cls_i, preds != trgs)]
        cweights = (cfmaps > 0).sum(dim=0).div(cfmaps.size(0))
        eweights = (efmaps > 0).sum(dim=0).div(efmaps.size(0))
        clsi_err_mat = torch.sub(cweights, eweights)
        cls_err_matrix.append(clsi_err_mat)
    cls_err_matrix = torch.stack(cls_err_matrix, dim=0)

    print('[info] Done of weighting the feature activations')

    return cls_err_matrix


@dispatcher.register('wgtchange')
def weight_change(opt, model, _, device):
    _, (_, valloader, testloader) = load_dataset(opt, noise=(True, True), prob=1)
    model = model.to(device)
    origin_state = model.state_dict()

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
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

    best_acc = 0
    for epoch in range(5):
        print('Calibrate epoch: {}'.format(epoch))
        train(model, valloader, optimizer, criterion, device)
        acc, _ = test(model, testloader, criterion, device)
        if acc > best_acc:
            updated_state = model.state_dict()
            best_acc = acc
        scheduler.step()

    suspicious = {}
    for k in updated_state.keys():
        if not 'conv' in k: continue
        num_filters = updated_state[k].size(0)
        diff_state = updated_state[k] - origin_state[k]
        sum_change = diff_state.abs().view(num_filters, -1).sum(dim=1)
        r = opt.suspicious_ratio
        indices = sum_change.topk((int)(num_filters*r)).indices
        suspicious[k] = indices.tolist()

    return suspicious


@dispatcher.register('lowrank')
def lower_rank(opt, model, _, device):
    _, (_, valloader, _) = load_dataset(opt, noise=(True, False), prob=1)
    model = model.to(device)
    model.eval()

    suspicious = {}
    num_modules = len(list(model.modules()))
    for lname, module in tqdm(model.named_modules(), total=num_modules, desc='Modules'):
        if isinstance(module, nn.Conv2d):
            fmaps, *_ = extract_feature_map(lname, model, valloader, device)
            fmaps = torch.nn.functional.relu(fmaps)

            b, c = fmaps.size(0), fmaps.size(1)
            rank = torch.tensor([torch.matrix_rank(fmaps[i, j, :, :]).item()
                                 for i in range(b) for j in range(c)]) \
                        .view(b, c).float().sum(0).cpu()

            rank = rank.tolist()
            indices = sorted(range(len(rank)), key=lambda i: rank[i])
            suspicious[lname] = indices

    return suspicious


@dispatcher.register('perfloss')
def performance_loss(opt, model, _, device):
    _, valloader = load_dataset(opt, split='val', noise=True, noise_type='expand')
    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    base_acc, _ = test(model, valloader, criterion, device)

    def _mask_out_channel(chn):
        def __hook(module, finput, foutput):
            foutput[:, chn] = 0
            return foutput
        return __hook

    suspicious = {}
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for lname in tqdm(conv_names, desc='Modules'):
        module = rgetattr(model, lname)
        perfloss = []
        for chn in tqdm(range(module.out_channels), desc='Filters', leave=False):
            handle = module.register_forward_hook(_mask_out_channel(chn))
            acc, _ = test(model, valloader, criterion, device, tqdm_leave=False)
            perfloss.append(base_acc - acc)
            handle.remove()

        score = sorted(perfloss)
        indices = sorted(range(len(perfloss)), key=lambda i: perfloss[i])
        suspicious[lname] = {
            'score': score,
            'indices': indices
        }

    return suspicious


@dispatcher.register('mactcalib')
@torch.no_grad()
def multi_activation_calibrate(opt, model, _, device):
    _, valloader1 = load_dataset(opt, split='val', noise=False)
    _, valloader2 = load_dataset(opt, split='val', noise=True, noise_type='expand')
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    base_acc, _ = test(model, valloader2, criterion, device, tqdm_leave=False)

    suspicious = {}
    num_modules = len(list(model.modules()))
    for lname, module in tqdm(model.named_modules(), total=num_modules, desc='Modules'):
        if isinstance(module, nn.Conv2d):
            fmaps, _, _ = extract_feature_map(lname, model, valloader1, device)

            def _substitute_feature(filter_index):
                def __hook(module, finput, foutput):
                    global fmaps_idx
                    batch_size = foutput.size(0)
                    sample_index = list(range(fmaps_idx, min(fmaps_idx+batch_size, fmaps.size(0)))) \
                                 + list(range(0, max(0, fmaps_idx+batch_size-fmaps.size(0))))
                    foutput[:, filter_index] = torch.where(
                        fmaps[sample_index, filter_index] < 0,
                        torch.zeros_like(foutput[:, filter_index]),
                        foutput[:, filter_index]
                    )
                    foutput[:, filter_index] = torch.where(
                        torch.logical_and(fmaps[sample_index, filter_index] > 0, foutput[:, filter_index] < 0),
                        -foutput[:, filter_index],
                        foutput[:, filter_index]
                    )
                    fmaps_idx = (fmaps_idx + batch_size) % fmaps.size(0)
                    return foutput
                return __hook

            recover = []
            for fidx in tqdm(range(module.out_channels), desc='Filters', leave=False):
                handler = module.register_forward_hook(_substitute_feature(fidx))
                global fmaps_idx
                fmaps_idx = 0
                calib_acc, _ = test(model, valloader2, criterion, device, tqdm_leave=False)
                recover.append(calib_acc - base_acc)
                handler.remove()

            score = sorted(recover, reverse=True)
            indices = sorted(range(len(recover)), key=lambda i: recover[i], reverse=True)
            suspicious[lname] = {
                'score': score,
                'indices': indices
            }

    return suspicious


@dispatcher.register('featswap')
@torch.no_grad()
def featuremap_swap(opt, model, _, device):
    assert hasattr(opt, 'gblur_std'), 'argument gblur_std should be set manually'
    _, valloader1 = load_dataset(opt, split='val', noise=False)
    #  _, valloader2 = load_dataset(opt, split='val', noise=True, noise_type='replace', gblur_std=opt.gblur_std)
    _, valloader2 = load_dataset(opt, split='val', noise=True, noise_type='random')
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    base_acc, _ = test(model, valloader2, criterion, device, tqdm_leave=False)

    suspicious = {}
    num_modules = len(list(model.modules()))
    for lname, module in tqdm(model.named_modules(), total=num_modules, desc='Modules'):
        if isinstance(module, nn.Conv2d):
            fmaps, _, _ = extract_feature_map(lname, model, valloader1, device)

            def _substitute_feature(filter_index):
                def __hook(module, finput, foutput):
                    global fmaps_idx
                    batch_size = foutput.size(0)
                    foutput[:, filter_index] = fmaps[fmaps_idx:fmaps_idx+batch_size, filter_index]
                    fmaps_idx += batch_size
                    return foutput
                return __hook

            recover = []
            for fidx in tqdm(range(module.out_channels), desc='Filters', leave=False):
                handler = module.register_forward_hook(_substitute_feature(fidx))
                global fmaps_idx
                fmaps_idx = 0
                swap_acc, _ = test(model, valloader2, criterion, device, tqdm_leave=False)
                recover.append(swap_acc - base_acc)
                handler.remove()

            score = sorted(recover, reverse=True)
            indices = sorted(range(len(recover)), key=lambda i: recover[i], reverse=True)
            suspicious[lname] = {
                'score': score,
                'indices': indices
            }

    return suspicious


def main():
    opt = parser.parse_args()
    print(opt)
    guard_folder(opt)

    device = torch.device(f'cuda:{opt.gpu}' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = load_model(opt, pretrained=True)
    ckp = load_checkpoint(opt) if opt.fs_method == 'bpindiret' else None

    result = dispatcher(opt, model, ckp, device)
    if opt.fs_method == 'featwgting':
        result_name = 'feature_error_probability.pkl'
    else:
        result_name = 'susp_filters.json'

    if opt.gblur_std is not None:
        a, b = int(opt.gblur_std), int(opt.gblur_std*10%10)
        suffix = f'{opt.fs_method}_std{a}d{b}'
    else:
        suffix = opt.fs_method
    export_object(opt, result_name, suffix, result)


if __name__ == '__main__':
    main()

