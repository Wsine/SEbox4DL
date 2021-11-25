import copy
from zipfile import ZipFile

import torch
from torch import nn

from src.runners.eval import eval_accuracy
from src.utils import*


def __absent_mutate__(ctx, model, data_loader):
    '''
    buggy filters means that after we delete the filter, the valid accuracy will increase somehow.
    So this is where a key distinction between terms comes in handy:
    whereas in the 1 channel case, where the term filter and kernel are interchangeable. In the general case,
    they’re actually pretty different. Each filter actually happens to be a collection of kernels, with there
    being one kernel for every single input channel to the layer, and each kernel being unique.

    :param ctx: context
    :param model: trained model
    :param data_loader: validation or test loader
    :return: find the buggy filters for dataset level
    '''
    buggy_filters = {'acc_diff': [], 'buggy_filter_index': []}
    buggy_filters_by_layer = []

    std_acc = eval_accuracy(ctx=ctx, model=model, testloader=data_loader, desc='std_acc')
    print(f"Standard accuracy is: {std_acc}")

    state_dict_original = copy.deepcopy(model.state_dict())
    layer_list = list(state_dict_original)

    mutated_models = []

    for layer_index, layer in enumerate(state_dict_original):
        layer_size = state_dict_original[layer].size()
        acc_diff_layer = []
        buggy_filter_index_layer = []

        keep_model_layer = False
        # Explore and find step: finding the buggy filters
        if len(layer_size) == 4:
            # print(f"----Exploration: Current Layer is {layer}, Size is {state_dict_original[layer].size()}----")
            for filter_index in range(layer_size[0]):
                state_dict_modified = copy.deepcopy(state_dict_original)
                state_dict_modified[layer][filter_index] = state_dict_modified[layer][filter_index] * 0

                # also set the corresponding bias to zero
                if str(layer_list[layer_index+1]).endswith('bias'):
                    state_dict_modified[layer_list[layer_index + 1]][filter_index] = 0

                model.load_state_dict(state_dict_modified)
                mutate_acc = eval_accuracy(ctx=ctx, model=model, testloader=data_loader, desc='std_acc')

                acc_diff = torch.tensor(mutate_acc) - torch.tensor(std_acc)
                acc_diff_layer.append(acc_diff.tolist())
                if acc_diff > 0:
                    buggy_filter_index_layer.append(filter_index)

                    # one layer only keep one mutated model
                    if keep_model_layer is False:
                        mutated_models.append(copy.deepcopy(model.state_dict()))
                        keep_model_layer = True

            # keep the buggy filter info
            buggy_filters['acc_diff'].extend(acc_diff_layer)
            buggy_filters['buggy_filter_index'].extend(buggy_filter_index_layer)
            buggy_filters_by_layer.append({'layer': layer, 'acc_diff_layer': acc_diff_layer, 'buggy_filter_index_layer': buggy_filter_index_layer})

    return buggy_filters, buggy_filters_by_layer, mutated_models


def absent_mutate(ctx, model, valloader):
    guard_folder(ctx)
    base_acc = eval_accuracy(ctx, model, valloader, desc='Eval')

    def _mask_out_channel(chn):
        def __hook(module, finput, foutput):
            foutput[:, chn] = 0
            return foutput
        return __hook

    result = {}
    conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for lidx, lname in enumerate(ctx.tqdm(conv_names, desc='Modules')):
        module = rgetattr(model, lname)
        acc_diff = []
        for chn in ctx.tqdm(range(module.out_channels), desc='Filters', leave=False):
            handle = module.register_forward_hook(_mask_out_channel(chn))
            acc = eval_accuracy(ctx, model, valloader, desc='Eval')
            acc_diff.append(acc - base_acc)
            handle.remove()
        result[lname] = acc_diff
        if lidx > 5:
            break

    return result


def pack_mutants(ctx, model, analysis):
    model_state = model.state_dict()

    mutant_idx = 0
    for layer, diff in analysis.items():
        for i, d in enumerate(diff):
            if d > 0:  # accuracy improved => buggy filter
                copy_state = copy.deepcopy(model_state)
                for k in copy_state.keys():
                    if k.startswith(layer):
                        copy_state[k][i] = 0
                save_state = {
                    'net': copy_state,
                    'diff_acc': d,
                }
                torch.save(save_state, get_model_path(ctx, state=f'mutant_{mutant_idx}'))
                mutant_idx += 1

    mutants_path = os.path.join(get_output_path(ctx), 'mutants.zip')
    with ZipFile(mutants_path, 'w') as zipobj:
        for idx in range(mutant_idx):
            model_path = get_model_path(ctx, state=f'mutant_{idx}')
            zipobj.write(model_path)

    return mutants_path

