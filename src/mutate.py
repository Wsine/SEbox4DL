import torch
import copy
from src.eval import eval_accuracy


def absent_mutate(ctx, model, data_loader):
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
    for layer_index, layer in enumerate(state_dict_original):
        layer_size = state_dict_original[layer].size()
        acc_diff_layer = []
        buggy_filter_index_layer = []

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

            # keep the buggy filter info
            buggy_filters['acc_diff'].extend(acc_diff_layer)
            buggy_filters['buggy_filter_index'].extend(buggy_filter_index_layer)
            buggy_filters_by_layer.append({'layer': layer, 'acc_diff_layer': acc_diff_layer, 'buggy_filter_index_layer': buggy_filter_index_layer})
    return buggy_filters, buggy_filters_by_layer
