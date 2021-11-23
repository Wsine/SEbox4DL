import torch
import numpy as np
from src.utils import rgetattr


generated_number = 50000
check_nan_result = 0
def check_layer_nan(module, fea_in, fea_out):
    global check_nan_result
    out = fea_out.isnan().sum()
    check_nan_result += out


def construct_model(model):
    for layer_name, _ in model.named_modules():
        try:
            module = rgetattr(model, layer_name)
            module.register_forward_hook(hook=check_layer_nan)
        except:
            continue
    return model


def fuzz_numeric(ctx, model):
    # 1. construct the new model with a hook to check the NaN problem
    new_model = construct_model(model=model)

    # 2. generate new image
    new_data_list = []
    for index in range(generated_number):
        bgr_image = np.random.randint(0, 256, size=[3, 64, 64])
        new_data_list.append(bgr_image)

    # 3. test and check the NaN problem, if true store the image
    fuzz_data_list = []
    global check_nan_result
    # for new_data in new_data_list:
    for new_data in new_data_list:
        check_nan_result = 0
        new_model(torch.tensor(new_data, dtype=torch.float).unsqueeze(dim=0).to(ctx.device))
        if check_nan_result > 0:
            print("Find the data!")
            fuzz_data_list.append(new_data)

    return fuzz_data_list


