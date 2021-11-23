import time

import torch

from src.utils import AttrDispatcher


fuzz_dispatcher = AttrDispatcher('fuzzer_runner')
input_indicator = False


@fuzz_dispatcher.register('tensorfuzz')
def tensorfuzz_method(_, model):

    def _hook_fn(module, finput, foutput):
        if foutput.isnan().any() or foutput.isinf().any():
            global input_indicator
            input_indicator = True

    for m in model.modules():
        m.register_forward_hook(_hook_fn)

    return model


def fuzz_model(ctx, model):
    model = fuzz_dispatcher(ctx.opt, model)

    fuzz_result = []

    elapsed_mins = 0
    start_time = time.time()
    while True:
        elapsed = (time.time() - start_time) // 60
        if elapsed > elapsed_mins:
            print('elapsed {} mins...'.format(elapsed))
            elapsed_mins = elapsed
        if elapsed > ctx.opt.timeout:
            break

        global input_indicator
        input_indicator = False
        model_input = torch.rand((1, 3, 32, 32)).to(ctx.device)
        model(model_input)
        if input_indicator is True:
            fuzz_result.append(model_input[0].cpu().numpy())

    return fuzz_result

