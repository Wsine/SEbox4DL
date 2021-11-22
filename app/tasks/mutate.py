import random
from argparse import Namespace
import torch
import streamlit as st
from app import sidebar
from app.context import st_stdout, st_stderr
from src.model import load_model
from src.dataset import load_dataset
from src.mutate import absent_mutate
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_sidebar(ctx):
    opt = Namespace()
    sidebar.load_datasets(opt)
    sidebar.load_models(opt)
    sidebar.load_evaluate_configs(opt)
    return opt


def run(ctx):
    st.write('# Task: Mutate')
    st.write('## Configs')
    st.json(vars(ctx.opt))

    st.write('## Logs')

    ctx.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    with st.spinner(text='Loading model...'), st.expander('See loading process'):
        with st_stdout('code'), st_stderr('code'):
            model = load_model(ctx.opt).to(ctx.device)
    st.success(':balloon: model loaded.')

    with st.spinner(text='Loading dataset...'), st.expander('See loading process'):
        with st_stdout('code'), st_stderr('code'):
            _, testloader = load_dataset(ctx.opt, split='test')
    st.success(':balloon: dataset loaded.')
    buggy_filters, buggy_filters_layer, mutated_models = absent_mutate(ctx=ctx, model=model, data_loader=testloader)
    sorted_acc_diff, sorted_indices = torch.sort(torch.tensor(buggy_filters['acc_diff']), descending=True)

    st.write('## Results')
    st.write('### 1. Buggy filters test accuracy difference')
    acc_diff = sorted_acc_diff.tolist()
    plt.title('Test accuracy difference with filter absent')
    plt.plot(range(len(acc_diff)), acc_diff)
    plt.xlabel('Filter Index')
    plt.ylabel('Test Accuracy Difference')
    st.pyplot()

    st.write('### 2. Download mutated models')
    try:
        model_index_list = random.sample(range(0, len(mutated_models)), 2)
        for model_index in model_index_list:
            torch.save(mutated_models[model_index], "mutated_model_randomly_" + str(model_index) + ".pth")
            st.download_button(label="Download mutated model " + str(model_index),
                               data="mutated_model_randomly_" + str(model_index) + ".pth",
                               file_name="mutated_model_randomly_" + str(model_index) + ".pth")
    except Exception as excep:
        st.write('No mutated models can be downloaded! ' + str(excep))

    st.write('### 3. Buggy filters list')
    for buggy_filters_one_layer in buggy_filters_layer:
        if len(buggy_filters_one_layer['buggy_filter_index_layer']) > 0:
            st.write("Layer: " + buggy_filters_one_layer['layer'].strip('.weight') + " buggy filter index: " + str(buggy_filters_one_layer['buggy_filter_index_layer']))

    st.balloons()
