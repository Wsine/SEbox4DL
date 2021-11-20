from argparse import Namespace
import torch
import streamlit as st
from app import sidebar
from app.context import st_stdout, st_stderr
from src.model import load_model
from src.dataset import load_dataset
from src.mutate import absent_mutate
import numpy as np
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_sidebar(ctx):
    opt = Namespace()
    sidebar.load_datasets(opt)
    sidebar.load_models(opt)
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
    buggy_filters, buggy_filters_layer = absent_mutate(ctx=ctx, model=model, data_loader=testloader)
    sorted_acc_diff, sorted_indices = torch.sort(torch.tensor(buggy_filters['acc_diff']), descending=True)

    st.write('## Results')
    st.write('### 1. Buggy filters test accuracy difference')
    acc_diff = np.array(sorted_acc_diff.tolist())
    plt.title('Test accuracy difference with filter absent')
    plt.plot(range(len(acc_diff)), acc_diff)
    plt.xlabel('Filter Index')
    plt.ylabel('Test Accuracy Difference')
    st.pyplot()

    st.write('### 2. Buggy filters list')
    for buggy_filters_one_layer in buggy_filters_layer:
        st.write("Layer: " + buggy_filters_one_layer['layer'] + " buggy filter index: " + str(buggy_filters_one_layer['buggy_filter_index_layer']))

    st.balloons()
