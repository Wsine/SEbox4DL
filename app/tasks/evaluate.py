from argparse import Namespace

import torch
import streamlit as st

from app import sidebar
from app.context import st_stdout, st_stderr
from src.model import load_model
from src.dataset import load_dataset
from src.eval import eval_accuracy


def load_sidebar(ctx):
    opt = Namespace()
    sidebar.load_datasets(opt)
    sidebar.load_models(opt)
    return opt


def run(ctx):
    st.write('# Task: Evaluate')

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
    std_acc = eval_accuracy(ctx, model, testloader, desc='std_acc')

    _, noiseloader = load_dataset(ctx.opt, split='test', noise=True, noise_type='append')
    rob_acc = eval_accuracy(ctx, model, noiseloader, desc='rob_acc')


    st.write('## Results')
    col1, col2 = st.columns(2)
    col1.metric('Std Accuracy', '{:.2f}%'.format(std_acc))
    col2.metric('Rob Accuracy', '{:.2f}%'.format(rob_acc), '{:.2f}%'.format(rob_acc - std_acc))

    st.balloons()

