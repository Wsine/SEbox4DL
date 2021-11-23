from argparse import Namespace

import torch
import streamlit as st

from app import sidebar
from app.context import st_stdout, st_stderr
from src.model import load_model
from src.dataset import load_dataset
from src.runners.train import train_model


def load_sidebar(_):
    opt = Namespace()
    sidebar.load_datasets(opt, load_noise=True)
    sidebar.load_models(opt)
    sidebar.load_train_options(opt)
    return opt


def run(ctx):
    st.write('# Task: Train')

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
            noise = ctx.opt.add_noise != 'none'
            _, trainloader = load_dataset(ctx, split='train', noise=noise, noise_type='random')
            _, valloader = load_dataset(ctx, split='val', noise=noise, noise_type='random')
    st.success(':balloon: dataset loaded.')

    with st.spinner(text='Training...'):
        acc, loss = train_model(ctx, model, trainloader, valloader, is_finetune=True)

    st.write('## Results')
    col1, col2 = st.columns(2)
    col1.metric('Accuracy', '{:.2f}%'.format(acc))
    col2.metric('Loss', '{:.4f}'.format(loss))

    st.balloons()

