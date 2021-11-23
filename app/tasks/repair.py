from argparse import Namespace

import torch
import streamlit as st

from app import sidebar
from app.context import st_stdout, st_stderr
from src.model import load_model
from src.dataset import load_dataset
from src.runners.eval import eval_accuracy
from src.runners.repair import repair_model


def load_sidebar(_):
    opt = Namespace()
    sidebar.load_datasets(opt, load_noise=True)
    sidebar.load_models(opt)
    sidebar.load_repair_options(opt)
    sidebar.load_train_options(opt)
    return opt


def run(ctx):
    st.write('# Task: Evaluate')

    st.write('## Configs')
    st.json(vars(ctx.opt))

    st.write('## Logs')

    ctx.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    with st.spinner(text='Loading model...'), st.expander('See loading process'):
        with st_stdout('code'), st_stderr('code'):
            model = load_model(ctx.opt)
    st.success(':balloon: model loaded.')

    with st.spinner(text='Loading dataset...'), st.expander('See loading process'):
        with st_stdout('code'), st_stderr('code'):
            noise = ctx.opt.add_noise != 'none'
            _, trainloader = load_dataset(ctx, split='train', noise=noise, noise_type='random')
            _, valloader = load_dataset(ctx, split='val', noise=noise, noise_type='expand')
    st.success(':balloon: dataset loaded.')

    with st.spinner(text='Repairing...'):
        repaired_model, model_path = repair_model(ctx, model, trainloader, valloader)

    _, clearloader = load_dataset(ctx, split='test', noise=False)
    std_acc = eval_accuracy(ctx, repaired_model, clearloader, desc='std_acc')
    _, noiseloader = load_dataset(ctx, split='test', noise=True, noise_type='append')
    rob_acc = eval_accuracy(ctx, repaired_model, noiseloader, desc='rob_acc')

    st.write('## Results')
    col1, col2 = st.columns(2)
    col1.metric('Std Accuracy', '{:.2f}%'.format(std_acc))
    col2.metric('Rob Accuracy', '{:.2f}%'.format(rob_acc), '{:.2f}%'.format(rob_acc - std_acc))

    _, _, col3 = st.columns(3)
    with open(model_path, 'rb') as f:
        col3.download_button('Download model', data=f, file_name='repaired_model.pth')

    st.balloons()

