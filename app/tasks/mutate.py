from argparse import Namespace

import torch
import streamlit as st

from app import sidebar
from app.context import st_stdout, st_stderr
from src.model import load_model
from src.dataset import load_dataset
from src.runners.mutate import absent_mutate, pack_mutants


def load_sidebar(_):
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
            _, valloader = load_dataset(ctx, split='val')
    st.success(':balloon: dataset loaded.')

    with st.spinner(text='Running mutation...'):
        mutate_analysis = absent_mutate(ctx, model, valloader)
        mutants_path = pack_mutants(ctx, model, mutate_analysis)

    st.write('## Results')

    flatten_acc = [v for d in mutate_analysis.values() for v in d]
    flatten_acc = sorted(flatten_acc, reverse=True)
    st.write('**absent filter effects on accuracy**')
    st.bar_chart({'filters': flatten_acc})

    _, _, col3 = st.columns(3)
    with open(mutants_path, 'rb') as f:
        col3.download_button('Download mutants', data=f, file_name='mutants.zip', mime='application/zip')

    st.balloons()

