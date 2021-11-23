from argparse import Namespace

import torch
import streamlit as st

from app import sidebar
from app.context import st_stdout, st_stderr
from src.model import load_model
from src.runners.fuzz import fuzz_model


def load_sidebar(_):
    opt = Namespace()
    sidebar.load_models(opt)
    sidebar.load_fuzz_options(opt)
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

    with st.spinner(text='Fuzzing model...'), st.expander('See fuzzing process'):
        with st_stdout('code'), st_stderr('code'):
            fuzz_result = fuzz_model(ctx, model)

    st.write('## Results')
    if len(fuzz_result) > 0:
        with st.expander('See revealed inputs'):
            for t in fuzz_result:
                st.image(t)
    else:
        st.write('No input that revealing bugs found.')

    st.balloons()

