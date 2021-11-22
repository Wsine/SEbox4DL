import random
from argparse import Namespace
import torch
import streamlit as st
from app import sidebar
from app.context import st_stdout, st_stderr
from src.model import load_model
from src.fuzz import fuzz_numeric
import cv2
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_sidebar(ctx):
    opt = Namespace()
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

    fuzz_data = fuzz_numeric(ctx=ctx, model=model)

    st.write('## Results')
    st.write('### 1. Generated data')
    st.write('Get ' + str(len(fuzz_data)) + ' new images with NaN problem by fuzzing method from the randomly noisy data')

    st.write('### 2. Download generated data')
    try:
        data_index_list = random.sample(range(0, len(fuzz_data)), 2)
        for data_index in data_index_list:
            cv2.imwrite("generated_data_" + str(data_index) + ".png", fuzz_data[data_index])
            st.download_button(label="Download generated data " + str(data_index),
                               data="generated_data_" + str(data_index) + ".png",
                               file_name="generated_data_" + str(data_index) + ".png")
    except Exception as excep:
        st.write('No generated data can be downloaded! ' + str(excep))
