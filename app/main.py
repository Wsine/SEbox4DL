import os
from importlib import import_module

import streamlit as st
from stqdm import stqdm
from app.context import create_context

def set_page_config(_):
    st.set_page_config(page_title='Ponyta', page_icon=':horse:')  # type: ignore


def set_task_config(ctx):
    tasks = ['none'] \
          + [f.name.rstrip('.py') for f in os.scandir(os.path.join('app', 'tasks')) \
             if not f.name.startswith('_')]

    with st.sidebar:
        st.write('## Task')
        ctx.task = st.sidebar.selectbox('which task to perform with this toolbox?', tasks)

    if ctx.task != 'none':
        task = import_module(f'app.tasks.{ctx.task}')
        st.sidebar.write('## Task Config')
        ctx.opt = task.load_sidebar(ctx)  # type: ignore

        _, _, col3 = st.sidebar.columns(3)
        col3.button('RUN', on_click=task.run, args=(ctx,))  # type: ignore


def set_ctx_helper(ctx):
    ctx.cache = st.cache
    ctx.tqdm = stqdm
    ctx.data_dir = os.environ.get('SEBOX4DL_DATA_DIR', 'data')
    ctx.output_dir = os.environ.get('SEBOX4DL_OUTPUT_DIR', 'output')


def main():
    with create_context('Ponyta') as ctx:
        set_page_config(ctx)
        set_task_config(ctx)
        set_ctx_helper(ctx)


if __name__ == '__main__':
    main()

