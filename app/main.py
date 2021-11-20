import os
from importlib import import_module

import streamlit as st
from stqdm import stqdm
from app.context import create_context
# export PYTHONPATH="$(pwd)"
# todo: torun: streamlit run app/main.py --server.headless true

def set_page_config(ctx):
    st.set_page_config(page_title='Ponyta', page_icon=':horse:')  # type: ignore


def set_task_config(ctx):
    tasks = ['none'] \
          + [f.name.rstrip('.py') for f in os.scandir(os.path.join('app', 'tasks')) \
             if not f.name.startswith('_')]

    with st.sidebar:
        st.write('## Task')
        ctx.task = st.sidebar.selectbox('which task to perform with this toolbox?', tasks)
        #  ctx.task = st.sidebar.selectbox('which task to perform with this toolbox?', tasks, index=4)

    if ctx.task != 'none':
        task = import_module(f'app.tasks.{ctx.task}')
        st.sidebar.write('## Task Config')
        ctx.opt = task.load_sidebar(ctx)  # type: ignore

        _, _, col3 = st.sidebar.columns(3)
        col3.button('RUN', on_click=task.run, args=(ctx,))  # type: ignore


def set_ctx_helper(ctx):
    ctx.tqdm = stqdm


def main():
    with create_context('Ponyta') as ctx:
        set_page_config(ctx)
        set_task_config(ctx)
        set_ctx_helper(ctx)


if __name__ == '__main__':
    main()

