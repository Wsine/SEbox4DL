import os
from importlib import import_module

import streamlit as st


ICON_URL = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/285/scientist_1f9d1-200d-1f52c.png"
st.set_page_config(page_title='Ponyta', page_icon=ICON_URL)  # type: ignore

tasks = [f.name for f in os.scandir('tasks') if f.is_dir()]

with st.sidebar:
    st.info('🎈 **NEW:** Select your task first before config the task')
    st.write('## Task')
    input_task = st.sidebar.selectbox('The task to perform with this toolbox?', tasks)

task = import_module(f'tasks.{input_task}.main')
task.run()  # type: ignore

