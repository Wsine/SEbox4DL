import sys
from io import StringIO
from threading import current_thread
from contextlib import contextmanager

import streamlit as st
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME


class Context(object):
    def __init__(self, name):
        self.ctx_name = name


@contextmanager
def create_context(name):
    ctx = Context(name)
    yield ctx


def sidebar_ctx(func):
    def wrap(*args, **kwargs):
        with st.sidebar:
            func(*args, **kwargs)
    return wrap


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield

