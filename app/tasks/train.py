import streamlit as st
from argparse import Namespace
import torch
from app import sidebar
from app.context import st_stdout, st_stderr
from src.model import load_model
from src.dataset import load_dataset
from src.train import train_model
from torch.nn import CrossEntropyLoss, MSELoss

def load_sidebar(ctx):
    # opt: record params
    opt = Namespace()
    sidebar.load_datasets(opt)
    sidebar.load_models(opt)
    sidebar.load_train_configs(opt)
    print("train: ", opt)
    return opt

def run(ctx):
    """
    when press RUN
    :param ctx:
    :return:
    """
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
            _, trainloader = load_dataset(ctx.opt, split='train')
            _, validloader = load_dataset(ctx.opt, split='val')
    st.success(':balloon: dataset loaded.')
    # train
    optimizer = _init_optimizer(ctx.opt.optimizer, model)
    criterion = _init_criterion(ctx.opt.loss)
    # can add more kinds
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    with st.spinner(text='Training...'), st.expander('See process in detail'):
        with st_stdout('code'), st_stderr('code'):
            train_model(ctx, model, trainloader, validloader, optimizer, criterion, scheduler)
    st.balloons()

def _init_optimizer(optimizer, model):
    if optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters())
    elif optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.01)
    elif optimizer.lower() == "adagrad":
        return torch.optim.Adagrad(model.parameters())

def _init_criterion(criterion):
    if criterion.lower() == "cross-entropy":
        return CrossEntropyLoss()
    elif criterion.lower() == "mean squared error":
        return MSELoss()


