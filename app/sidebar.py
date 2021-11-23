import os

import streamlit as st
import torch
import torchvision

from app.context import sidebar_ctx


@sidebar_ctx
def load_datasets(opt):
    datasets = list(torchvision.datasets.__all__)
    opt.dataset = st.selectbox('which dataset to be used?', datasets)
    #  opt.dataset = st.selectbox('which dataset to be used?', datasets, index=7)
    if opt.dataset == 'ImageFolder':
        folders = [f.name for f in os.scandir('data') if f.is_dir()]
        opt.image_folder = st.selectbox('image folder to load:', folders)
    elif opt.dataset == 'DatasetFolder':
        folders = [f.name for f in os.scandir('data') if f.is_dir()]
        opt.dataset_folder = st.selectbox('dataset folder to load:', folders)

    opt.batch_size = st.number_input('how large is the batch size?', min_value=1, value=64)


@sidebar_ctx
def load_models(opt):
    opt.model_source_type = st.radio(
        'which model source to be used?',
        ['official', '3rdparty', 'local']
    )
    if opt.model_source_type == 'official':
        models_tasks = ['segmentation', 'detection', 'video', 'quantization']
        models = ['classification/' + m for m in torchvision.models.__dict__.keys() \
                  if not m.startswith('_') and not m[0].isupper() and \
                     m != 'utils' and not m in models_tasks]
        models += [f'{t}/' + m  \
                   for t in models_tasks  \
                   for m in eval(f'torchvision.models.{t}.__dict__.keys()') \
                   if not m.startswith('_') and not m[0].isupper() and  \
                      m != 'utils' and not m in models_tasks]
        opt.model = st.selectbox('which model to be tested?', models)
    elif opt.model_source_type == '3rdparty':
        opt.model_source = st.text_input(
            'where is the model source?',
            placeholder='example: chenyaofo/pytorch-cifar-models',
            help='refer to: https://pytorch.org/hub/'
        )
        if len(opt.model_source) > 0:
            models = torch.hub.list(opt.model_source)
            opt.model = st.selectbox('which model to be tested?', models)
    elif opt.model_source_type == 'local':
        # TODO: scan and add local models
        pass
    else:
        raise ValueError('Invalid input source')


@sidebar_ctx
def load_pretrained(opt, preset=True):
    opt.pretrained = st.checkbox('use (pre)trained weights?', preset)


@sidebar_ctx
def load_train_options(opt):
    optimizers = [o for o in torch.optim.__dict__.keys() if o[0].isupper()]
    opt.optimizer = st.selectbox('which optimizer to be used?', optimizers, index=optimizers.index('SGD'))

    criteria = [l for l in torch.nn.__dict__.keys() if l.endswith('Loss')]
    opt.criterion = st.selectbox('which loss function to be used?', criteria, index=criteria.index('CrossEntropyLoss'))

    opt.max_epoch = st.number_input('What is the maximum training epoch', min_value=1, value=20)
    opt.lr = st.number_input('What is the learning rate', value=0.01, format='%.4f')
    opt.momentum = st.number_input('What is the momentum', value=0.9, format='%.4f')
    opt.weight_decay = st.number_input('What is the weight decay', value=5e-4, format='%.4f')

