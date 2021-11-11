<img align="right" height="200" src="https://s1.52poke.wiki/wiki/thumb/f/f8/083Farfetch%27d.png/300px-083Farfetch%27d.png">

# DeepPatch

Project Code: Farfetch'd

For the technical work, please refer to the following publication.

## Publication

Under review

## Prerequisites

- Nvidia CUDA
- Python
- Pipenv

The project is maintained with [Pipenv](https://pipenv.pypa.io/en/latest/), and is highly recommended for a python project. Please refer to the link for more description and installation.

This project is tested under Ubuntu18.04, Python 3.6 and CUDA 11.

## Installation

It is easy and convenient to install all the same dependencies as the proposed with just one command.

```bash
pipenv sync
```

## How to run

The project contains four stages.

```mermaid
graph LR
pretrain --> assess --> correct --> evaluate
```

- Pretrain (optional): automatically download the model the dataset and evaluate their pretrained performance.
- Assess: prioritize the filters to be blamed
- Correct: correct the model with patching units
- Evaluate: evaluate the performance of patched model



Here, we take the resnet32 model and cifar10 dataset as an example.

The bootstrap commands are well organized with the help of pipenv.



You can list out and inspect the example commands with the below command.

```bash
~/workspace/deeppatch{main} > pipenv scripts
Command   Script
--------  ------------------------------------------------------------------------------------
pretrain  python src/eval.py    -m resnet32 -d cifar10
assess    python src/select.py  -m resnet32 -d cifar10 -f perfloss
correct   python src/correct.py -m resnet32 -d cifar10 -f perfloss -c patch --crt_type replace
evaluate  python src/switch.py  -m resent32 -d cifar10 -f perfloss -c patch --crt_type replace
```



To execute a single stage command, use `pipenv run pretrain` to bootstrap, and you may pass a flag `--help` to find and understand required and optional arguments.

```bash
~/workspace/deeppatch{main} > pipenv run assess --help
Loading .env environment variables...
usage: select.py [-h] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
                 [--device {cpu,cuda}] [--gpu {0,1,2,3}] [-b BATCH_SIZE] -m
                 {resnet32,mobilenetv2_x0_5,vgg13_bn,shufflenetv2_x1_0} [-r]
                 -d {cifar10,cifar100} [-n {gaussion}] [--lr LR]
                 [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
                 [-e MAX_EPOCH] -f {featswap,perfloss,ratioestim} -c
                 {patch,finetune} [--crt_type {crtunit,replace}]
                 [--crt_epoch CRT_EPOCH] [--susp_ratio SUSP_RATIO]
                 [--susp_side {front,rear,random}]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR
  --output_dir OUTPUT_DIR
  --device {cpu,cuda}
  --gpu {0,1,2,3}
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -m {resnet32,mobilenetv2_x0_5,vgg13_bn,shufflenetv2_x1_0}, --model {resnet32,mobilenetv2_x0_5,vgg13_bn,shufflenetv2_x1_0}
  -r, --resume
  -f {featswap,perfloss,ratioestim}, --fs_method {featswap,perfloss,ratioestim}
  -c {patch,finetune}, --crt_method {patch,finetune}
  --crt_type {crtunit,replace}
  --crt_epoch CRT_EPOCH
  --susp_ratio SUSP_RATIO
  --susp_side {front,rear,random}

dataset:
  -d {cifar10,cifar100}, --dataset {cifar10,cifar100}
  -n {gaussion}, --noise_type {gaussion}

optimizer:
  --lr LR               learning rate
  --momentum MOMENTUM
  --weight_decay WEIGHT_DECAY
  -e MAX_EPOCH, --max_epoch MAX_EPOCH
```

A progress bar will be shown and the results are logged under a default folder named `output`.
