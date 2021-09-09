# DeepPatch

## Publication

Under review



## Installation

The project is maintained with [Pipenv](https://pipenv.pypa.io/en/latest/). Please refer to the link for installing Pipenv.

The dependencies are very convenient to install by one command. The versions are same as proposed here.

```bash
pipenv sync
```



## How to run

The executions are well organized with the help of Pipenv.

```
~/workspace/deeppatch{main} > pipenv scripts
Command   Script
--------  ------------------------------------------------------------------------------------
notebook  jupyter notebook --config=.notebook_config.py
pretrain  python src/eval.py    -m resnet32 -d cifar10
assess    python src/select.py  -m resnet32 -d cifar10 -f perfloss
correct   python src/correct.py -m resnet32 -d cifar10 -f perfloss -c patch --crt_type replace
evaluate  python src/switch.py  -m resent32 -d cifar10 -f perfloss -c patch --crt_type replace
```

