import torch

from utils import get_model_path


def load_model(opt):
    if "cifar" in opt.dataset:
        num_classes = (int)(opt.dataset.lstrip("cifar"))
        if opt.model == "resnet34":
            from models.cifar.resnet import ResNet34
            return ResNet34(num_classes=num_classes)
        elif opt.model == "dcalexnet":
            from models.cifar.dcalexnet import DcAlexNet
            return DcAlexNet(num_classes=num_classes)
        else:
            raise ValueError("Invalid model name")
    else:
        raise ValueError("Invalid dataset name")

def resume_model(opt, state="best"):
    model = load_model(opt)
    ckp = torch.load(
        get_model_path(opt, state=state),
        map_location=torch.device("cpu")
    )
    model.load_state_dict(ckp["net"])
    return model, ckp

