def load_model(opt):
    if "cifar" in opt.dataset:
        if opt.model == "resnet34":
            from models.cifar.resnet import ResNet34
            return ResNet34()
        elif opt.model == "resnet50":
            from models.cifar.resnet import ResNet50
            return ResNet50()
        else:
            raise ValueError("Invalid model name")
    else:
        raise ValueError("Invalid dataset name")

