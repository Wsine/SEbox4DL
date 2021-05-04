import os


def get_model_path(opt):
    path = os.path.join(opt.output_dir, f"{opt.dataset}_{opt.model}.pth")
    return path

def guard_folder(folder):
    if isinstance(folder, str):
        folder = [folder]
    for f in folder:
        if not os.path.isdir(f):
            os.makedirs(f)

