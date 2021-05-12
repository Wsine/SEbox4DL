import os
import json
import pickle


def get_model_path(opt):
    path = os.path.join(opt.output_dir, f"{opt.dataset}_{opt.model}.pth")
    return path


def guard_folder(folder):
    if isinstance(folder, str):
        folder = [folder]
    for f in folder:
        if not os.path.isdir(f):
            os.makedirs(f)


def cache_object(opt, filename, func, *args, **kwargs):
    mode = 'b' if filename.endswith('.pkl') else ''
    filepath = os.path.join(opt.cache_dir, f"{opt.dataset}_{opt.model}_{filename}")
    if os.path.exists(filepath):
        with open(filepath, f'r{mode}') as f:
            obj = pickle.load(f) if mode == 'b' else json.load(f)
    else:
        obj = func(*args, **kwargs)
        with open(filepath, f'w{mode}') as f:
            if mode == 'b':
                pickle.dump(obj, f)
            else:
                json.dump(obj, f, indent=2, ensure_ascii=False)
    return obj


def preview_object(obj):
    def _reduce_object(obj):
        if isinstance(obj, list):
            return [_reduce_object(o) for o in obj[:2]]
        elif isinstance(obj, dict):
            return {k: _reduce_object(v) for k, v in obj.items()}
        else:
            return obj

    print(json.dumps(_reduce_object(obj), indent=2, ensure_ascii=False))


def export_object(opt, filename, obj):
    mode = 'b' if filename.endswith('.pkl') else ''
    filepath = os.path.join(opt.output_dir, f"{opt.dataset}_{opt.model}_{filename}")
    with open(filepath, f'w{mode}') as f:
        if mode == 'b':
            pickle.dump(obj, f)
        else:
            json.dump(obj, f, indent=2, ensure_ascii=False)

