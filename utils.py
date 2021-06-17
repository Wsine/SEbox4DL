import os
import errno
import json
import pickle
import functools
import hashlib


def get_model_path(opt, state="best"):
    path = os.path.join(
        opt.output_dir, opt.dataset, opt.model, f"model_{state}.pth")
    return path


def guard_folder(opt, folder=None):
    if not folder:
        folder = []
    elif isinstance(folder, str):
        folder = [folder]
    folder.append(os.path.join(opt.output_dir, opt.dataset, opt.model))
    for f in folder:
        if not os.path.isdir(f):
            os.makedirs(f)


def dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def cache_object(filename):
    def _decorator(func):
        def __func_wrapper(*args, **kwargs):
            cache_dir = args[0].cache_dir if hasattr(args[0], 'cache_dir') else 'cache'
            filepath = os.path.join(cache_dir, filename)
            try:
                cache = pickle.load(open(filepath, 'rb'))
            except IOError:
                cache = {}
            paramskey = dict_hash(kwargs)
            if paramskey not in cache:
                cache[paramskey] = func(*args, **kwargs)
                if not os.path.isdir(cache_dir):
                    os.makedirs(cache_dir)
                pickle.dump(cache, open(filepath, 'wb'))
            else:
                print('[info] {} cache hit'.format(func.__name__))
            return cache[paramskey]
        return __func_wrapper
    return _decorator


def preview_object(obj):
    def _reduce_object(obj):
        if isinstance(obj, list):
            return [_reduce_object(o) for o in obj[:2]]
        elif isinstance(obj, dict):
            return {k: _reduce_object(v) for k, v in obj.items()}
        else:
            return obj

    print(json.dumps(_reduce_object(obj), indent=2, ensure_ascii=False))


def export_object(opt, filename, code, obj):
    mode = 'b' if filename.endswith('.pkl') else ''
    dstdir = os.path.join(opt.output_dir, opt.dataset, opt.model)

    basename, ext = os.path.splitext(filename)
    filepath = os.path.join(dstdir, f"{basename}_{code}{ext}")
    with open(filepath, f'w{mode}') as f:
        if mode == 'b':
            pickle.dump(obj, f)
        else:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    sympath = os.path.join(dstdir, f"{filename}")
    try:
        os.symlink(f"{basename}_{code}{ext}", sympath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(sympath)
            os.symlink(f"{basename}_{code}{ext}", sympath)
        else:
            raise e


# borrow from: https://stackoverflow.com/questions/31174295/
#   getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

