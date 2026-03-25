# config.py
import yaml
from copy import deepcopy

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def merge_args_into_cfg(cfg, args):
    cfg = deepcopy(cfg)
    for k, v in vars(args).items():
        if k == "config":
            continue
        if v is not None:
            cfg[k] = v
    return cfg