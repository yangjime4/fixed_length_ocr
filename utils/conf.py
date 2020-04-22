from yaml import safe_load
import os


def get(key):
    with open("../config/config.yaml", 'rb') as f:
        cont = f.read()
    cf = safe_load(cont)
    return cf.get(key)
