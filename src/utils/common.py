"""
Utilities commonly used anywhere such as python module, helper functions, file I/O for hdf5, json, yaml, csv, etc
"""
from src.utils.logger import get_logger
import os
import sys
import argparse
import glob
import re
import pprint
import random
import json 
import yaml
import h5py
import scipy
import torch as pt
import numpy as np
import pandas as pd
import warnings
import traceback
import gc
import copy 
import time
from pathlib import Path
from typing import Any, Literal, List, Tuple, Dict, Type, Optional, Union



warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==================================================
# CONTRIBUTION START: Lazy loader for GalaxiesMLDataset,
# I/O helpers for json
# Contributor: Leslie Horace
# ==================================================

class GalaxiesMLDataset(pt.utils.data.Dataset):
    def __init__(
        self,
        hdf5_path,
        input_key="image",
        target_key=None,
        transform=None,
        target_transform=None,
        max_samples=None,
    ):
        self.hdf5_path = str(hdf5_path)
        self.input_key = input_key
        self.target_key = target_key
        self.transform = transform
        self.target_transform = target_transform
        self.max_samples = max_samples
        self._file = None

        with h5py.File(self.hdf5_path, "r") as f:
            if self.input_key not in f:
                raise KeyError(f"Missing dataset key: {self.input_key}")

            self.length = f[self.input_key].shape[0]

            if self.target_key is not None:
                if isinstance(self.target_key, (list, tuple)):
                    for key in self.target_key:
                        if key not in f:
                            raise KeyError(f"Missing dataset key: {key}")
                        if f[key].shape[0] != self.length:
                            raise ValueError(f"Length mismatch for target key: {key}")
                else:
                    if self.target_key not in f:
                        raise KeyError(f"Missing dataset key: {self.target_key}")
                    if f[self.target_key].shape[0] != self.length:
                        raise ValueError(f"Length mismatch for target key: {self.target_key}")

            if self.max_samples is not None:
                self.length = min(self.length, self.max_samples)

            self.sample_shape = tuple(f[self.input_key].shape[1:])

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self):
        return self.length

    def _to_input_tensor(self, value):
        arr = np.asarray(value)
        if arr.shape == ():
            return pt.tensor(arr.item(), dtype=pt.float32)
        return pt.from_numpy(arr).to(pt.float32)

    def _to_target_tensor(self, value):
        arr = np.asarray(value)

        if arr.shape == ():
            if np.issubdtype(arr.dtype, np.integer):
                return pt.tensor(arr.item(), dtype=pt.long)
            if np.issubdtype(arr.dtype, np.floating):
                return pt.tensor(arr.item(), dtype=pt.float32)
            return pt.tensor(arr.item())

        t = pt.from_numpy(arr)
        if pt.is_floating_point(t):
            return t.to(pt.float32)
        return t

    def __getitem__(self, idx):
        f = self._get_file()

        x = self._to_input_tensor(f[self.input_key][idx])
        if self.transform is not None:
            x = self.transform(x)

        if self.target_key is None:
            y = x
        elif isinstance(self.target_key, (list, tuple)):
            y = {key: self._to_target_tensor(f[key][idx]) for key in self.target_key}
            if self.target_transform is not None:
                y = self.target_transform(y)
        else:
            y = self._to_target_tensor(f[self.target_key][idx])
            if self.target_transform is not None:
                y = self.target_transform(y)

        return x, y

class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            return cls({k: cls._wrap(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            self[k] = self._wrap(v)



def format_json(input_dict):
    def _convert(v):
        if isinstance(v, dict):
            return {k: _convert(val) for k, val in v.items()}

        if isinstance(v, (list, tuple)):
            return [_convert(x) for x in v]

        if isinstance(v, pt.Tensor):
            return v.detach().cpu().tolist()

        if isinstance(v, np.ndarray):
            return v.tolist()

        if isinstance(v, np.generic):
            return v.item()

        if v is None or isinstance(v, (str, int, float, bool)):
            return v

        return str(v)

    return _convert(input_dict)

def save_to_json(file_path, content, mode='w', indent=4, sort_keys=False):
    try:
        if isinstance(content, dict):
            content = format_json(content)
        elif isinstance(content, list):
            content = [format_json(item) if isinstance(item, dict) else item for item in content]
        with open(file_path, mode) as json_file:
            json.dump(content, json_file, indent=indent, sort_keys=sort_keys)
    except Exception as e:
        get_logger().error(f"Error saving to JSON file: {file_path}")
        raise e

def read_from_json(file_path):
    try:
        with open(file_path, 'r') as json_file:
            content = json.load(json_file)
        return content
    except Exception as e:
        get_logger().error(f"Error reading to JSON file: {file_path}")
        raise e

def tensor_to_image(x):
    if isinstance(x, pt.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x)

def validate_tensor(name, x):
    if x is None:
        raise ValueError(f"{name} is None")
    if not pt.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if not pt.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    

# ==================================================
# CONTRIBUTION End: Lazy loader for GalaxiesMLDataset,
# json I/O, plotting tools, helpers
# ==================================================