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
import tarfile
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Literal, List, Tuple, Dict, Type, Optional, Union
from collections import OrderedDict


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==================================================
# CONTRIBUTION START: Lazy loader for GalaxiesMLDataset,
# I/O helpers for json, HDF5StackWriter
# Contributor: Leslie Horace
# ==================================================


def _detach_tensor(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, pt.Tensor):
        return value.detach().cpu().numpy()
    arr = np.asarray(value)
    return arr


def _normalize_sample_array(arr, batched):
    if batched:
        if arr.ndim == 0:
            return arr.reshape(1)
        return arr

    if arr.ndim == 0:
        return arr.reshape(1)
    return arr[None, ...]


class HDF5StackWriter:
    def __init__(
        self,
        hdf5_path,
        chunk_rows=64,
        compression=None,
        compression_opts=None,
        overwrite=False,
        flush_every=0,
    ):
        mode = "w" if overwrite else "a"
        self.file = h5py.File(hdf5_path, mode)
        self.chunk_rows = chunk_rows
        self.flush_every = flush_every
        self.append_count = 0
        self.compression = compression
        self.compression_opts = compression_opts

        self.datasets = {key: self.file[key] for key in self.file.keys()}

    def _serialize_batch(self, batch_dict, batched=True):
        out = {}
        batch_size = None

        for key, value in batch_dict.items():
            arr = _detach_tensor(value)
            arr = _normalize_sample_array(arr, batched=batched)

            if arr.dtype.kind in {"U", "S", "O"}:
                try:
                    arr = arr.astype(np.int64)
                except (ValueError, TypeError):
                    try:
                        arr = arr.astype(np.float32)
                    except (ValueError, TypeError):
                        raise TypeError(f"{key} has non-numeric dtype {arr.dtype}")

            if batch_size is None:
                batch_size = arr.shape[0]
            elif arr.shape[0] != batch_size:
                raise ValueError("All arrays must have same batch size")

            out[key] = arr

        return out

    def _create_dataset(self, key, arr):
        if arr.ndim == 0:
            raise ValueError(f"{key} must have a batch dimension")

        rows = max(1, arr.shape[0])
        chunk_rows = min(self.chunk_rows, rows)

        dset = self.file.create_dataset(
            key,
            data=arr,
            maxshape=(None,) + arr.shape[1:],
            chunks=(chunk_rows,) + arr.shape[1:],
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        self.datasets[key] = dset
        
    def append(self, batch_dict, batched=True):
        batch_dict = self._serialize_batch(batch_dict, batched=batched)

        if not self.datasets:
            for key, arr in batch_dict.items():
                self._create_dataset(key, arr)
        else:
            if set(batch_dict.keys()) != set(self.datasets.keys()):
                raise ValueError("Batch keys do not match existing datasets")

        for key, arr in batch_dict.items():
            dset = self.datasets[key]

            if dset.shape[1:] != arr.shape[1:]:
                raise ValueError(
                    f"Shape mismatch for {key}: got {arr.shape[1:]}, expected {dset.shape[1:]}"
                )

            start = dset.shape[0]
            stop = start + arr.shape[0]
            dset.resize((stop,) + dset.shape[1:])
            dset[start:stop] = arr

        self.append_count += 1
        if self.flush_every > 0 and self.append_count % self.flush_every == 0:
            self.file.flush()


    def close(self):
        if self.file is None:
            return
        self.file.flush()
        self.file.close()
        self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def dump_hdf5_structure(path: Path) -> Path:
    out_txt = path.with_suffix(path.suffix + ".structure.txt")

    def write_attrs(obj, fh, indent):
        if len(obj.attrs) == 0:
            return
        fh.write(f"{indent}@attrs\n")
        for k, v in obj.attrs.items():
            fh.write(f"{indent}  - {k}: {repr(v)}\n")

    with h5py.File(path, "r") as f, open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(f"FILENAME: {path.name}\n\n")

        def recurse(name, obj, depth=0):
            indent = "  " * depth

            if isinstance(obj, h5py.Group):
                fh.write(f"{indent}[GROUP] {name or '/'}\n")
                write_attrs(obj, fh, indent + "  ")

                for key in obj.keys():
                    child = obj[key]
                    child_name = f"{name}/{key}" if name else key
                    recurse(child_name, child, depth + 1)

            elif isinstance(obj, h5py.Dataset):
                fh.write(
                    f"{indent}[DATASET] {name} | "
                    f"shape={obj.shape} | dtype={obj.dtype}\n"
                )
                write_attrs(obj, fh, indent + "  ")

        recurse("", f, 0)

    return out_txt

def get_versioned_backup_path(path):
    path = Path(path)

    if path.name.endswith(".tar.gz"):
        base_name = path.name[:-7]
        suffix = ".tar.gz"
    else:
        base_name = path.stem
        suffix = path.suffix

    version = 1
    while True:
        backup_path = path.with_name(f"{base_name}.v{version}{suffix}")
        if not backup_path.exists():
            return backup_path
        version += 1


def make_tar_gz(src_dir, tar_path):
    src_dir = Path(src_dir)
    tar_path = Path(tar_path)

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    tar_path.parent.mkdir(parents=True, exist_ok=True)

    if tar_path.exists():
        backup_path = get_versioned_backup_path(tar_path)
        tar_path.rename(backup_path)

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)

    return tar_path


class GalaxiesMLDataset(pt.utils.data.Dataset):
    def __init__(
        self,
        hdf5_path,
        data_keys=None,
        max_samples=None,
        return_dict=True,
    ):
        self.hdf5_path = str(hdf5_path)
        self.max_samples = max_samples
        self.return_dict = bool(return_dict)
        self._file = None

        with h5py.File(self.hdf5_path, "r") as f:
            if data_keys is None:
                data_keys = list(f.keys())
            elif isinstance(data_keys, str):
                data_keys = [data_keys]

            if isinstance(data_keys, dict):
                # output_name -> hdf5_key
                self.key_map = dict(data_keys)
            else:
                # output_name == hdf5_key
                data_keys = list(data_keys)
                self.key_map = {key: key for key in data_keys}

            if len(self.key_map) < 1:
                raise ValueError("data_keys must contain at least one HDF5 dataset key")

            self.output_keys = list(self.key_map.keys())
            self.data_keys = list(self.key_map.values())

            missing = [key for key in self.data_keys if key not in f]
            if missing:
                raise KeyError(
                    f"Missing dataset keys in {self.hdf5_path}: {missing}\n"
                    f"Available keys: {list(f.keys())}"
                )

            self.length = f[self.data_keys[0]].shape[0]

            for key in self.data_keys:
                if f[key].shape[0] != self.length:
                    raise ValueError(
                        f"Length mismatch for key {key}: "
                        f"got {f[key].shape[0]}, expected {self.length}"
                    )

            if self.max_samples is not None:
                self.length = min(self.length, int(self.max_samples))

            self.sample_shapes = {
                output_key: tuple(f[hdf5_key].shape[1:])
                for output_key, hdf5_key in self.key_map.items()
            }

            self.sample_shape = self.sample_shapes[self.output_keys[0]]

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self):
        return self.length

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def _to_tensor_or_value(self, value):
        arr = np.asarray(value)

        if arr.shape == ():
            item = arr.item()

            if hasattr(item, "decode"):
                return item.decode("utf-8")

            if np.issubdtype(arr.dtype, np.integer):
                return pt.tensor(item, dtype=pt.long)

            if np.issubdtype(arr.dtype, np.floating):
                return pt.tensor(item, dtype=pt.float32)

            return item

        if arr.dtype.kind in {"S", "U", "O"}:
            decoded = []
            for item in arr:
                if hasattr(item, "decode"):
                    decoded.append(item.decode("utf-8"))
                else:
                    decoded.append(item)
            return decoded

        tensor = pt.from_numpy(arr)

        if pt.is_floating_point(tensor):
            return tensor.to(pt.float32)

        return tensor

    def __getitem__(self, idx):
        f = self._get_file()

        sample = {
            output_key: self._to_tensor_or_value(f[hdf5_key][idx])
            for output_key, hdf5_key in self.key_map.items()
        }

        if self.return_dict:
            return AttrDict(sample)

        return tuple(sample[key] for key in self.output_keys)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass



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

def save_to_yaml(file_path, content):
    try:
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file, default_flow_style=False, sort_keys=False)
    except Exception as e:
        get_logger().error(f"Error saving to YAML file: {file_path}")
        raise e

def load_from_yaml(file_path):
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        get_logger().error(f"Error reading YAML file: {file_path}")
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