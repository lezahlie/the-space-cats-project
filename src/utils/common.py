"""
Utilities commonly used anywhere such as python module, helper functions, file I/O for hdf5, json, yaml, csv, etc
"""
from utils.logger import get_logger
import os
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
import warnings
import traceback
import gc
from pathlib import Path
from typing import Any, Literal, List, Tuple, Dict, Type, Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class GalaxiesMLDataset(pt.utils.data.Dataset):
    def __init__(self, hdf5_path, image_key="image", transform=None):
        self.hdf5_path = str(hdf5_path)
        self.image_key = image_key
        self.transform = transform
        self._file = None

        with h5py.File(self.hdf5_path, "r") as f:
            if self.image_key not in f:
                raise KeyError(f"Missing dataset key: {self.image_key}")
            self.length = f[self.image_key].shape[0]
            self.sample_shape = tuple(f[self.image_key].shape[1:])

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_file()
        x = f[self.image_key][idx]
        x = pt.tensor(x, dtype=pt.float32)

        if self.transform is not None:
            x = self.transform(x)

        return x