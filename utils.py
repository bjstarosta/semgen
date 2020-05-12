import os
import re
import json
import click

import numpy as np
import skimage.external.tifffile as tifffile

import semgen


def find_filenames(dir, format=None):
    if format is None:
        format = semgen.IMG_PTRN

    ret = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if not f.endswith(semgen.IMG_EXT):
                continue
            d = string_to_dict(f, format)
            if d is None:
                continue
            ret.append(os.path.join(dir, f))

    return ret

def get_filenames(dir, n, format=None, overwrite=False):
    if format is None:
        format = semgen.IMG_PTRN

    format = format + semgen.IMG_EXT
    if overwrite == True:
        return _get_filenames(dir, n, 0, format)

    i = 0
    for root, dirs, files in os.walk(dir):
        for f in files:
            if not f.endswith(semgen.IMG_EXT):
                continue
            d = string_to_dict(f, format)
            j = int(d['0'])
            if j >= i:
                i = j + 1

    return _get_filenames(dir, n, i, format)

def _get_filenames(dir, n, i, format):
    ret = []
    for i in range(i, i+n):
        ret.append(os.path.join(dir, format.format(i)))
    return ret

def string_to_dict(string, pattern):
    regex = re.sub(r'{(.+?)(?:\:.+)}', r'(?P<_\1>.+)', pattern)
    values = re.search(regex, string)
    if values is None:
        return None
    values = list(values.groups())
    keys = re.findall(r'{(.+?)(?:\:.+)}', pattern)
    return dict(zip(keys, values))

def read_params(dir):
    with click.open_file(os.path.join(dir, semgen.PARAM_FILE), 'r') as f:
        data = json.load(f)
    return data

def write_params(dir, data):
    data['_version'] = semgen.VERSION
    with click.open_file(os.path.join(dir, semgen.PARAM_FILE), 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)

def feature_scale(img, a, b, min=None, max=None, type=None):
    """Min-max normalises a 2-dim array containing image data.

    See: https://en.wikipedia.org/wiki/Feature_scaling

    Args:
        img (ndarray): Array containing image data.
        a (float): Minimum value of the result.
        b (float): Maximum value of the result.
        min (float): Minimum allowed value of the input.
        max (float): Maximum allowed value of the input.
        type (object): Convert output to this value type. E.g. np.uint8

    Returns:
        ndarray: Array containing normalised image data.

    """
    img = np.asarray(img, dtype=np.float16)
    if min is None:
        min = np.min(img)
    if max is None:
        max = np.max(img)
    img = a + ((img - min) * (b - a)) / (max - min)
    if type is not None:
        img = img.astype(type)
    return img

def load_image(path):
    with tifffile.TiffFile(path) as tif:
        return tif.asarray()

def save_image(path, img):
    tifffile.imsave(path, img)
