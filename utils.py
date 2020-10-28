# -*- coding: utf-8 -*-
"""SEMGen - Utility functions.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import re

import numpy as np
import skimage.transform as transform

try:
    import skimage.external.tifffile as tifffile
except ImportError:
    import tifffile


IMG_PTRN = "semgen-{0:04d}"
IMG_EXT = '.tif'


def find_filenames(dir, format=None):
    """Find filenames within a directory according to a particular format.

    This is done non-recursively and relies on reversing the string.format()
    operation on filenames.

    Args:
        dir (str): Directory to search through.
        format (str): string.format() combatible format definition according
            to which the filenames should be validated.

    Returns:
        list of str: List of filenames present in the passed directory
            conforming to the format definition.

    """
    if format is None:
        format = IMG_PTRN

    ret = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if not f.endswith(IMG_EXT):
                continue
            d = string_to_dict(f, format)
            if d is None:
                continue
            ret.append(os.path.join(dir, f))

    return ret


def remap_filenames(file_list, new_path):
    """Replace the path element of a passed list of file paths.

    This does not carry out any copying on the filesystem level, it simply
    transforms a passed list of paths to files. The original path is stripped
    out without being matched against anything.

    Args:
        file_list (list of str): List of file paths.
        new_path (str): Replacement path for all files in the list.

    Returns:
        list of str: List of file paths amended to the new path.

    """
    ret = []
    for f in file_list:
        ret.append(os.path.join(new_path, os.path.split(f)[1]))
    return ret


def get_filenames(dir, n, format=None, overwrite=False):
    """Return a list of valid filenames to use when saving data.

    Takes into account files already present in a directory while generating
    filenames by finding any files that conform to the passed format with
    the highest generated offset.

    Args:
        dir (str): Target directory.
        n (int): Amount of filenames to generate.
        format (type): string.format() combatible format definition according
            to which the filenames should be generated.
        overwrite (bool): If set to True it will not scan the target
            directory to determine the starting offset.

    Returns:
        list of str: List of file paths.

    """
    if format is None:
        format = IMG_PTRN

    format = format + IMG_EXT
    if overwrite is True:
        return _get_filenames(dir, n, 0, format)

    i = 0
    for root, dirs, files in os.walk(dir):
        for f in files:
            if not f.endswith(IMG_EXT):
                continue
            d = string_to_dict(f, format)
            j = int(d['0'])
            if j >= i:
                i = j + 1

    return _get_filenames(dir, n, i, format)


def _get_filenames(dir, n, i, format):
    ret = []
    for i in range(i, i + n):
        ret.append(os.path.join(dir, format.format(i)))
    return ret


def string_to_dict(string, pattern):
    """Use a string.format() pattern to extract variables from a string.

    This function performs the inverse operation to string.format().
    It takes a compatible pattern and uses it as a guide to extract
    data from a passed string.

    Args:
        string (str): String to extract data from.
        pattern (str): string.format() combatible format definition according
            to which the data should be extracted from the string.

    Returns:
        dict: Extracted string data in dictionary form.

    """
    regex = re.sub(r'{(.+?)(?:\:.+)}', r'(?P<_\1>.+)', pattern)
    values = re.search(regex, string)
    if values is None:
        return None
    values = list(values.groups())
    keys = re.findall(r'{(.+?)(?:\:.+)}', pattern)
    return dict(zip(keys, values))


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
    img = np.asarray(img, dtype=np.float64)
    if min is None:
        min = np.min(img)
    if max is None:
        max = np.max(img)
    img = a + ((img - min) * (b - a)) / (max - min)
    if type is not None:
        img = img.astype(type)
    return img


def resize_image(im, dim):
    """Resize an image in numpy array format.

    Args:
        im (numpy.ndarray): Image data.
        dim (tuple of int): New image dimensions in the (width, height) format.

    Returns:
        numpy.ndarray: Resized image data.

    """
    return transform.resize(
        im, (dim[1], dim[0]), preserve_range=True).astype(np.uint8)


def load_image(path):
    """Read image from the passed path and return it in numpy array form.

    Args:
        path (str): Path to image file.

    Returns:
        numpy.ndarray: Image data.

    """
    with tifffile.TiffFile(path) as tif:
        return tif.asarray()


def save_image(path, img):
    """Save numpy array data as an image file in the passed path.

    Args:
        path (str): Path to image file.
        img (numpy.ndarray): Image data.

    """
    tifffile.imsave(path, img)
