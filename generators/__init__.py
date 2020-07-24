# -*- coding: utf-8 -*-
"""SEMGen - Generators module.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import importlib
import logging

__all__ = []


class Generator(object):
    """Abstract parent class for all the generator subclasses.

    Subclasses become iterators operating on a queue.

    Attributes:
        dim (tuple): Dimensions of the generated image in (width, height)
            format.
    """

    def __init__(self):
        self.dim = (400, 400)
        self.params = []
        self.params_current = None
        self._queue = 0

    def _advance_queue(self):
        self._queue -= 1
        logging.debug("Generating image (Queue: {0})".format(self._queue))
        if len(self.params) > 0:
            self.params_current = self.params.pop(0)
        else:
            logging.debug(
                "Image generation parameter list empty, autogenerating")
            self.params_current = self.generate_params()

    def __iter__(self):
        while self._queue > 0:
            self._advance_queue()
            yield self.generate()

    def __next__(self):
        if self._queue > 0:
            self._advance_queue()
            return self.generate()
        else:
            return

    def __enter__(self):
        # dummy for the "with" keyword so progress bars can be disabled easily
        return self

    def __exit__(self, type, value, traceback):
        pass

    def queue_images(self, num):
        self._queue = self._queue + int(num)

    def queue_is_empty(self):
        if self._queue > 0:
            return False
        else:
            return True

    def generate_params(self):
        return {}

    def generate(self):
        raise NotImplementedError

    def _getpx(self, im, x, y):
        if x < 0 or x >= im.shape[1] or y < 0 or y >= im.shape[0]:
            return
        return im[y, x]

    def _setpx(self, im, x, y, c):
        if x < 0 or x >= im.shape[1] or y < 0 or y >= im.shape[0]:
            return
        im[y, x] = c

    def _slice(self, src, dst, x, y):
        """Safely slices two numpy arrays together.

        Truncates the source array as necessary to fit the destination array.
        X and Y coordinates can be negative numbers, and the source will be
        truncated accordingly.

        Args:
            src (ndarray): Source array to be sliced into the destination.
            dst (ndarray): Destination array.
            x (int): X-axis coordinate to paste the array at.
            y (int): y-axis coordinate to paste the array at.

        Returns:
            A numpy array the size of the destination array.

        """
        if (x < -src.shape[1] or y < -src.shape[0]
        or x > dst.shape[1] or y > dst.shape[0]):
            return dst

        if x < 0:
            src = src[:, x:]
            x = 0
        if y < 0:
            src = src[y:, :]
            y = 0
        if (x + src.shape[1]) > dst.shape[1]:
            d = (x + src.shape[1]) - dst.shape[1]
            src = src[:, :-d]
        if (y + src.shape[0]) > dst.shape[0]:
            d = (y + src.shape[0]) - dst.shape[0]
            src = src[:-d, :]

        dst[y:y + src.shape[0], x:x + src.shape[1]] = src
        return dst


def factory(mod, cls):
    mod = importlib.import_module('generators.' + mod)
    return getattr(mod, cls)()
