# -*- coding: utf-8 -*-
"""SEMGen - Distorters module.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import importlib
import logging

import utils

__all__ = []


class Distorter(object):
    """Abstract parent class for all the distorter subclasses.

    Subclasses become iterators operating on a queue.
    """

    def __init__(self):
        self.params = []
        self.params_current = None
        self._queue = []

    def _advance_queue(self):
        logging.debug("Distorting image (Queue: {0})".format(self._queue))
        if len(self.params) > 0:
            self.params_current = self.params.pop(0)
        else:
            logging.debug(
                "Image generation parameter list empty, autogenerating")
            self.params_current = self.generate_params()

    def __iter__(self):
        while len(self._queue) > 0:
            self._advance_queue()
            yield self.distort(utils.load_image(self._queue.pop(0)))

    def __next__(self):
        if len(self._queue) > 0:
            self._advance_queue()
            return self.distort(utils.load_image(self._queue.pop(0)))
        else:
            return

    def queue_images(self, list):
        self._queue = self._queue + list

    def queue_is_empty(self):
        if len(self._queue) > 0:
            return False
        else:
            return True

    def generate_params(self):
        return {}

    def distort(self, image):
        return

    def _getpx(self, im, x, y):
        if x < 0 or x >= im.shape[1] or y < 0 or y >= im.shape[0]:
            return
        return im[y, x]

    def _setpx(self, im, x, y, c):
        if x < 0 or x >= im.shape[1] or y < 0 or y >= im.shape[0]:
            return
        im[y, x] = c


def factory(mod, cls):
    mod = importlib.import_module('distorters.' + mod)
    return getattr(mod, cls)()
