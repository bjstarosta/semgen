# -*- coding: utf-8 -*-
"""SEMGen - Distorters module.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import importlib
import inspect
import processor

__all__ = []


class Distorter(processor.Processor):
    pass


def factory(mod):
    """Import the last defined class from the specified module.

    Args:
        mod (str): Distorter module file name. Must be part of the 'distorters'
            module.

    Returns:
        distorters.Distorter: Derived class.

    """
    mod = importlib.import_module('distorters.' + mod)
    cls = inspect.getmembers(mod, inspect.isclass)[-1][0]
    return getattr(mod, cls)()
