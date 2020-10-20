# -*- coding: utf-8 -*-
"""SEMGen - Generators module.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import importlib
import inspect
import processor

__all__ = []


class Generator(processor.Processor):
    pass


def factory(mod):
    """Import the last defined class from the specified module.

    Args:
        mod (str): Generator module file name. Must be part of the 'generators'
            module.

    Returns:
        generators.Generator: Derived class.

    """
    mod = importlib.import_module('generators.' + mod)
    cls = inspect.getmembers(mod, inspect.isclass)[-1][0]
    return getattr(mod, cls)()
