# -*- coding: utf-8 -*-
"""SEMGen - Constant background generator class.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import numpy as np

import generators
import utils


class ConstantGenerator(generators.Generator):
    """Generates images of a single shade of grey.

    Attributes:
        grey_limit (tuple): Limits the minimum and maximum grey level value
            in the generated images. Defaults to (0, 1).

    """

    def __init__(self):
        """Init function."""
        super().__init__()

        self.grey_limit = (0., 1.)

    def generate_params(self):
        return {}

    def generate(self):
        rn = np.random.uniform(self.grey_limit[0], self.grey_limit[1])
        f = np.ones(self.dim) * rn

        im = utils.feature_scale(f, 0, 255, 0., 1., 'uint8')
        return im
