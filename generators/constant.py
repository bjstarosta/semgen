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
        rs = np.random.RandomState()
        p = {
            'range': rs.uniform(self.grey_limit[0], self.grey_limit[1])
        }
        return p

    def process(self, task):
        p = task.params
        f = np.ones(self.dim) * p['range']

        task.image = utils.feature_scale(f, 0, 255, 0., 1., 'uint8')
        return task
