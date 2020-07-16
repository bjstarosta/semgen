import logging
import numpy as np

import generators
import utils


class GradientGenerator(generators.Generator):
    """Generates images of linear and radial gradients.

    Attributes:
        types (list): List of gradient types to generate. Allowed values:
            linear, radial
        types_pr (list): List of floats that sets the probability distribution
            of gradient type choice when picked randomly. Assigned in the same
            order as the 'types' list. Must sum to 1 amd have the same size as
            'types'. Defaults to uniform distribution.
        random_pos (bool): If set to false all gradients will be generated
            at the origin position (0,0) and with the default direction.
        grey_range (float): Percentage of the range of greys to be used on a
            single image. If set to less than the difference between max
            and min grey limit, the start of the range will be generated
            randomly, creating a variation in the grey range in all generated
            images.
        grey_limit (tuple): Limits the minimum and maximum grey level value
            in the generated images. Defaults to (0, 1).

    """

    def __init__(self):
        super().__init__()

        self.types = ['linear', 'radial']
        self.types_pr = [0.5, 0.5]
        self.random_pos = True
        self.grey_range = 1.
        self.grey_limit = (0., 1.)

    def generate_params(self):
        p = {
            'type': np.random.choice(self.types, 1, p=self.types_pr)[0]
        }
        if p['type'] == 'linear':
            p['angle'] = np.random.uniform(0, 2*np.pi)
        if p['type'] == 'radial':
            p['origin'] = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        return p

    def generate(self):
        p = self.params_current

        range = self.grey_limit
        if self.grey_range < 1:
            q = range[1] - self.grey_range
            rn = np.random.uniform(range[0], q)
            range = (rn, rn + self.grey_range)

        if p['type'] == 'linear':
            logging.debug("Linear gradient: angle={0:.3f} rad ({1:.3f} deg)".format(
                p['angle'], 180 * p['angle'] / np.pi))
            im = self._linear_gradient(self.dim, p['angle'], range)
        if p['type'] == 'radial':
            logging.debug("Radial gradient: origin=x:{0},y:{1}".format(
                p['origin'][0], p['origin'][1]))
            im = self._radial_gradient(self.dim, p['origin'], range)

        im = utils.feature_scale(im, 0, 255, 0., 1., 'uint8')
        return im

    def _linear_gradient(self, dim, angle=0, range=(0,1)):
        x = np.linspace(0, 1, dim[0])
        y = np.linspace(0, 1, dim[1])
        x, y = np.meshgrid(x, y)

        ret = np.cos(angle)*x + np.sin(angle)*y
        ret = np.interp(ret, (ret.min(), ret.max()), range)
        return ret

    def _radial_gradient(self, dim, origin=(0,0), range=(0,1)):
        x = np.linspace(-1, 1, dim[0])
        y = np.linspace(-1, 1, dim[1])
        x, y = np.meshgrid(x, y)

        ret = np.sqrt((x - origin[0]) ** 2 + (y - origin[1]) ** 2)
        ret = np.interp(ret, (ret.min(), ret.max()), range)
        return ret
