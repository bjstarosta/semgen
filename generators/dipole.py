import logging
import numpy as np

import generators
import utils


class DipoleGenerator(generators.Generator):
    """Generates images of radial dipole-like gradients.

    Attributes:
        dipole_type (float): Determines the shape of the generated image.
            Value approaching 1 will generate an image resembling a raised point,
            2-3 for electric dipole visualisation, >3 approaches a circle divided
            down its middle.
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

        self.dipole_type = 1
        self.clip_min = (-0.75, 1)
        self.clip_max = (-0.25, 1)
        self.grey_range = 1.
        self.grey_limit = (0., 1.)

    def generate_params(self):
        p = {
            'angle': np.random.uniform(0, 2*np.pi)
        }
        return p

    def generate(self):
        p = self.params_current

        logging.debug("Params: angle={0:.3f} rad ({1:.3f} deg)".format(
            p['angle'], 180 * p['angle'] / np.pi))

        range = self.grey_limit
        if self.grey_range < 1:
            rn = np.random.uniform(range[0], range[1] - self.grey_range)
            range = (rn, rn + self.grey_range)

        x = np.linspace(-1, 1, self.dim[0])
        y = np.linspace(-1, 1, self.dim[1])
        x, y = np.meshgrid(x, y)

        # rotate axes according to angle
        rotmatrix = np.array(
            [[np.cos(p['angle']), np.sin(p['angle'])],
            [-np.sin(p['angle']), np.cos(p['angle'])]])
        x, y = np.einsum('ji, mni -> jmn', rotmatrix, np.dstack([x, y]))

        # dipole equation
        f = x / np.sqrt(x ** 2 + y ** 2)**self.dipole_type

        clip = (
            np.random.uniform(self.clip_min[0], self.clip_max[0]),
            np.random.uniform(self.clip_min[1], self.clip_max[1]))
        f = np.clip(f, clip[0], clip[1])
        f = np.interp(f, (f.min(), f.max()), range)

        im = utils.feature_scale(f, 0, 255, 0., 1., 'uint8')
        return im
