# -*- coding: utf-8 -*-
"""SEMGen - Dipole generator class.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import logging
import numpy as np

import generators
import utils


class DipoleGenerator(generators.Generator):
    """Generates dipole equation based approximations of threading dislocations.

    Attributes:
        dipole_n (int|tuple of ints): Number of dipoles to generate on the
            image. If tuple of ints, the first index will serve as the lower
            limit of N and the second as the upper, with the actual number
            generated randomly.
        dipole_offset (tuple of floats): Minimum and maximum offset of
            generated dipoles. If set to the maximum value (-1, 1) this offset
            covers the whole image. The actual offset will be chosen randomly.
        dipole_min_d (float): Minimum distance between generated dipoles.
        dipole_rot (tuple of floats): Minimum and maximum rotation in radians
            for the generated dipoles. The actual rotation will be chosen
            randomly.
        dipole_rot_dev (float): Rotation deviation in radians. While the
            dipole rotation will be propagated to all dipoles on the same
            image, this parameter allows for a small amount of deviation to
            be applied to each individual dipole.
        dipole_contrast (tuple of floats): Minimum and maximum dipole contrast.
            Valid values range from 0 to 1. Will be chosen randomly from this
            range.
        dipole_mask_size (tuple of floats): Minimum and maximum dipole mask
            size. The mask is a 2-dim Gaussian used to extract only the area
            surrounding the centre of the dipole before placing it on the
            final generated image. Higher numbers mean a smaller mask.
            First index should be the smaller number. Will be chosen randomly
            from this range.
        enable_gradient (bool): Generates linear gradients as a background if
            set to True. If False, the background will consist of 50% grey.
        gradient_limit (tuple of floats): Minimum and maximum gray level limit
            for background gradients. The value range is -1 to 1 with -1 being
            black and 1 being white.
        gradient_range (tuple of floats): Minimum and maximum value of the
            gradient range, i.e. the range between the darkest and lightest
            gray used. The actual parameter will be chosen randomly using these
            limits.

    """

    def __init__(self):
        """Init function."""
        super().__init__()

        self.dipole_n = 1
        self.dipole_offset = (-1, 1)
        self.dipole_min_d = 0.05
        self.dipole_rot = (0, 2 * np.pi)
        self.dipole_rot_dev = 0.1 * np.pi
        self.dipole_contrast = (0.05, 0.5)
        self.dipole_mask_size = (3, 6)

        self.enable_gradient = True
        self.gradient_limit = (-0.5, 0.5)
        self.gradient_range = (0.1, 0.5)

    def generate_params(self):
        if isinstance(self.dipole_n, int):
            n_ = self.dipole_n
        else:
            n_ = np.random.randint(self.dipole_n[0], self.dipole_n[1])

        p = {
            'n': n_,
            'rot': np.random.uniform(self.dipole_rot[0], self.dipole_rot[1]),
            'rot_dev': self.dipole_rot_dev,
            'dipoles': []
        }

        if self.enable_gradient is True:
            grad_range = np.random.uniform(
                self.gradient_range[0], self.gradient_range[1])
            grad_start = np.random.uniform(
                self.gradient_limit[0], self.gradient_limit[1] - grad_range)
            grad_end = grad_start + grad_range
            p['gradient'] = (grad_start, grad_end)

        offsets = []
        min_d = self.dipole_min_d ** 2
        for i in range(n_):
            # test for minimum distance
            while True:
                offset = (
                    np.random.uniform(
                        self.dipole_offset[0], self.dipole_offset[1]),
                    np.random.uniform(
                        self.dipole_offset[0], self.dipole_offset[1]))

                ok = True
                for o in offsets:
                    c = (offset[0] - o[0]) ** 2 + (offset[1] - o[1]) ** 2
                    if c < min_d:
                        ok = False
                if ok is True:
                    offsets.append(offset)
                    break

            p['dipoles'].append({
                'offset': offset,
                'rot': np.random.uniform(
                    p['rot'] - p['rot_dev'], p['rot'] + p['rot_dev']),
                'contrast': np.random.uniform(
                    self.dipole_contrast[0], self.dipole_contrast[1]),
                'mask_size': np.random.uniform(
                    self.dipole_mask_size[0], self.dipole_mask_size[1])
            })

        return p

    def generate(self):
        p = self.params_current

        _debugstr = "Params: n={0}, angle={1:.3f}rad ({2:.3f}deg), "\
            "angle dev={3:.3f}rad ({4:.3f}deg)"
        logging.debug(_debugstr.format(p['n'],
            p['rot'], 180 * p['rot'] / np.pi,
            p['rot_dev'], 180 * p['rot_dev'] / np.pi))

        dim_xy = (self.dim[1], self.dim[0])

        if 'gradient' in p:
            im = self._draw_linear_gradient(self.dim, p['rot'], p['gradient'])
        else:
            im = np.zeros(self.dim)

        for i in range(p['n']):
            gen = self._draw_dipole(dim_xy,
                p['dipoles'][i]['rot'],
                p['dipoles'][i]['offset']) * p['dipoles'][i]['contrast']
            mask = self._draw_gaussian(dim_xy,
                p['dipoles'][i]['mask_size'],
                p['dipoles'][i]['offset'])
            gen_ = gen * mask
            im = im + gen_

        im = np.clip(im, -1, 1)
        im = utils.feature_scale(im, 0, 255, -1., 1., 'uint8')

        if self.labels is not None:
            self.labels.add_label([p['n'], p['rot']])
        return im

    def _draw_dipole(self, dim, rot, offset=(0, 0), pow=1):
        """Calculate the dipole equation in 2-dim and return result as matrix.

        Args:
            dim (tuple): Result matrix dimensions in (x, y) format.
            rot (float): Rotation in radians.
            offset (tuple): Offset from centre in (x, y) format.
                Offset of (0, 0) means centre. Non-zero numbers shift the
                result in the x and y direction respectively.
            pow (float): Radix of the dipole equation.
                Value approaching 1 will generate an image resembling a raised
                point, 2-3 for electric dipole visualisation, >3 approaches a
                circle divided down its middle.

        Returns:
            numpy.ndarray: Matrix containing result of the dipole equation.

        """
        X = np.linspace(-1 - offset[0], 1 - offset[0], dim[0])
        Y = np.linspace(-1 - offset[1], 1 - offset[1], dim[1])
        X, Y = np.meshgrid(X, Y)

        rotmatrix = np.array(
            [[np.cos(rot), np.sin(rot)],
            [-np.sin(rot), np.cos(rot)]]
        )
        X, Y = np.einsum('ji, mni -> jmn', rotmatrix, np.dstack([X, Y]))

        res = X / np.sqrt(X ** 2 + Y ** 2)**pow
        return res

    def _draw_gaussian(self, dim, size=1, offset=(0, 0)):
        """Calculate the Gaussian in 2-dim and return result as matrix.

        Args:
            dim (tuple): Result matrix dimensions in (x, y) format.
            size (float): Size of Gaussian "blob".
                Larger number means smaller size.
            offset (tuple): Offset from centre in (x, y) format.
                Offset of (0, 0) means centre. Non-zero numbers shift the
                result in the x and y direction respectively.

        Returns:
            numpy.ndarray: Matrix containing result of the 2-dim Gaussian.

        """
        X = np.linspace(-1 - offset[0], 1 - offset[0], dim[0])
        Y = np.linspace(-1 - offset[1], 1 - offset[1], dim[1])
        X, Y = np.meshgrid(X, Y)

        res = np.exp(-((X**2 / 2 * size**2) + (Y**2 / 2 * size**2)))
        res = np.interp(res, (res.min(), res.max()), (0, 1))
        return res

    def _draw_linear_gradient(self, dim, angle=0, range=(0, 1)):
        X = np.linspace(0, 1, dim[0])
        Y = np.linspace(0, 1, dim[1])
        X, Y = np.meshgrid(X, Y)

        ret = np.cos(angle) * X + np.sin(angle) * Y
        ret = np.interp(ret, (ret.min(), ret.max()), range)
        return ret
