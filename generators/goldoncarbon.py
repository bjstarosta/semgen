# -*- coding: utf-8 -*-
"""SEMGen - Gold on carbon generator class.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import logging
import numpy as np

import generators
import utils


class GoldOnCarbonGenerator(generators.Generator):
    """Simulate images closely mimicking the gold-on-carbon SEM test sample.

    These images that usually consist of bright grains on a dark background.

    See: [1] P. Cizmar, A. E. Vladár, B. Ming, and M. T. Postek,
    ‘Simulated SEM images for resolution measurement’,
    Scanning, vol. 30, no. 5, pp. 381–391, 2008, doi: 10.1002/sca.20120.
    URL: https://onlinelibrary.wiley.com/doi/pdf/10.1002/sca.20120

    Attributes:
        grain_n (tuple): Expectation of amount of generated grains in
            (min, max) format.
        grain_r (tuple): Radius expectation of grain shape in (min, max)
            format.
        grain_a1 (tuple): First amplitude deformation factor expectation
            in (min, max) format.
        grain_a2 (tuple): Second amplitude deformation factor expectation
            in (min, max) format.
        grain_f1 (tuple): First phase shift deformation factor expectation
            in (min, max) format.
        grain_f2 (tuple): Second phase shift deformation factor expectation
            in (min, max) format.
        grain_colour (float): Shade of gray to fill in the grains with.
            Values between 0 and 1, where 1 is completely white.
        grain_edge_colour (float): Shade of gray applied on the grains for the
            SEM edge effect. Values between 0 and 1, where 1 is completely
            white.
        grain_edge_steepness (float): Controls the steepness of the edge
            effect transition. Higher number means wider edge.
        grain_fill_k (float): Variable coefficient for grain fill (structure)
            matrix size.
        grain_fill_gain (float): The grain fill texture matrix can come out
            with really small numbers depending on image size, so this allows
            its multiplication to increase the impact of the texture.
        bg_fill_gain (float): See above.
        bg_fill_offset (float): Allows increasing the brightness of the
            background texture.
        bg_pm_size (int): The size of the background pattern matrix.
        max_grain_candidates (int): Maximum number of generated grain
            candidates.

            The generate() function tries to generate non overlapping grains
            to fulfill the number of expected grains on the image. If a grain
            candidate overlaps existing grains on the image, it is discarded.
            Depending on the size and number of already generated grains this
            can potentially lead to an infinite loop. Setting this number to
            a finite integer prevents that.

    """

    def __init__(self):
        super().__init__()

        self.grain_n = (10, 16)
        self.grain_r = (self.dim[0] * 0.05, self.dim[0] * 0.3)
        self.grain_a1 = (0, 0.45)
        self.grain_a2 = (0, 0.1)
        self.grain_f1 = (0, 2 * np.pi)
        self.grain_f2 = (0, 2 * np.pi)

        self.grain_colour = 0.30  # C_g
        self.grain_edge_colour = 0.7  # C_e
        self.grain_edge_width = 5  # r_t
        self.grain_edge_steepness = 0.43  # b
        self.grain_fill_k = 3
        self.grain_fill_gain = 100
        self.bg_fill_gain = 50
        self.bg_fill_offset = 0.1
        self.bg_pm_size = 5
        self.max_grain_candidates = 100

        self._margin = 1
        self._step = 7200
        self._rstep = 0.1
        self.debug = {}

    def generate_params(self):
        rs = np.random.RandomState()

        # mxs = int((self.grain_fill_k * self.dim[0]) / 175)
        # set by trial and error on a 200x200 image
        mxs = int((self.grain_fill_k * 6) / 2)
        params = {
            'grain_n': rs.randint(self.grain_n[0], self.grain_n[1]),
            'grain_tex': rs.random_sample((mxs, mxs)),
            'bg_tex': rs.random_sample(
                (self.bg_pm_size, self.bg_pm_size)),
            'grains': []
        }
        for i in range(0, params['grain_n']):
            params['grains'].append({
                'x': rs.randint(0, self.dim[0]),
                'y': rs.randint(0, self.dim[1]),
                'r': rs.uniform(self.grain_r[0], self.grain_r[1]),
                'a1': rs.uniform(self.grain_a1[0], self.grain_a1[1]),
                'a2': rs.uniform(self.grain_a2[0], self.grain_a2[1]),
                'f1': rs.uniform(self.grain_f1[0], self.grain_f1[1]),
                'f2': rs.uniform(self.grain_f2[0], self.grain_f2[1])
            })
        return params

    def process(self, task):
        """Generate a visual representation of bright gold grains on carbon.

        Output has none of the SEM specific noise or distortion applied to it.
        To apply SEM noise see distorters.SEMNoise.

        Args:
            params (list): A list of parameters to use while generating the
                image. If None is passed, the parameters will be generated
                randomly according to the relevant object properties.

        Returns:
            A tuple containing a numpy array containing a grayscale image
            with pixels encoded in float format [0-1] in its first element,
            and a list of parameters used to create the image in its second.

        """
        p = task.params

        im = np.zeros((self.dim[1], self.dim[0]))
        grain_mask = np.zeros((self.dim[1], self.dim[0]), dtype="bool")

        # Draw grain shapes
        grain_masks = []
        params_cn = []
        for j in range(0, p['grain_n']):
            i = j + 1
            if i > self.max_grain_candidates:
                break

            g = p['grains'][j]

            logging.debug(
                "Candidate grain {0}: x={1}, y={2}, r={3:.3f}".format(
                    i, g['x'], g['y'], g['r']))
            mask_cn, params_cn_ = self._draw_grain_mask(
                g['r'], g['x'], g['y'], g['a1'], g['a2'], g['f1'], g['f2'])

            # Discard candidate if there is overlap
            if np.max(grain_mask & mask_cn) is True:
                logging.debug("Candidate overlaps, discarding.")
                continue
            else:
                logging.debug("!! Candidate accepted.")

            # im = im + self._draw_grain(params_cn)
            grain_mask = grain_mask + mask_cn
            grain_masks.append(mask_cn)
            params_cn.append(params_cn_)

        self.debug['grain_mask'] = grain_mask

        # Draw all grains
        logging.debug(
            "Drawing {0} grains using {1} workers.".format(
                len(params_cn), self.n_workers))
        for i in params_cn:
            im = im + self._draw_grain(i)

        # Draw grain texture
        logging.debug("Drawing grain texture.")
        fft = self._fft_texture(p['grain_tex'])
        tx = fft * self.grain_fill_gain
        self.debug['grain_texture'] = tx
        for m in grain_masks:
            im = im + np.where(m, tx, 0)

        # Draw background
        logging.debug("Filling background.")
        fft = self._fft_texture(p['bg_tex'])
        tx = (fft * self.bg_fill_gain) + self.bg_fill_offset
        im = im + np.where(~grain_mask, tx, 0)

        # Transform to byte data
        im = np.clip(im, a_min=0., a_max=1.)

        task.image = utils.feature_scale(im, 0, 255, 0., 1., 'uint8')
        return task

    def _draw_grain_mask(self, r, x0, y0, a1, a2, f1, f2):
        """Draws a parametric grain-like shape.

        The characteristics of the shape are defined by the specified
        arguments and the method returns a binary mask of it.

        Args:
            r (float): Grain shape radius.
            x0 (int): Centre point of grain shape in the X axis.
            y0 (int): Centre point of grain shape in the Y axis.
            a1 (float): First deformation factor amplitude.
                Distorts grain according to hourglass shape.
            a2 (float): Second deformation factor amplitude.
                Distorts grain in three directions.
            f1 (float): First deformation factor phase shift.
                Affects rotation of grain and distortion.
            f2 (float): Second deformation factor phase shift.
                Affects rotation of grain and distortion.

        Returns:
            A tuple containing the mask in its first element, and the
            parameters used to create it in the second.

        """
        phi = np.linspace(0, 2 * np.pi, self._step)  # domain

        # deformation func
        d = 1 + a1 * np.sin(2 * phi + f1) + a2 * np.sin(3 * phi + f2)
        x = r * d * np.cos(phi)  # + x0
        y = r * d * np.sin(phi)  # + y0

        lx = int(np.round(x0 + np.min(x)))  # x0 - point furthest left
        ty = int(np.round(y0 + np.min(y)))  # y0 - point furthest top
        rx = int(np.round(x0 + np.max(x)))  # x0 + point furthest right
        by = int(np.round(y0 + np.max(y)))  # y0 + point furthest bottom

        # make all coords positive integers for pixel positions
        x = x + np.abs(np.min(x)) + self._margin
        y = y + np.abs(np.min(y)) + self._margin
        xy = np.column_stack((x, y)).astype('uint32')

        # set up mask and draw the outline
        dim = (
            int(np.max(x) + (self._margin * 2)),
            int(np.max(y)) + (self._margin * 2))
        mask = np.zeros((dim[1], dim[0]), dtype="bool")
        for px, py in xy:
            self._setpx(mask, px, py, 1)

        # fill the mask by raycasting from left, right, top and bottom
        mask = np.maximum.accumulate(mask, 1) &\
            np.maximum.accumulate(mask[:, ::-1], 1)[:, ::-1] &\
            np.maximum.accumulate(mask[::-1, :], 0)[::-1, :] &\
            np.maximum.accumulate(mask, 0)

        # crop mask to image dimensions
        x0_ = lx
        y0_ = ty
        if rx > self.dim[0]:
            mask = mask[:, 0:(mask.shape[1] - (rx - self.dim[0]))]
        if by > self.dim[1]:
            mask = mask[0:(mask.shape[0] - (by - self.dim[1])), :]
        if lx < 0:
            mask = mask[:, np.abs(x0_):]
            x0_ = 0
        if ty < 0:
            mask = mask[np.abs(y0_):, :]
            y0_ = 0

        # remove margins
        mask = mask[self._margin:-self._margin, self._margin:-self._margin]

        # translate the position of the mask
        mask_ = np.zeros((self.dim[1], self.dim[0]), dtype="bool")
        mask_ = self._slice(mask, mask_, x0_, y0_)
        return (mask_, (r, d, phi, lx, ty, mask))

    def _draw_grain(self, params):
        """Use passed parameters to generate a simulation of a grain.

        Parameters are generated during the use of _draw_grain_mask().

        Args:
            params (tuple): Accepts the second element of the tuple returned by
                _draw_grain_mask() to recreate exactly the boundaries of that
                particular mask.

        Returns:
            A 2-dim array containing the grain image.

        """
        (r, d, phi, lx, ty, mask) = params

        # use the mask to generate the grain shape
        dim = (
            mask.shape[0] + (self._margin * 2),
            mask.shape[1] + (self._margin * 2))
        grain = np.zeros(dim)
        grain[self._margin:-self._margin, self._margin:-self._margin] = \
            mask.astype('uint32') * self.grain_colour

        # apply edge effect
        r_ = r
        while r_ > 0:
            re = r - r_
            if re > self.grain_edge_width:
                break

            x = r_ * d * np.cos(phi)
            y = r_ * d * np.sin(phi)
            x = x + np.abs(np.min(x)) + re + self._margin
            y = y + np.abs(np.min(y)) + re + self._margin
            if lx < 0:
                x = x + lx
            if ty < 0:
                y = y + ty
            xy = np.column_stack((x, y)).astype('uint32')

            for px, py in xy:
                cn = self.grain_colour - self.grain_edge_colour
                cde = -self.grain_edge_steepness * self.grain_edge_width
                cd = np.exp(cde) - 1

                c = (cn / cd) * (np.exp(-self.grain_edge_steepness * re) - 1)
                c = c + self.grain_edge_colour

                s = self._getpx(grain, px, py)
                if s is not None and s == self.grain_colour:
                    self._setpx(grain, px, py, c)

            r_ = r_ - self._rstep

        # remove margins
        grain = grain[self._margin:-self._margin, self._margin:-self._margin]

        # translate the position of the grain
        x0_ = lx
        y0_ = ty
        if lx < 0:
            x0_ = 0
        if ty < 0:
            y0_ = 0

        grain_ = np.zeros((self.dim[1], self.dim[0]))
        grain_ = self._slice(grain, grain_, x0_, y0_)
        logging.debug(
            "Finished drawing grain at x={1}, y={2}, r={0:.3f}.".format(
                r, lx, ty))
        return grain_

    def _fft_texture(self, m):
        fft = np.fft.fft2(m)
        m = np.zeros((self.dim[1], self.dim[0]), dtype="complex")
        m[:fft.shape[0], :fft.shape[1]] = fft
        return np.fft.ifft2(m).real
