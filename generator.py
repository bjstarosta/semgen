import logging
import multiprocessing

import numpy as np
import utils


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
            logging.debug("Image generation parameter list empty, autogenerating")
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
        return

    def _getpx(self, im, x, y):
        if x < 0 or x >= im.shape[1] or y < 0 or y >= im.shape[0]:
            return
        return im[y, x]

    def _setpx(self, im, x, y, c):
        if x < 0 or x >= im.shape[1] or y < 0 or y >= im.shape[0]:
            return
        im[y, x] = c

    def _slice(self, src, dst, x, y):
        """Safely slices two numpy arrays together, truncating the source
        array as necessary to fit the destination array.

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
        if x < -src.shape[1] or y < -src.shape[0] or x > dst.shape[1] or y > dst.shape[0]:
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

        dst[y:y+src.shape[0], x:x+src.shape[1]] = src
        return dst


class GradientGenerator(Generator):
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


class GoldOnCarbonGenerator(Generator):
    """Simulates images that closely mimic the gold-on-carbon SEM test sample
    images that usually consist of bright grains on a dark background.

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
        n_workers (int): Number of worker processes to spawn during grain
            drawing.

    """

    def __init__(self):
        super().__init__()

        self.grain_n = (10, 16)
        self.grain_r = (self.dim[0] * 0.05, self.dim[0] * 0.3)
        self.grain_a1 = (0, 0.45)
        self.grain_a2 = (0, 0.1)
        self.grain_f1 = (0, 2*np.pi)
        self.grain_f2 = (0, 2*np.pi)

        self.grain_colour = 0.30 # C_g
        self.grain_edge_colour = 0.7 # C_e
        self.grain_edge_width = 5 # r_t
        self.grain_edge_steepness = 0.43 # b
        self.grain_fill_k = 3
        self.grain_fill_gain = 100
        self.bg_fill_gain = 50
        self.bg_fill_offset = 0.1
        self.bg_pm_size = 5
        self.max_grain_candidates = 100
        self.n_workers = 8

        self._margin = 1
        self._step = 7200
        self._rstep = 0.1
        self.debug = {}

    def generate_params(self):
        #mxs = int((self.grain_fill_k * self.dim[0]) / 175)
        mxs = int((self.grain_fill_k * 6) / 2) # set by trial and error on a 200x200 image
        params = {
            'grain_n': np.random.randint(self.grain_n[0], self.grain_n[1]),
            'grain_tex': np.random.random_sample((mxs, mxs)),
            'bg_tex': np.random.random_sample((self.bg_pm_size, self.bg_pm_size)),
            'grains': []
        }
        for i in range(0, params['grain_n']):
            params['grains'].append({
                'x': np.random.randint(0, self.dim[0]),
                'y': np.random.randint(0, self.dim[1]),
                'r': np.random.uniform(self.grain_r[0], self.grain_r[1]),
                'a1': np.random.uniform(self.grain_a1[0], self.grain_a1[1]),
                'a2': np.random.uniform(self.grain_a2[0], self.grain_a2[1]),
                'f1': np.random.uniform(self.grain_f1[0], self.grain_f1[1]),
                'f2': np.random.uniform(self.grain_f2[0], self.grain_f2[1])
            })
        return params

    def generate(self):
        """Generates a visual representation of bright gold grains on carbon
        through a perfect (non-noisy) SEM.

        Args:
            params (list): A list of parameters to use while generating the
                image. If None is passed, the parameters will be generated
                randomly according to the relevant object properties.

        Returns:
            A tuple containing a numpy array containing a grayscale image
            with pixels encoded in float format [0-1] in its first element,
            and a list of parameters used to create the image in its second.

        """
        im = np.zeros((self.dim[1], self.dim[0]))
        grain_mask = np.zeros((self.dim[1], self.dim[0]), dtype="bool")

        # Draw grain shapes
        grain_masks = []
        params_cn = []
        for j in range(0, self.params_current['grain_n']):
            i = j + 1
            if i > self.max_grain_candidates:
                break

            g = self.params_current['grains'][j]

            logging.debug("Candidate grain {0}: x={1}, y={2}, r={3:.3f}".format(
                i, g['x'], g['y'], g['r']))
            mask_cn, params_cn_ = self._draw_grain_mask(
                g['r'], g['x'], g['y'], g['a1'], g['a2'], g['f1'], g['f2'])

            # Discard candidate if there is overlap
            if np.max(grain_mask & mask_cn) == True:
                logging.debug("Candidate overlaps, discarding.")
                continue
            else:
                logging.debug("!! Candidate accepted.")

            #im = im + self._draw_grain(params_cn)
            grain_mask = grain_mask + mask_cn
            grain_masks.append(mask_cn)
            params_cn.append(params_cn_)

        self.debug['grain_mask'] = grain_mask

        # Draw all grains
        logging.debug("Drawing {0} grains using {1} workers.".format(len(params_cn), self.n_workers))
        if self.n_workers <= 1:
            for i in params_cn:
                im = im + self._draw_grain(i)
        else:
            with multiprocessing.Pool(self.n_workers) as pool:
                for i in pool.map(self._draw_grain, params_cn):
                    im = im + i

        # Draw grain texture
        logging.debug("Drawing grain texture.")
        fft = self._fft_texture(self.params_current['grain_tex'])
        tx = fft * self.grain_fill_gain
        self.debug['grain_texture'] = tx
        for m in grain_masks:
            im = im + np.where(m, tx, 0)

        # Draw background
        logging.debug("Filling background.")
        fft = self._fft_texture(self.params_current['bg_tex'])
        tx = (fft * self.bg_fill_gain) + self.bg_fill_offset
        im = im + np.where(~grain_mask, tx, 0)

        # Transform to byte data
        im = np.clip(im, a_min=0., a_max=1.)
        im = utils.feature_scale(im, 0, 255, 0., 1., 'uint8')

        return im

    def _draw_grain_mask(self, r, x0, y0, a1, a2, f1, f2):
        """Draws a parametric grain-like shape according to the specified
        arguments and returns a binary mask of it.

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
            A tuple containing the mask in its first element, and the parameters
            used to create it in the second.

        """
        phi = np.linspace(0, 2*np.pi, self._step) # domain
        d = 1 + a1 * np.sin(2 * phi + f1) + a2 * np.sin(3 * phi + f2)  # deformation func
        x = r * d * np.cos(phi) # + x0
        y = r * d * np.sin(phi) # + y0

        lx = int(np.round(x0 + np.min(x))) # x0 - point furthest left
        ty = int(np.round(y0 + np.min(y))) # y0 - point furthest top
        rx = int(np.round(x0 + np.max(x))) # x0 + point furthest right
        by = int(np.round(y0 + np.max(y))) # y0 + point furthest bottom

        # make all coords positive integers for pixel positions
        x = x + np.abs(np.min(x)) + self._margin
        y = y + np.abs(np.min(y)) + self._margin
        xy = np.column_stack((x, y)).astype('uint32')

        # set up mask and draw the outline
        dim = (int(np.max(x) + (self._margin*2)), int(np.max(y)) + (self._margin*2))
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
            mask = mask[:, 0:(mask.shape[1]-(rx-self.dim[0]))]
        if by > self.dim[1]:
            mask = mask[0:(mask.shape[0]-(by-self.dim[1])), :]
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
        """Uses parameters generated during the use of _draw_grain_mask() to
        generate a simulation of a grain.

        Args:
            params (tuple): Accepts the second element of the tuple returned by
                _draw_grain_mask() to recreate exactly the boundaries of that
                particular mask.

        Returns:
            A 2-dim array containing the grain image.

        """
        (r, d, phi, lx, ty, mask) = params

        # use the mask to generate the grain shape
        dim = (mask.shape[0] + (self._margin*2), mask.shape[1] + (self._margin*2))
        grain = np.zeros(dim)
        grain[self._margin:-self._margin, self._margin:-self._margin] = mask.astype('uint32') * self.grain_colour

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
                cd = np.exp(-self.grain_edge_steepness * self.grain_edge_width) - 1
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
        logging.debug("Finished drawing grain at x={1}, y={2}, r={0:.3f}.".format(r, lx, ty))
        return grain_

    def _fft_texture(self, m):
        fft = np.fft.fft2(m)
        m = np.zeros((self.dim[1], self.dim[0]), dtype="complex")
        m[:fft.shape[0], :fft.shape[1]] = fft
        return np.fft.ifft2(m).real


class ECCIDefectGenerator(Generator):
    """Simulates GaN surface images of dislocations.

    Attributes:

    """

    def __init__(self):
        pass
