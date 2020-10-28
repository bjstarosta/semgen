# -*- coding: utf-8 -*-
"""SEMGen - SEM noise and distortion generating class.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import logging
import numpy as np
import scipy as sp

import distorters


class SEMNoiseGenerator(distorters.Distorter):
    """Simulate SEM imaging distortion on existing images.

    SEM imaging distortion components (focus blur, drift, vibration, noise)
    are based on the physical principles of SEM operation.

    See: [1] P. Cizmar, A. E. Vladár, B. Ming, and M. T. Postek,
    ‘Simulated SEM images for resolution measurement’,
    Scanning, vol. 30, no. 5, pp. 381–391, 2008, doi: 10.1002/sca.20120.
    URL: https://onlinelibrary.wiley.com/doi/pdf/10.1002/sca.20120

    Attributes:
        gm_size (int): Size of the Gaussian convolution matrix. Higher means
            more blurring. Should be an odd number.
        astigmatism_coeff (float): Astigmatism coefficient. Values different
            than 1 distort the Gaussian function along either the X or Y axis.
        astigmatism_rotation (float): Astigmatism rotation in radians.
        scan_passes (int): Number of times the vibration function is applied
            to the image. Higher number means a more diffuse drift effect at
            the cost of an increase in processing time.
        v_complexity (int): The vibration function is a superposition of
            random waves. This is the number of waves in this superposition.
        A_lim (tuple): Lower and upper limit of the amplitude of the vibration
            function, or the maximum amount of pixels a drift can occur over.
        f_lim (tuple): Lower and upper limit of the frequency of the vibration
            function, or the width of the drift distortion.
            Values that are significant when compared to the dimensions of the
            distorted image will make it more likely that black pixels will
            appear.
        Q_g (float): Coefficient of the Gaussian noise magnitude.
        Q_p (float): Coefficient of the Poisson noise magnitude.

    """

    def __init__(self):
        super().__init__()

        self.gm_size = 15
        self.astigmatism_coeff = 0.95
        self.astigmatism_rotation = (1 / 4) * np.pi
        self.scan_passes = 2
        self.v_complexity = 4
        self.A_lim = (5, 10)
        self.f_lim = (20, 25)
        self.Q_g = 0.0129
        self.Q_p = 0.0422
        # self.Q_g = 0.0720
        # self.Q_p = 0.0164
        self._rs = None

        self.debug = {}

    def generate_params(self):
        self._rs = np.random.RandomState()
        return {}

    def process(self, task):
        self._rs = np.random.RandomState()
        image = task.image

        # Out-of-focus effects (gaussian blur with astigmatism)
        logging.debug(
            ("Focus component: gm size={0},"
            "astig. c={1:.3f}, astig. rot={2:.3f}").format(
                self.gm_size,
                self.astigmatism_coeff,
                self.astigmatism_rotation))

        image = sp.ndimage.convolve(image, self.gaussian_matrix(
            step=self.gm_size,
            s=self.astigmatism_coeff,
            phi_s=self.astigmatism_rotation,
            norm=True
        ), mode='reflect')
        self.debug['post-focus'] = np.copy(image)

        # Drift and vibration
        t_end = 2 * np.pi * self.scan_passes
        t = np.linspace(0, t_end,
            image.shape[0] * image.shape[1] * self.scan_passes)
        xv, yv = self.vibration_function(t, self.v_complexity,
            self.A_lim, self.f_lim)

        logging.debug(
            ("Drift component: scan passes={0}, "
            "t size={1}, superposition complexity={2}").format(
                self.scan_passes, len(t), self.v_complexity))

        image_ = np.copy(image)
        # image = np.zeros((image.shape[1], image.shape[0]))
        i = 0
        xv_f = np.floor(xv)
        xv_c = np.ceil(xv)
        yv_f = np.floor(yv)
        yv_c = np.ceil(yv)

        def bi_px_newvals(x, y, coords):
            return (np.array([[1 - x, x]])
                @ coords
                @ np.array([[1 - y, y]]).T)[0][0]

        it = np.nditer(image_, flags=['multi_index'])
        for j in range(0, self.scan_passes):
            logging.debug("- Raster scan pass {0}".format(j + 1))
            it.reset()
            for px in it:
                xs = it.multi_index[1]
                ys = it.multi_index[0]

                if xv_f[i] == xv_c[i] and yv_f[i] == yv_c[i]:
                    self._setpx(image, int(xs + xv[i]), int(ys + yv[i]), px)

                else:
                    # Bilinear interpolation for fractional pixel values
                    xs_ = int(xs + xv_f[i])
                    ys_ = int(ys + yv_f[i])

                    if xv[i] > 0:
                        xsm = [xs_, xs_ + 1]
                    elif xv[i] < 0:
                        xsm = [xs_ - 1, xs_]
                    else:
                        xsm = [xs_, xs_]

                    if yv[i] > 0:
                        ysm = [ys_, ys_ + 1]
                    elif xv[i] < 0:
                        ysm = [ys_ - 1, ys_]
                    else:
                        ysm = [ys_, ys_]

                    coords = np.array([
                        [self._getpx(image_, xsm[0], ysm[0]),
                        self._getpx(image_, xsm[1], ysm[0])],
                        [self._getpx(image_, xsm[0], ysm[1]),
                        self._getpx(image_, xsm[1], ysm[1])]
                    ])

                    if None not in coords:
                        # print(bi_px_newvals(0, 0))
                        self._setpx(image, xsm[0], ysm[0],
                            bi_px_newvals(0, 0, coords))
                        self._setpx(image, xsm[1], ysm[0],
                            bi_px_newvals(1, 0, coords))
                        self._setpx(image, xsm[0], ysm[1],
                            bi_px_newvals(0, 1, coords))
                        self._setpx(image, xsm[1], ysm[1],
                            bi_px_newvals(1, 1, coords))

                i = i + 1

        self.debug['post-drift'] = np.copy(image)

        # Gaussian and Poisson noise sum
        logging.debug(
            ("Noise component: Poisson coeff.={0:.3f}, "
            "Gaussian coeff.={1:.3f}").format(
                self.Q_p, self.Q_g))
        image = self.noise_cmpnt(image, self.Q_g, self.Q_p)

        self.debug['t'] = t
        self.debug['xv'] = xv
        self.debug['yv'] = yv

        task.image = image.astype('uint8')
        return task

    def gaussian_matrix(self,
    sigma=1, domain=3, step=5, s=1, phi_s=0, norm=False):
        """Return a matrix of specific size containing the 2D Gaussian function.

        Args:
            sigma (float): Gaussian RMS width (standard deviation).
            domain (float): Half width of domain in both X and Y direction.
                At sigma=1 the Gaussian function approaches zero at around
                3*sigma in every direction.
            step (int): Axis length of the output matrix. The number of
                elements in the matrix will be this number squared.
            s (float): Astigmatism coefficient. Values different than 1
                distort the Gaussian function along either the X or Y axis.
            phi_s (float): Astigmatism rotation in radians.
            norm (bool): If True, will normalise the output matrix so that the
                sum of its elements is equal to 1.

        Returns:
            A step*step size matrix.

        """
        sp = np.linspace(-domain, domain, step)
        x, y = np.meshgrid(sp, sp)

        x_ = s * (x * np.cos(phi_s) + y * np.sin(phi_s))
        y_ = (1 / s) * (-x * np.sin(phi_s) + y * np.cos(phi_s))

        m = (1 / (2 * np.pi * sigma**2))
        p = m * np.exp(-(x_**2 + y_**2) / 2 * (sigma**2))

        if norm is True:
            p = p / np.sum(p)

        return p

    def vibration_function(self, t, sn, A_lim, f_lim):
        """Return X and Y components of a vibration function.

        Approximates all of the sources of vibration affecting SEM images, and
        combines them into a superposition of sine functions which can be used
        for pixel displacement.

        Args:
            img (ndarray): Array containing image data.
            t (ndarray): One dimensional array containing all of the temporal
                datapoints.
            sn (int): Amount of wave components present in the superposition of
                sine functions.
            A_lim (tuple): Lower and upper limit of the amplitude in each
                wave component.
            f_lim (tuple): Lower and upper limit of the frequency in each
                wave component.

        Returns:
            A tuple containing the x and y components of the vibration function
            respectively.

        """
        v = np.vectorize(
            lambda t, A, f, p: A * np.sin(f * t + p)
        )

        x_v = 0
        y_v = 0
        for i in range(0, sn):
            x_v = x_v + v(t,
                self._rs.uniform(A_lim[0], A_lim[1]),
                self._rs.uniform(f_lim[0], f_lim[1]),
                self._rs.uniform(0, 2 * np.pi))
            y_v = y_v + v(t,
                self._rs.uniform(A_lim[0], A_lim[1]),
                self._rs.uniform(f_lim[0], f_lim[1]),
                self._rs.uniform(0, 2 * np.pi))

        return x_v, y_v

    def noise_cmpnt(self, img, gaussian_c, poisson_c):
        """Apply additive noise to an array containing image data.

        Approximates all of the sources of noise present in SEM images, and
        combines them into a single noise component of the form:

            C_3 = C_2 + (Q_g + Q_p * sqrt(C_2)) * R_i

        where C_2 is the grey level of the current pixel, Q_g and Q_p are
        respectively the gaussian and poisson noise magnitude coefficients,
        and R_i is a uniform distribution random number.

        Args:
            img (ndarray): Array containing image data.
            gaussian_c (float): Gaussian noise magnitude coefficient.
                Increase this for more background independent noise.
                Domain is between 0 and 1.
            poisson_c (float): Poisson noise magnitude coefficient.
                Increase this for more background dependent noise.
                Domain is between 0 and 1.

        Returns:
            An array containing image data with the noise component applied.

        """
        fn = np.vectorize(
            lambda x: x + ((gaussian_c + (poisson_c * np.sqrt(x)))
                * self._rs.uniform(-1, 1))
        )

        a = img.max()
        if isinstance(a, int) or a > 1:
            img = np.interp(img, (0, 255), (0., 1.))

        img = np.clip(fn(img), a_min=0., a_max=1.)
        img = np.interp(img, (0., 1.), (0, 255)).astype(int)
        return img
