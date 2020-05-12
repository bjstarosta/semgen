import numpy as np
import scipy as sp
from semgen import load_image


class Distorter(object):
    """Abstract parent class for all the distorter subclasses.

    Subclasses become iterators operating on a queue."""

    def __init__(self):
        self._queue = []

    def __iter__(self):
        while len(self._queue) > 0:
            yield self.distort(load_image(self._queue.pop(0)))

    def __next__(self):
        if len(self._queue) > 0:
            return self.distort(load_image(self._queue.pop(0)))
        else:
            return

    def queue_images(self, list):
        self._queue = self._queue + list

    def queue_is_empty(self):
        if len(self._queue) > 0:
            return False
        else:
            return True

    def distort(self, image):
        return

    def _getpx(self, im, x, y):
        if x < 0 or x >= im.shape[1] or y < 0 or y >= im.shape[0]:
            return
        return im[y, x]

    def _setpx(self, im, x, y, c):
        if x < 0 or x >= im.shape[1] or y < 0 or y >= im.shape[0]:
            return
        im[y, x] = c

    def _loadfile(self, path):
        with tifffile.TiffFile('/home/bjs/python/test/Nouf data/26.09.13 nanodash aligned set 1/SH4_3C1.TIF') as tif:
            img = tif.asarray()


class SEMNoiseGenerator(Distorter):
    """Simulates SEM imaging distortion (focus blur, drift, vibration, noise)
    based on the physical principles of SEM operation.

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
        A_max (float): Maximum amplitude of the vibration function, or the
            maximum amount of pixels a drift can occur over.
        v_complexity (int): The vibration function is a superposition of
            random waves. This is the number of waves in this superposition.
        Q_g (int): Coefficient of the Gaussian noise magnitude.
        Q_p (int): Coefficient of the Poisson noise magnitude.

    """

    def __init__(self):
        super().__init__()

        self.gm_size = 15
        self.astigmatism_coeff = 1
        self.astigmatism_rotation = (1/4)*np.pi
        self.A_max = 10
        self.v_complexity = 4
        self.Q_g = 84
        self.Q_p = 21

    def distort(self, image, params=None):

        params_ret = []

        # Out-of-focus effects (gaussian blur with astigmatism)
        phi_s = (1/4)*np.pi
        p = self._gaussian_matrix(
            step=self.gm_size,
            s=self.astigmatism_coeff,
            phi_s=self.astigmatism_rotation,
            norm=True
        )
        image = sp.ndimage.convolve(image, p, mode='reflect')

        # Drift and vibration
        xv = 0
        yv = 0
        t = np.linspace(0, 2*np.pi, image.shape[0] * image.shape[1])
        for i in range(0, self.v_complexity):
            a_x = self.A_max * np.random.uniform(0, 1)
            a_y = self.A_max * np.random.uniform(0, 1)
            f_x = np.random.uniform(image.shape[1] / 5, image.shape[1] / 4)
            f_y = np.random.uniform(image.shape[0] / 5, image.shape[0] / 4)
            ph_x = np.random.uniform(0, 2*np.pi)
            ph_y = np.random.uniform(0, 2*np.pi)

            # Vibration function (superposition of random sines)
            xv = xv + (a_x * np.sin(f_x * t + ph_x))
            yv = yv + (a_y * np.sin(f_y * t + ph_y))

        #image_ = np.zeros((image.shape[1], image.shape[0]))
        image_ = np.copy(image)
        i = 0
        xv_f = np.floor(xv)
        xv_c = np.ceil(xv)
        yv_f = np.floor(yv)
        yv_c = np.ceil(yv)
        for ys, row in enumerate(image):
            for xs, px in enumerate(row):
                if xv_f[i] == xv_c[i] and yv_f[i] == yv_c[i]:
                    self._setpx(image_, int(xs + xv[i]), int(ys + yv[i]), px)

                else:
                    # Bilinear interpolation for fractional pixel values
                    xs_ = int(xs + xv_f[i])
                    ys_ = int(ys + yv_f[i])

                    if xv[i] > 0:
                        xsm = [xs_, xs_+1]
                    elif xv[i] < 0:
                        xsm = [xs_-1, xs_]
                    else:
                        xsm = [xs_, xs_]

                    if yv[i] > 0:
                        ysm = [ys_, ys_+1]
                    elif xv[i] < 0:
                        ysm = [ys_-1, ys_]
                    else:
                        ysm = [ys_, ys_]

                    coords = np.array([
                        [self._getpx(image, xsm[0], ysm[0]), self._getpx(image, xsm[1], ysm[0])],
                        [self._getpx(image, xsm[0], ysm[1]), self._getpx(image, xsm[1], ysm[1])]
                    ])
                    px_ = lambda x, y: (np.array([[1-x, x]]) @ coords @ np.array([[1-y, y]]).T)[0][0]

                    if None not in coords:
                        #print(px_(0, 0))
                        self._setpx(image_, xsm[0], ysm[0], px_(0, 0))
                        self._setpx(image_, xsm[1], ysm[0], px_(1, 0))
                        self._setpx(image_, xsm[0], ysm[1], px_(0, 1))
                        self._setpx(image_, xsm[1], ysm[1], px_(1, 1))

                i = i + 1

        # Gaussian and Poisson noise sum
        for ys, row in enumerate(image_):
            for xs, px in enumerate(row):
                px_ = px + (self.Q_g + self.Q_p * np.sqrt(px)) * np.random.normal(0, 0.01)
                self._setpx(image_, xs, ys, px_)

        return image_, params_ret

    def _gaussian_matrix(self, sigma=1, domain=3, step=5, s=1, phi_s=0, norm=False):
        """Returns a matrix of the specified size containing the 2D Gaussian
        function.

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
        y_ = (1/s) * (-x * np.sin(phi_s) + y * np.cos(phi_s))
        p = (1 / (2*np.pi * sigma**2)) * np.exp(-(x_**2 + y_**2) / 2*(sigma**2))

        if norm == True:
            p = p / np.sum(p)

        return p
