
from .utils import gaussian_kernel, compute_curvature
import numpy as np


class CurvatureScaleSpace(object):
    """ Curvature Scale Space

    A simple curvature scale space implementation based on
    Mohkatarian et. al. paper. Full algorithm detailed in
    Okal msc thesis

    """

    def __init__(self):
        pass

    def find_zero_crossings(self, kappa):
        """ find_zero_crossings(kappa)
        Locate the zero crossing points of the curvature signal kappa(t)
        """

        crossings = []

        for i in range(0, kappa.size - 2):
            if (kappa[i] < 0.0 and kappa[i + 1] > 0.0) or (kappa[i] > 0.0 and kappa[i + 1] < 0.0):
                crossings.append(i)

        return crossings


    def generate_css(self, curve, max_sigma, step_sigma):
        """ generate_css(curve, max_sigma, step_sigma)
        Generates a CSS image representation by repetatively smoothing the initial curve L_0 with increasing sigma
        """

        cols = curve[0, :].size
        rows = max_sigma // step_sigma
        css = np.zeros(shape=(int(rows), cols))

        srange = np.linspace(1, int(max_sigma) - 1, int(rows))
        for i, sigma in enumerate(srange):
            kappa = compute_curvature(curve, sigma)

            # find interest points
            xs = self.find_zero_crossings(kappa)

            # save the interest points
            if len(xs) > 0 and sigma < max_sigma - 1:
                for c in xs:
                    css[i, c] = 1  # change to any positive

            else:
                return css


    def generate_visual_css(self, rawcss, closeness):
        """ generate_visual_css(rawcss, closeness)
        Generate a 1D signal that can be plotted to depict the CSS by taking
        column maximums. Further checks for close interest points and nicely
        smoothes them with weighted moving average
        """

        flat_signal = np.amax(rawcss, axis=0)

        # minor smoothing via moving averages
        window = closeness
        weights = gaussian_kernel(window, 0, window, False)  # gaussian weights
        sig = np.convolve(flat_signal, weights)[window - 1:-(window - 1)]

        return sig

    def generate_eigen_css(self, rawcss, return_all=False):
        """ generate_eigen_css(rawcss, return_all)
        Generates Eigen-CSS features
        """
        rowsum = np.sum(rawcss, axis=0)
        csum = np.sum(rawcss, axis=1)

        # hack to trim c
        colsum = csum[0:rowsum.size]

        freq = np.fft.fft(rowsum)
        mag = abs(freq)

        tilde_rowsum = np.fft.ifft(mag)

        feature = np.concatenate([tilde_rowsum, colsum], axis=0)

        if not return_all:
            return feature
        else:
            return feature, rowsum, tilde_rowsum, colsum
