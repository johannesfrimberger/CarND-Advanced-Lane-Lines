import numpy as np


class Line():
    """
    Class to hold and
    """
    def __init__(self):
        """

        """
        #
        self.n_samples = 10

        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = []
        # y values for detected line pixels
        self.ally = []

    def update(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """

        current_fit = np.polyfit(x, y, 2)

        if self.detected:
            alpha = 0.05
            self.best_fit = (1.0 - alpha) * self.best_fit + alpha * current_fit
        else:
            self.best_fit = current_fit

        self.current_fit = current_fit
        self.detected = True
