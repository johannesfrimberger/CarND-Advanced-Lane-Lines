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
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = []
        # y values for detected line pixels
        self.ally = []

    def update(self, x, y):
        self.current_fit = np.polyfit(x, y, 2)

        if len(self.allx) < self.n_samples:
            self.allx.append(x)
            self.ally.append(y)
        else:
            self.allx.pop(0)
            self.ally.pop(0)

            self.allx.append(x)
            self.ally.append(y)

        data_x = self.allx[0]
        data_y = self.ally[0]

        for n in range(1, len(self.allx)):
            data_x = np.concatenate((data_x, self.allx[n]), axis=0)
            data_y = np.concatenate((data_y, self.ally[n]), axis=0)

        self.best_fit = np.polyfit(data_x, data_y, 2)