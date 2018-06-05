import numpy as np

class MuEncoder(object):
    def __init__(self, datarange, mu=255):
        self.mu = mu
        self.datarange = datarange

    def normalize(self, x):
        return ((np.float32(x) - self.datarange[0]) / self.datarange[1] - 0.5) * 2

    def expand(self, x):
        return (x / 2 + 0.5) * self.datarange[1] + self.datarange[0]

    def encode(self, x):
        x = self.normalize(x)
        return np.sign(x) * np.log(1 + self.mu * np.abs(x)) / np.log(1 + self.mu)

    def decode(self, x):
        x = np.sign(x) * self.mu**-1 * ((1 + self.mu)**np.abs(x) - 1)
        return self.expand(x)
