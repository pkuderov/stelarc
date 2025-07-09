import numpy as np
import torch


class SoftClip:
    """
    Empirically constructed soft-clip func.
    It starts the clipping a bit before the threshold and decays very fast.
    """

    def __init__(self, eps, start_from):
        self.eps = eps
        self.start_from = start_from
        self.max_hardness = 24
        self.hardness = 1.0 + self.max_hardness * (1.0 - self.eps)

    def __call__(self, x):
        eps, alpha, beta = self.eps, self.start_from, self.hardness
        z = torch.clamp(alpha * eps + torch.exp(-torch.abs(x)), max=1.0)
        return torch.pow(z, beta), z

    def plot(self):
        import matplotlib.pyplot as plt

        xs = np.linspace(0.00001, 1.0, 100)
        ys, zs = self(torch.log(torch.from_numpy(xs)))
        plt.plot(xs, ys)
        plt.plot(xs, zs)
        plt.show()


def test_soft_clip():
    SoftClip(eps=0.1, start_from=0.75).plot()


if __name__ == '__main__':
    test_soft_clip()
