import torch


def sine_transfor_box(z):
    lb = torch.zeros_like(z)
    ub = torch.ones_like(z)

    z_bounded = lb + (torch.sin(z) + 1) * (ub - lb) / 2
    return z_bounded
