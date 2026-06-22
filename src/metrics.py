import numpy as np

def ks_vs_oracle(snis, oracle) -> float:
    z = snis.z
    cumw = np.cumsum(snis.w)
    Fo = oracle.cdf_at(z)
    below = np.concatenate([[0.0], cumw[:-1]])     # F_snis just left of each jump
    d_at = np.abs(cumw - Fo)
    d_below = np.abs(below - Fo)
    return float(max(d_at.max(), d_below.max()))


def w1_vs_oracle(snis, oracle) -> float:
    z = oracle.grid.z
    Fs = snis.cdf_at(z)
    Fo = oracle.cdf
    return float(np.trapezoid(np.abs(np.asarray(Fs) - Fo), z))


def ks_two_samples(z, oracle) -> float:
    zs = np.sort(z)
    m = len(zs)
    Fo = oracle.cdf_at(zs)
    upper = np.arange(1, m + 1) / m
    lower = np.arange(0, m) / m
    return float(max(np.abs(upper - Fo).max(), np.abs(lower - Fo).max()))


def kolmogorov_const() -> float:
    return np.sqrt(np.pi / 2) * np.log(2)