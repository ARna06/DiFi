import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import brentq, fsolve
 
from .generators import (BinView, VaRView, ExpectationView, VaRCVaRView, ReverseMomentView, ReverseMomentExpView)
 
 
@dataclass
class SNISResult:
    z: np.ndarray              
    w: np.ndarray            
    params: dict
    rho_hat: float
    ess: float
    ess_frac: float
    realised: dict = field(default_factory=dict)
 
    def cdf_at(self, t):
        cw = np.cumsum(self.w)
        idx = np.searchsorted(self.z, t, side="right") - 1
        t = np.atleast_1d(t)
        out = np.where(np.asarray(idx) < 0, 0.0, cw[np.clip(idx, 0, len(cw) - 1)])
        return out if out.size > 1 else float(out)
 
 
def _finalize(z, u, params, realised):
    order = np.argsort(z)
    z = z[order]
    u = u[order]
    s = u.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("degenerate SNIS weights")
    w = u / s
    ubar = u.mean()
    rho_hat = float(np.mean((u / ubar) ** 2))
    ess = float(1.0 / np.sum(w ** 2))
    return SNISResult(z=z, w=w, params=params, rho_hat=rho_hat, ess=ess, ess_frac=ess / len(z), realised=realised)
 
 
def _snis_bins(view: BinView, z):
    idx = view.bin_index(z)
    L1 = len(view.alpha)
    counts = np.array([np.sum(idx == i) for i in range(L1)])
    u = np.zeros_like(z, dtype=float)
    for i in range(L1):
        if counts[i] > 0:
            u[idx == i] = view.alpha[i] / counts[i]
    w_tmp = u / u.sum()
    realised = {"bin_prob": np.array([w_tmp[idx == i].sum() for i in range(L1)]),
                "empty_bins": int(np.sum(counts == 0))}
    return _finalize(z, u, {"alpha": view.alpha, "counts": counts}, realised)
 
 
def _snis_expectation(view: ExpectationView, z):
    s = z.std() + 1e-12
    def mean_of(theta):
        e = np.exp(theta * (z - z.mean()))
        return np.sum(e * z) / np.sum(e)
    f = lambda th: mean_of(th) - view.mu_target
    lo, hi = -60.0 / s, 60.0 / s
    if f(lo) * f(hi) > 0:                      
        theta = lo if abs(f(lo)) < abs(f(hi)) else hi
    else:
        theta = brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)
    u = np.exp(theta * (z - z.mean()))
    return _finalize(z, u, {"theta": theta},
                     {"mean": float(np.sum(u * z) / u.sum())})
 
 
def _snis_var_cvar(view: VaRCVaRView, z, branch: int = 2):
    a, q, s = view.level, view.q, view.s
    if branch == 2:
        tilt = z > q
    else:
        tilt = z <= q
    flat = ~tilt
    zt = z[tilt]
    if zt.size == 0 or flat.sum() == 0:
        raise ValueError("VaR+CVaR SNIS: a region has no samples")
    zr = zt - zt.mean()
    def root(theta):
        e = np.exp(theta * zr)
        return np.sum((zt - s) * e)
    ths = np.linspace(-30.0 / (z.std() + 1e-9), 30.0 / (z.std() + 1e-9), 400)
    vals = np.array([root(t) for t in ths])
    up = np.where((vals[:-1] < 0) & (vals[1:] >= 0))[0]
    theta = brentq(root, ths[up[0]], ths[up[0] + 1], xtol=1e-12) if up.size else 0.0
    u = np.zeros_like(z, dtype=float)
    u[flat] = a / flat.sum()
    e = np.exp(theta * (z[tilt] - zt.mean()))
    u[tilt] = (1.0 - a) * e / e.sum()
    w = u / u.sum()
    cvar = float(np.sum(z[tilt] * w[tilt]) / w[tilt].sum()) if w[tilt].sum() > 0 else np.nan
    return _finalize(z, u, {"theta": theta, "branch": branch},
                     {"P_below_q": float(w[z <= q].sum()), "cvar": cvar})
 
 
def _snis_rkl_moment(view: ReverseMomentView, z):
    def secmom(theta):
        u = 1.0 / (1.0 + theta * z ** 2)
        return np.sum(u * z ** 2) / np.sum(u)
    nat = np.mean(z ** 2)
    if view.m2_target >= nat:
        theta = 0.0
    else:
        f = lambda th: secmom(th) - view.m2_target
        hi = 1e6
        theta = brentq(f, 0.0, hi, xtol=1e-14, rtol=1e-12)
    u = 1.0 / (1.0 + theta * z ** 2)
    return _finalize(z, u, {"theta": theta},
                     {"secmom": float(np.sum(u * z ** 2) / u.sum())})
 
 
def _snis_rkl_moment_exp(view: ReverseMomentExpView, z):
    m1t, m2t = view.m1_target, view.m2_target
    def resid(p):
        d, e = p
        L = 1.0 + d * (z - m1t) + e * (z ** 2 - m2t)
        if np.any(L <= 1e-9):
            return [1e6, 1e6]
        u = 1.0 / L
        U = u.sum()
        return [np.sum(u * z) / U - m1t, np.sum(u * z ** 2) / U - m2t]
    sol = fsolve(resid, x0=[0.0, 0.0], xtol=1e-12)
    d, e = sol
    L = 1.0 + d * (z - m1t) + e * (z ** 2 - m2t)
    u = 1.0 / np.maximum(L, 1e-12)
    U = u.sum()
    return _finalize(z, u, {"delta": d, "eta": e},
                     {"mean": float(np.sum(u * z) / U),
                      "secmom": float(np.sum(u * z ** 2) / U)})
 
 
def fixed_weight_snis(oracle, z) -> SNISResult:

    z = np.asarray(z, dtype=float)
    gr = oracle.grid
    w_grid = oracle.g / np.maximum(gr.f, 1e-300)        # exact optimal RN derivative
    u = np.interp(z, gr.z, w_grid)                      # evaluate at the sample points
    u = np.maximum(u, 0.0)
    return _finalize(z, u, {"mode": "fixed_weight"}, {})
 
 
def snis_estimate(view, z, **kw) -> SNISResult:
    z = np.asarray(z, dtype=float)
    if isinstance(view, VaRView):
        view = BinView(edges=np.array([view.q]),
                       alpha=np.array([view.level, 1 - view.level]), kind="var")
    if isinstance(view, BinView):
        return _snis_bins(view, z)
    if isinstance(view, ExpectationView):
        return _snis_expectation(view, z)
    if isinstance(view, VaRCVaRView):
        return _snis_var_cvar(view, z, **kw)
    if isinstance(view, ReverseMomentView):
        return _snis_rkl_moment(view, z)
    if isinstance(view, ReverseMomentExpView):
        return _snis_rkl_moment_exp(view, z)
    raise TypeError(type(view))