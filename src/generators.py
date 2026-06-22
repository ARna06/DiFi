import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import brentq, fsolve
from scipy.stats import norm

@dataclass
class Grid1D:
    z: np.ndarray
    f: np.ndarray          
    m: float
    s: float
 
    def E(self, h: np.ndarray) -> float:
        return float(np.trapezoid(h * self.f, self.z))
 
 
def make_grid(m: float, s: float, npts: int = 400001, span: float = 14.0) -> Grid1D:
    z = np.linspace(m - span * s, m + span * s, npts)
    f = norm.pdf(z, loc=m, scale=s)
    g = Grid1D(z=z, f=f, m=m, s=s)
    g.f = f / g.E(np.ones_like(z))
    return g
 

@dataclass
class Oracle:
    name: str
    family: str # "forward" (KL(Y||X)) or "reverse" (KL(X||Y))
    grid: Grid1D
    u: np.ndarray = None # unnormalised weight on grid (optional)
    g_override: np.ndarray = None  # tilted density on grid (optional, already ~normalised)
    params: dict = field(default_factory=dict)
 
    Zu: float = 0.0
    g: np.ndarray = None        
    cdf: np.ndarray = None
    rho: float = 0.0
    kl: float = 0.0
 
    def __post_init__(self):
        gr = self.grid
        if self.g_override is not None:
            self.g = self.g_override / np.trapezoid(self.g_override, gr.z)
        else:
            self.Zu = gr.E(self.u)
            self.g = self.u * gr.f / self.Zu
        c = np.concatenate([[0.0], np.cumsum(0.5 * (self.g[1:] + self.g[:-1]) * np.diff(gr.z))])
        self.cdf = c / c[-1]
  
        f = np.maximum(gr.f, 1e-300)
        self.rho = float(np.trapezoid(self.g ** 2 / f, gr.z))          # 1 + chi^2(nu||mu)
        gg = np.maximum(self.g, 1e-300)
        if self.family == "forward":           # KL(nu||mu) = int g log(g/f)
            self.kl = float(np.trapezoid(self.g * np.log(gg / f), gr.z))
        elif self.family == "backward":                                   # KL(mu||nu) = int f log(f/g)
            self.kl = float(np.trapezoid(gr.f * np.log(f / gg), gr.z))
        else:
            raise NotImplementedError(f"Unknown family {self.family}")
 
    def cdf_at(self, t):
        return np.interp(t, self.grid.z, self.cdf)
 
    def sample(self, m: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(m)
        return np.interp(u, self.cdf, self.grid.z)
 
    # weight as a callable on arbitrary P&L values
    def weight_fn(self):
        raise NotImplementedError
    
@dataclass
class BinView:
    edges: np.ndarray           # interior edges, sorted ascending -> L+1 bins
    alpha: np.ndarray           # target probabilities, len = L+1, sum 1
    kind: str = "bins"
 
    def bin_index(self, z):
        return np.searchsorted(self.edges, z, side="right")  # 0..L
 
    def oracle(self, grid: Grid1D) -> Oracle:
        idx = self.bin_index(grid.z)
        beta = np.array([grid.E((idx == i).astype(float)) for i in range(len(self.alpha))])
        u = self.alpha[idx] / beta[idx]
        return Oracle(self.kind, "forward", grid, u,
                      params={"alpha": self.alpha, "beta": beta, "sum_alpha2_over_beta": float(np.sum(self.alpha ** 2 / beta))})
    
@dataclass
class VaRView:
    q: float                    # VaR target (the alpha-quantile of the P&L)
    level: float                # alpha in (0,1): P(Z <= q) = alpha
 
    def oracle(self, grid: Grid1D) -> Oracle:
        bv = BinView(edges=np.array([self.q]),
                     alpha=np.array([self.level, 1.0 - self.level]), kind="var")
        o = bv.oracle(grid)
        o.params["q"] = self.q
        o.params["level"] = self.level
        return o
 
 
@dataclass
class ExpectationView:
    mu_target: float            # forward-KL exponential tilt to a target mean
 
    def oracle(self, grid: Grid1D) -> Oracle:
        def mean_of(theta):
            w = np.exp(theta * (grid.z - grid.m))     # shifted for numerical stability
            return grid.E(w * grid.z) / grid.E(w)
        f = lambda th: mean_of(th) - self.mu_target
        lo, hi = -50.0 / grid.s, 50.0 / grid.s
        theta = brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)
        u = np.exp(theta * (grid.z - grid.m))
        return Oracle("expectation", "forward", grid, u, 
                      params={"theta": theta, "mu_target": self.mu_target})
 
 
@dataclass
class VaRCVaRView:
    q: float                    # VaR target (alpha-quantile)
    s: float                    # CVaR target (mean of the upper (1-alpha) tail), s > q
    level: float                # alpha
 
    def _kernel(self, grid: Grid1D, theta: float) -> np.ndarray:
        return np.exp(-((grid.z - grid.m) - theta * grid.s ** 2) ** 2 / (2 * grid.s ** 2))
 
    def _solve_theta(self, grid: Grid1D, region: np.ndarray) -> float:
        def root(theta):
            k = self._kernel(grid, theta) * region
            return float(np.trapezoid((grid.z - self.s) * k, grid.z))
        lo, hi = -30.0 / grid.s, 30.0 / grid.s
        ths = np.linspace(lo, hi, 600)
        vals = np.array([root(t) for t in ths])
        up = np.where((vals[:-1] < 0) & (vals[1:] >= 0))[0]
        if up.size == 0:
            return np.nan
        i = up[0]
        return brentq(root, ths[i], ths[i + 1], xtol=1e-12, rtol=1e-12)
 
    def _candidate(self, grid: Grid1D, i: int):
        a = self.level
        if i == 2:                            
            tilt_region = (grid.z > self.q).astype(float) 
            flat_region = (grid.z <= self.q).astype(float)
        else:
            tilt_region = (grid.z <= self.q).astype(float)
            flat_region = (grid.z > self.q).astype(float)
        theta = self._solve_theta(grid, tilt_region)
        if not np.isfinite(theta):
            return None
        k = self._kernel(grid, theta) * tilt_region
        Ik = np.trapezoid(k, grid.z)
        flat_mass = grid.E(flat_region)
        if Ik <= 0 or flat_mass <= 0:
            return None
        
        g = a * (flat_region * grid.f) / flat_mass + (1.0 - a) * k / Ik
        o = Oracle("var_cvar", "forward", grid, g_override=g,
                   params={"branch": i, "theta": theta, "q": self.q,
                           "s": self.s, "level": a})
        below = grid.z <= self.q
        upper = grid.z > self.q
        o.params["P_below_q"] = float(np.trapezoid(below * o.g, grid.z))
        mass_up = float(np.trapezoid(upper * o.g, grid.z))
        o.params["cvar_realised"] = (float(np.trapezoid(grid.z * upper * o.g, grid.z) / mass_up)
                                     if mass_up > 0 else np.nan)
        return o
 
    def oracle(self, grid: Grid1D) -> Oracle:
        cands = [c for c in (self._candidate(grid, i) for i in (2, 1)) if c is not None]
        if not cands:
            raise ValueError("VaR+CVaR: no feasible branch (check q, s ordering and level).")
 
        def feasible(o):
            return (abs(o.params["P_below_q"] - self.level) < 1e-3 and
                    abs(o.params["cvar_realised"] - self.s) < 1e-2 * abs(self.s) + 1e-3)
 
        feas = [o for o in cands if feasible(o)]
        pool = feas if feas else cands
        return min(pool, key=lambda o: o.kl)
 
 
@dataclass
class ReverseMomentView:
    m2_target: float
 
    def oracle(self, grid: Grid1D) -> Oracle:
        def secmom(theta):
            w = 1.0 / (1.0 + theta * grid.z ** 2)
            return grid.E(w * grid.z ** 2) / grid.E(w)
        nat = grid.E(grid.z ** 2)
        if self.m2_target >= nat:
            raise ValueError(f"reverse-KL rational tilt can only LOWER the 2nd moment "
                             f"(natural={nat:.4f}); target {self.m2_target:.4f} is infeasible on R.")
        f = lambda th: secmom(th) - self.m2_target
        theta = brentq(f, 0.0, 1e6, xtol=1e-14, rtol=1e-12)
        u = 1.0 / (1.0 + theta * grid.z ** 2)
        return Oracle("rkl_moment", "reverse", grid, u,
                      params={"theta": theta, "m2_target": self.m2_target, "m2_natural": nat})
 
 
@dataclass
class ReverseMomentExpView:
    m1_target: float
    m2_target: float
 
    def oracle(self, grid: Grid1D) -> Oracle:
        z = grid.z
 
        def L_of(params):
            d, e = params
            return 1.0 + d * (z - self.m1_target) + e * (z ** 2 - self.m2_target)
 
        def resid(params):
            L = L_of(params)
            if np.any(L <= 1e-9):
                return [1e6, 1e6]
            w = 1.0 / L
            Ew = grid.E(w)
            return [grid.E(w * z) / Ew - self.m1_target,
                    grid.E(w * z ** 2) / Ew - self.m2_target]
 
        sol, info, ier, msg = fsolve(resid, x0=[0.0, 0.0], full_output=True, xtol=1e-12)
        L = L_of(sol)
        if np.any(L <= 0):
            raise ValueError("P2 tilt lost positivity; target outside feasible cone.")
        u = 1.0 / L
        return Oracle("rkl_moment_exp", "reverse", grid, u,
                      params={"delta": sol[0], "eta": sol[1], "m1_target": self.m1_target, "m2_target": self.m2_target, "resid": float(np.hypot(*resid(sol)))})