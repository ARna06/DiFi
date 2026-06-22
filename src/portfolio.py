import numpy as np
from dataclasses import dataclass

@dataclass
class GaussianPortfolio:
    mu: np.ndarray       
    Sigma: np.ndarray     
    w: np.ndarray           
 
    @property
    def n(self) -> int:
        return self.mu.shape[0]
 
    @property
    def pnl_mean(self) -> float:
        return float(self.w @ self.mu)
 
    @property
    def pnl_std(self) -> float:
        return float(np.sqrt(self.w @ self.Sigma @ self.w))
 
    def sample_assets(self, m: int, rng: np.random.Generator) -> np.ndarray:
        L = np.linalg.cholesky(self.Sigma)
        z = rng.standard_normal((m, self.n))
        return self.mu[None, :] + z @ L.T
 
    def pnl(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w
    
def default_portfolio(n: int = 4, seed: int = 0) -> GaussianPortfolio:
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 0.02, size=n)                 
    A = rng.normal(0, 1, size=(n, n))
    C = A @ A.T
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)
    volatilities = rng.uniform(0.10, 0.25, size=n)
    Sigma = corr * np.outer(volatilities, volatilities)
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(n)
    w = rng.normal(0, 1, size=n)
    w = w / np.sum(np.abs(w))                           
    return GaussianPortfolio(mu=mu, Sigma=Sigma, w=w)