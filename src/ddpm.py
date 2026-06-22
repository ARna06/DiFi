import numpy as np
import torch
import torch.nn as nn


class _SinusoidalTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):                         
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        a = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class _EpsNet(nn.Module):
    def __init__(self, n, hidden=128, temb=32):
        super().__init__()
        self.temb = _SinusoidalTime(temb)
        self.net = nn.Sequential(
            nn.Linear(n + temb, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], dim=-1))


class DDPM:
    def __init__(self, n, T=200, beta0=1e-4, beta1=2e-2, hidden=128, seed=0, device="cuda"):
        torch.manual_seed(seed)
        self.n, self.T, self.device = n, T, device
        self.net = _EpsNet(n, hidden).to(device)
        betas = torch.linspace(beta0, beta1, T, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.abar = torch.cumprod(self.alphas, dim=0)
        self.mean_ = None
        self.std_ = None

    def _fit_scaler(self, X):
        self.mean_ = X.mean(0, keepdim=True)
        self.std_ = X.std(0, keepdim=True) + 1e-8

    def _std(self, X):
        return (X - self.mean_) / self.std_

    def _unstd(self, X):
        return X * self.std_ + self.mean_

    def train(self, sample_fn, steps=6000, batch=4096, lr=2e-3, verbose=True):
        """sample_fn(m, rng) -> (m,n) numpy array of true asset draws (infinite data)."""
        rng = np.random.default_rng(12345)
        X0 = torch.tensor(sample_fn(20000, rng), dtype=torch.float32, device=self.device)
        self._fit_scaler(X0)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        losses = []
        for it in range(steps):
            Xnp = sample_fn(batch, rng)
            x0 = self._std(torch.tensor(Xnp, dtype=torch.float32, device=self.device))
            t = torch.randint(0, self.T, (batch,), device=self.device)
            ab = self.abar[t][:, None]
            eps = torch.randn_like(x0)
            xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
            pred = self.net(xt, t.float() / self.T)
            loss = ((pred - eps) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
            if verbose and (it % 1000 == 0 or it == steps - 1):
                print(f"  ddpm step {it:5d}  loss {np.mean(losses[-200:]):.5f}")
        return losses

    @torch.no_grad()
    def sample(self, m, seed=0):
        torch.manual_seed(seed)
        x = torch.randn(m, self.n, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((m,), i, device=self.device)
            ab = self.abar[i]
            eps = self.net(x, t.float() / self.T)
            mean = (x - self.betas[i] / torch.sqrt(1 - ab) * eps) / torch.sqrt(self.alphas[i])
            if i > 0:
                x = mean + torch.sqrt(self.betas[i]) * torch.randn_like(x)
            else:
                x = mean
        return self._unstd(x).cpu().numpy()