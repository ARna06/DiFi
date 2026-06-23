import numpy as np
from scipy.stats import norm

from .generators import make_grid, BinView, VaRView, ExpectationView, VaRCVaRView, ReverseMomentView, ReverseMomentExpView
from .importance_sampling import snis_estimate, fixed_weight_snis
from .metrics import ks_vs_oracle, w1_vs_oracle, ks_two_samples


def build_views(m_pnl: float, s_pnl: float, grid):
    z25, z75 = norm.ppf([0.25, 0.75], loc=m_pnl, scale=s_pnl)
    z05 = norm.ppf(0.05, loc=m_pnl, scale=s_pnl)
    z95 = norm.ppf(0.95, loc=m_pnl, scale=s_pnl)
    nat_secmom = m_pnl ** 2 + s_pnl ** 2
    # natural CVaR_0.95 (mean of the top 5%) of N(m,s): m + s*phi(z*)/(1-0.95)
    zstar = norm.ppf(0.95)
    nat_cvar95 = m_pnl + s_pnl * norm.pdf(zstar) / 0.05

    views = {}

    # 1) BINS: 3 buckets at q25 and q75, but shift mass from the middle to the tails (natural is .25/.50/.25, we go to .40/.45/.15)
    bv = BinView(edges=np.array([z25, z75]),
                 alpha=np.array([0.40, 0.45, 0.15]))
    views["bins"] = (bv, bv.oracle(grid),
                     "3 buckets at base q25/q75; natural (.25,.50,.25) -> (.40,.45,.15)")

    # 2) VaR: pin the 5% quantile but double the tail probability there.
    vv = VaRView(q=float(z05), level=0.10)
    views["var"] = (vv, vv.oracle(grid),
                    "P(Z<=q05) raised 0.05 -> 0.10 (q fixed at base 5% quantile)")

    # 3) VaR+CVaR: keep the 95% VaR, raise the mean of the top 5% by 12%.
    s_target = float(nat_cvar95 * 1.12)
    vc = VaRCVaRView(q=float(z95), s=s_target, level=0.95)
    views["var_cvar"] = (vc, vc.oracle(grid),
                         f"VaR95 fixed; CVaR95 raised {nat_cvar95:.4f} -> {s_target:.4f}")

    # 4) EXPECTATION: forward exponential tilt, shift mean up by half a P&L sd.
    mu_t = float(m_pnl + 0.5 * s_pnl)
    ev = ExpectationView(mu_target=mu_t)
    views["expectation"] = (ev, ev.oracle(grid),
                            f"mean shifted {m_pnl:.4f} -> {mu_t:.4f} (+0.5 sd)")

    # 5) REVERSE-KL moment (P1, p=2): lower the second moment to 70% of natural.
    m2_t = 0.70 * nat_secmom
    rm = ReverseMomentView(m2_target=float(m2_t))
    views["rkl_moment"] = (rm, rm.oracle(grid),
                           f"E[Z^2] lowered {nat_secmom:.5f} -> {m2_t:.5f} (rational tilt)")

    # 6) REVERSE-KL moment+expectation (P2): keep mean, lower second moment to 80%.
    m2_t2 = 0.80 * nat_secmom
    rme = ReverseMomentExpView(m1_target=float(m_pnl), m2_target=float(m2_t2))
    views["rkl_moment_exp"] = (rme, rme.oracle(grid),
                               f"mean fixed, E[Z^2] lowered {nat_secmom:.5f} -> {m2_t2:.5f}")

    return views



def _constraint_error(name, view, snis_res):
    """How far the reweighted sample is from the imposed target (should be ~0)."""
    r = snis_res.realised
    if name == "bins":
        return float(np.max(np.abs(r["bin_prob"] - view.alpha)))
    if name == "var":
        # 2-bin: realised P(Z<=q) vs level
        return float(abs(r["bin_prob"][0] - view.level))
    if name == "var_cvar":
        return float(max(abs(r["P_below_q"] - view.level),
                         abs(r["cvar"] - view.s) / max(abs(view.s), 1e-9)))
    if name == "expectation":
        return float(abs(r["mean"] - view.mu_target))
    if name == "rkl_moment":
        return float(abs(r["secmom"] - view.m2_target))
    if name == "rkl_moment_exp":
        return float(max(abs(r["mean"] - view.m1_target),
                         abs(r["secmom"] - view.m2_target)))
    return np.nan


def run_main(views, m, R, m_pnl, s_pnl, diff_pnl_pool, seed=0):
    """For each view, average over R replications at sample size m.

    Primary estimator = fixed-weight SNIS (exact optimal tilt), the one our theory governs; 
    baseline = iid from nu*.  We additionally run the
    constraint-matched estimator to report exact in-sample constraint
    satisfaction (cerr) and its own KS."""
    rng = np.random.default_rng(seed)
    out = {}
    Pn = len(diff_pnl_pool)
    for name, (view, oracle, note) in views.items():
        ks_base, ks_true, ks_diff = [], [], []
        ks_true_cm, ks_diff_cm = [], []
        w1_true = []
        rho_hat_true, ess_frac_true = [], []
        cerr_true, cerr_diff = [], []
        for r in range(R):
            # baseline: direct iid from the oracle law
            zb = oracle.sample(m, rng)
            ks_base.append(ks_two_samples(zb, oracle))
            # true-SNIS (fixed optimal weight): iid from true Gaussian P&L + reweight
            zt = rng.normal(m_pnl, s_pnl, size=m)
            sft = fixed_weight_snis(oracle, zt)
            ks_true.append(ks_vs_oracle(sft, oracle))
            w1_true.append(w1_vs_oracle(sft, oracle))
            rho_hat_true.append(sft.rho_hat)
            ess_frac_true.append(sft.ess_frac)
            # deployable constraint-matched on the same draw
            sct = snis_estimate(view, zt)
            ks_true_cm.append(ks_vs_oracle(sct, oracle))
            cerr_true.append(_constraint_error(name, view, sct))
            # diffusion-SNIS: chunk from frozen DDPM P&L pool
            idx = rng.choice(Pn, size=m, replace=False)
            zd = diff_pnl_pool[idx]
            ks_diff.append(ks_vs_oracle(fixed_weight_snis(oracle, zd), oracle))
            scd = snis_estimate(view, zd)
            ks_diff_cm.append(ks_vs_oracle(scd, oracle))
            cerr_diff.append(_constraint_error(name, view, scd))

        out[name] = dict(
            note=note, rho_oracle=oracle.rho, kl=oracle.kl,
            ess_frac_pred=1.0 / oracle.rho,
            ks_base=float(np.mean(ks_base)),
            ks_true=float(np.mean(ks_true)),
            ks_diff=float(np.mean(ks_diff)),
            ks_true_cm=float(np.mean(ks_true_cm)),
            ks_diff_cm=float(np.mean(ks_diff_cm)),
            ks_base_se=float(np.std(ks_base) / np.sqrt(R)),
            ks_true_se=float(np.std(ks_true) / np.sqrt(R)),
            ks_diff_se=float(np.std(ks_diff) / np.sqrt(R)),
            w1_true=float(np.mean(w1_true)),
            rho_hat=float(np.mean(rho_hat_true)),
            ess_frac=float(np.mean(ess_frac_true)),
            cerr_true=float(np.mean(cerr_true)),
            cerr_diff=float(np.mean(cerr_diff)),
            ratio_true_base=float(np.mean(ks_true) / np.mean(ks_base)),
            ratio_diff_true=float(np.mean(ks_diff) / np.mean(ks_true)),
            sqrt_rho=float(np.sqrt(oracle.rho)),
        )
    return out


def run_scaling(views, names, m_grid, R, m_pnl, s_pnl, diff_pnl_pool, seed=1):
    """E[KS] vs m for baseline / true-SNIS / diffusion-SNIS on a subset of views."""
    rng = np.random.default_rng(seed)
    Pn = len(diff_pnl_pool)
    res = {nm: dict(m=list(m_grid), base=[], true=[], diff=[]) for nm in names}
    for nm in names:
        view, oracle, _ = views[nm]
        for m in m_grid:
            kb, kt, kd = [], [], []
            for _ in range(R):
                zb = oracle.sample(m, rng); kb.append(ks_two_samples(zb, oracle))
                zt = rng.normal(m_pnl, s_pnl, size=m)
                kt.append(ks_vs_oracle(fixed_weight_snis(oracle, zt), oracle))
                idx = rng.choice(Pn, size=m, replace=False)
                kd.append(ks_vs_oracle(fixed_weight_snis(oracle, diff_pnl_pool[idx]), oracle))
            res[nm]["base"].append(float(np.mean(kb)))
            res[nm]["true"].append(float(np.mean(kt)))
            res[nm]["diff"].append(float(np.mean(kd)))
    return res


def run_clt_check(view, oracle, m_pnl, s_pnl, m, R, t_points, seed=2):
    """Var[sqrt(m)(Ghat_m(A) - pA)] -> sigma_A^2 = (1-2pA)rho_A + rho*pA^2
       for A = (-inf, t].  Uses the FIXED-weight SNIS estimator (the one Prop 10
       analyses): w = g/f is plugged in, not re-solved on the sample."""
    rng = np.random.default_rng(seed)
    grid = oracle.grid
    f = grid.f
    w_grid = oracle.g / np.maximum(f, 1e-300)       
    rho = oracle.rho
    rows = []
    for t in t_points:
        pA = float(oracle.cdf_at(t))
        inA = (grid.z <= t).astype(float)
        rhoA = float(np.trapezoid(w_grid ** 2 * inA * f, grid.z))  
        sigma2_pred = (1 - 2 * pA) * rhoA + rho * pA ** 2
        ests = []
        for _ in range(R):
            zt = rng.normal(m_pnl, s_pnl, size=m)
            s = fixed_weight_snis(oracle, zt)
            ests.append(s.cdf_at(t))
        ests = np.asarray(ests, float)
        sigma2_emp = float(m * np.var(ests))
        rows.append(dict(t=float(t), pA=pA, rhoA=rhoA,
                         sigma2_pred=float(sigma2_pred), sigma2_emp=sigma2_emp))
    return rows