"""
Lineage dynamics for Lurker Ecology (Lopin 2026), Section 6.8.

A minimal birth-and-death model of the number of lurker lineages L in the Galaxy and the number n
coexisting in a representative planetary system. All rates are per Myr. Parameters are illustrative;
the figure shows which qualitative regimes are possible, not an estimate for our system.

Model
-----
Galaxy:   civilizations reach the end of their suppression window at rate lam_civ. Each survives with
          probability Q(n) evaluated at the representative system's n: Q(0)=1, Q(1)=0, Q(n>=2)=(K/B)^(n/(n-1)).
          Each survivor founds a lineage (L += 1). Lineages split by replication error at rate mu per lineage.
Local:    lineages absent from the representative system arrive at rate alpha per absent lineage
          (stellar-encounter delivery). Local footholds decay at rate d per lineage. Pairs of lineages
          fight at rate kappa per pair, suppressed by (1-delta)^(n-2) as the number of watchers grows;
          each fight removes one local foothold (the lineage survives elsewhere).
Freeze:   optional. K(n) rises with watchers, B(n) falls as the marginal rival matters less; where
          K(n) >= B(n) nobody strikes and Q = 1.

Usage: python lineage_dynamics.py   -> writes lineage_dynamics.pdf/.png and prints summary numbers.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_MYR = 10000          # 10 Gyr
DT = 1.0

def q_fixed(n, kb=0.5):
    if n == 0: return 1.0
    if n == 1: return 0.0
    return kb ** (n / (n - 1))

def q_freeze(n, v_over_b=1.5, delta=0.1, n_half=5.0):
    """K(n)/B(n) = (V/B0) * [1-(1-delta)^(n-1)] * (1+(n-2)/n_half): exposure rises with watchers,
    the marginal benefit of removing one more rival falls. Q=1 once K>=B (nobody strikes)."""
    if n == 0: return 1.0
    if n == 1: return 0.0
    ratio = v_over_b * (1 - (1 - delta) ** (n - 1)) * (1 + (n - 2) / n_half)
    if ratio >= 1.0: return 1.0
    return ratio ** (n / (n - 1))

def simulate(lam_civ=0.1, mu=1e-4, alpha=0.05, d=0.01, kappa=0.02, delta=0.3,
             kb=0.5, freeze=False, n_half=5.0, v_over_b=1.5, seed=0, T=T_MYR, dt=DT):
    rng = np.random.default_rng(seed)
    steps = int(T / dt)
    L = 0; n = 0
    n_arr = np.zeros(steps, int); L_arr = np.zeros(steps, int)
    civ_t = []; civ_s = []
    t_escape = None
    qf = (lambda m: q_freeze(m, v_over_b, delta, n_half)) if freeze else (lambda m: q_fixed(m, kb))
    for i in range(steps):
        t = i * dt
        # civilizations reaching the end of their window
        for _ in range(rng.poisson(lam_civ * dt)):
            s = rng.random() < qf(n)
            civ_t.append(t); civ_s.append(s)
            if s: L += 1
        # replication error splits lineages
        L += rng.poisson(mu * L * dt)
        # arrivals of absent lineages
        absent = max(L - n, 0)
        n += min(rng.poisson(alpha * absent * dt), absent)
        # local decay
        n -= rng.binomial(n, min(d * dt, 1.0)) if n > 0 else 0
        # duels, suppressed by watchers
        if n >= 2:
            rate = kappa * n * (n - 1) / 2 * (1 - delta) ** (n - 2)
            n -= min(rng.poisson(rate * dt), n)
        n = max(min(n, L), 0)
        if t_escape is None and n >= 2: t_escape = t
        n_arr[i] = n; L_arr[i] = L
    return dict(t=np.arange(steps) * dt, n=n_arr, L=L_arr,
                civ_t=np.array(civ_t), civ_s=np.array(civ_s, bool), t_escape=t_escape)

def survival_by_epoch(runs, bins_gyr=10):
    edges = np.linspace(0, T_MYR, bins_gyr + 1)
    tot = np.zeros(bins_gyr); sur = np.zeros(bins_gyr)
    for r in runs:
        idx = np.clip(np.digitize(r["civ_t"], edges) - 1, 0, bins_gyr - 1)
        for k in range(bins_gyr):
            m = idx == k
            tot[k] += m.sum(); sur[k] += r["civ_s"][m].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (edges[:-1] + edges[1:]) / 2 / 1000, np.where(tot > 0, sur / tot, np.nan)

if __name__ == "__main__":
    NRUN = 30
    scenarios = {
        "sparse: 100 civilizations / 10 Gyr":    dict(lam_civ=0.01),
        "moderate: 1,000 / 10 Gyr":              dict(lam_civ=0.1),
        "abundant: 10,000 / 10 Gyr":             dict(lam_civ=1.0),
    }
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) local lineage count n(t); (b) survival fraction by epoch
    summary = {}
    for (label, kw), c in zip(scenarios.items(), colors):
        runs = [simulate(seed=s, **kw) for s in range(NRUN)]
        N = np.array([r["n"] for r in runs])
        t = runs[0]["t"] / 1000
        med = np.median(N, 0); lo = np.percentile(N, 10, 0); hi = np.percentile(N, 90, 0)
        ax[0, 0].plot(t, med, color=c, label=label); ax[0, 0].fill_between(t, lo, hi, color=c, alpha=0.15)
        x, y = survival_by_epoch(runs)
        ax[0, 1].plot(x, y, "o-", color=c, label=label)
        esc = [r["t_escape"] for r in runs if r["t_escape"] is not None]
        summary[label] = dict(n_final=float(np.median(N[:, -1])), L_final=float(np.median([r["L"][-1] for r in runs])),
                              escape_median_Myr=float(np.median(esc)) if esc else None, escaped=len(esc),
                              surv_first_Gyr=float(np.nanmean(y[:2])), surv_last_Gyr=float(np.nanmean(y[-2:])))
    ax[0, 0].set(xlabel="time (Gyr)", ylabel="lineages coexisting locally, n(t)", title="(a) Local lineage count", yscale="log", ylim=(0.8, 2e4))
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
    ax[0, 1].set(xlabel="epoch (Gyr)", ylabel="fraction of civilizations surviving window", ylim=(0, 1.05),
                 title="(b) Lateness protects?"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)

    # (c) escape from the one-lineage trap. Trap binds when probes spread faster than civilizations
    #     arise (alpha large) and footholds outlive the gaps (d small); then splitting is the only exit.
    mus = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    for d_, ls, lab in [(0.0, "s-", "no foothold decay"), (0.002, "o--", "foothold lifetime 500 Myr")]:
        med_esc = []; frac_esc = []
        for mu in mus:
            runs = [simulate(lam_civ=0.01, mu=mu, alpha=1.0, d=d_, seed=100 + s) for s in range(NRUN)]
            esc = [r["t_escape"] for r in runs if r["t_escape"] is not None]
            med_esc.append(np.median(esc) / 1000 if esc else np.nan); frac_esc.append(len(esc) / NRUN)
        ax[1, 0].semilogx(mus, med_esc, ls, color="k", label=lab)
        for m, f, y in zip(mus, frac_esc, med_esc):
            if not np.isnan(y): ax[1, 0].annotate(f"{int(f*100)}%", (m, y), textcoords="offset points", xytext=(4, 5), fontsize=7)
        print(f"trap escape ({lab}) vs mu:", dict(zip(mus, zip([round(x,2) for x in med_esc], frac_esc))))
    ax[1, 0].set(xlabel="lineage splitting rate mu (per lineage per Myr)", ylabel="median time to n >= 2 (Gyr)",
                 title="(c) Leaving the one-lineage trap (100 civilizations / 10 Gyr, fast spread)")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)

    # (d) frozen regime: Q(n) with fixed K/B vs K rising and B falling
    ns = np.arange(1, 31)
    ax[1, 1].plot(ns, [q_fixed(m, 0.5) for m in ns], "o-", color="#1f77b4", label="fixed K/B = 0.5 (Eq. 5)")
    ax[1, 1].plot(ns, [q_fixed(m, 0.15) for m in ns], "o-", color="#9ecae1", label="fixed K/B = 0.15")
    for nh, ls in [(5, "--"), (10, ":")]:
        ax[1, 1].plot(ns, [q_freeze(m, 1.5, 0.1, nh) for m in ns], ls, color="#d62728",
                      label=f"V/B=1.5, delta=0.1, n_half={nh}: freezes")
    ax[1, 1].plot(ns, [q_freeze(m, 0.8, 0.1, 1e12) for m in ns], "-.", color="#ff9896", label="V/B=0.8, B constant: never freezes")
    ax[1, 1].set(xlabel="lineages present, n", ylabel="survival probability Q(n)", ylim=(0, 1.05),
                 title="(d) Crowding: probabilistic vs frozen"); ax[1, 1].legend(fontsize=7); ax[1, 1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig("lineage_dynamics.pdf"); fig.savefig("lineage_dynamics.png", dpi=150)
    print("scenario summaries (medians over %d runs):" % NRUN)
    for k, v in summary.items(): print(" ", k, v)
    print("trap escape vs mu:", dict(zip(mus, zip(med_esc, frac_esc))))
    # freeze threshold
    for nh in (5, 10):
        nstar = next(m for m in range(2, 200) if q_freeze(m, 1.5, 0.1, nh) >= 1.0)
        print(f"freeze: Q reaches 1 at n* = {nstar} for V/B=1.5, delta=0.1, n_half={nh}")
