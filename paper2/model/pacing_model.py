"""
Pacing problem for Treading Carefully (Lopin 2026), Section 5.3.

A civilization's progress toward credible retaliation is x in [0,1]. Watchers act on an estimate of x.
Hazard (strike rate per year):
  state regime : h = h0 + h1 * 1[x_hat > x_c],   x_hat(t) = x(t - tau)   (inference lag tau)
  rate regime  : h = h0 + h2 * xdot^2                                     (visible activity)
  mixed        : both terms
Paths are two-phase: creep at speed v1 from 0 to x_c, sprint at speed v2 from x_c to 1.
Survival P = exp(-integral h dt) over the path (strikes after x=1 fail).
All parameters illustrative. Writes fig_pacing.pdf/.png and prints the sprint condition.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h0, h1, h2, x_c = 1e-4, 0.05, 1.0, 0.9   # per yr, per yr, (yr), dimensionless

def H_state(v1, v2, tau):
    """Integrated hazard, state regime. Lag makes the final tau years of the path free."""
    T1 = x_c / v1; Ts = (1 - x_c) / v2
    return h0 * (T1 + Ts) + h1 * max(Ts - tau, 0.0)

def H_rate(v1, v2):
    T1 = x_c / v1; Ts = (1 - x_c) / v2
    return h0 * (T1 + Ts) + h2 * (v1**2 * T1 + v2**2 * Ts)

def H_mixed(v1, v2, tau):
    Ts = (1 - x_c) / v2
    return H_rate(v1, v2) + h1 * max(Ts - tau, 0.0)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))

# (a) state regime: survival vs sprint duration, creep fixed at a century pace
v1 = 0.01
Ts = np.linspace(0.5, 30, 300); v2 = (1 - x_c) / Ts
for tau, c in [(2, "#1f77b4"), (5, "#d62728"), (10, "#2ca02c")]:
    ax[0].plot(Ts, [np.exp(-H_state(v1, v, tau)) for v in v2], color=c, label=f"inference lag = {tau} yr")
    ax[0].axvline(tau, color=c, ls=":", lw=1)
ax[0].set(xlabel="duration of final sprint (yr)", ylabel="survival probability", ylim=(0, 1.02),
          title="(a) Watchers infer from state: sprint inside the lag"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

# (b) rate regime: survival vs constant speed; interior optimum at v* = sqrt(h0/h2)
v = np.logspace(-3, 0, 300)
ax[1].semilogx(v, [np.exp(-H_rate(vv, vv)) for vv in v], color="k")
vstar = np.sqrt(h0 / h2); ax[1].axvline(vstar, color="k", ls=":", lw=1)
ax[1].annotate(f"v* = {vstar:.3f}/yr\n(whole path in {1/vstar:.0f} yr)", (vstar, 0.5), textcoords="offset points", xytext=(8, 0), fontsize=8)
ax[1].set(xlabel="constant speed of progress (fraction of path per yr)", ylabel="survival probability", ylim=(0, 1.02),
          title="(b) Watchers infer from rate: an optimal pace"); ax[1].grid(alpha=.3)

# (c) mixed: survival vs sprint speed for creep at v1, several lags
v2 = np.logspace(-3, 0, 300)
for tau, c in [(2, "#1f77b4"), (5, "#d62728"), (10, "#2ca02c")]:
    ax[2].semilogx(v2, [np.exp(-H_mixed(v1, vv, tau)) for vv in v2], color=c, label=f"lag = {tau} yr")
    ax[2].axvline((1 - x_c) / tau, color=c, ls=":", lw=1)
ax[2].axhline(np.exp(-H_mixed(v1, v1, 0.0)), color="gray", ls="--", lw=1, label="creep straight through (no lag)")
ax[2].set(xlabel="sprint speed v2 (fraction of path per yr); creep at 0.01/yr", ylabel="survival probability", ylim=(0, 1.02),
          title="(c) Mixed: slowest sprint that fits the lag"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

fig.tight_layout(); fig.savefig("fig_pacing.pdf"); fig.savefig("fig_pacing.png", dpi=150)
for tau in (2, 5, 10):
    v2s = (1 - x_c) / tau
    print(f"tau={tau:2d} yr: sprint-in-lag speed {v2s:.3f}/yr -> survival {np.exp(-H_mixed(v1, v2s, tau)):.3f};"
          f"  creep-through survival {np.exp(-H_mixed(v1, v1, 0.0)):.3f};  condition tau > {(1-x_c)/(h1/(h2*v1)+v1):.3f} yr")
print(f"rate regime optimum v*={vstar:.4f}/yr, H*={2*np.sqrt(h0*h2):.4f}")
