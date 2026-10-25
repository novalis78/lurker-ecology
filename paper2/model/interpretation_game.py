"""
Interpretation heterogeneity in the volunteer's dilemma (Treading Carefully, Section 5.5).

n resident lineages; a of them read the target's act as a trigger (potential strikers); d of them would
punish an attacker they can locate (deterrers), whether or not they read the act as a trigger.
Exposure cost of striking: K(d) = V * [1 - (1 - delta)^d]   (detection-and-punishment by each deterrer).
Survival of the target:
    a = 0 : Q = 1
    a = 1 : Q = 1 if K(d) >= B (lone striker deterred), else 0
    a >= 2: Q = min(K(d)/B, 1) ** (a/(a-1))        (symmetric mixed equilibrium of the VD among the a strikers)
Shows that fewer strikers is protective only through the punishers: reducing a from n toward 1 LOWERS Q
unless K(d) >= B.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def K(d, V=1.5, delta=0.1): return V * (1 - (1 - delta) ** d)

def Q(a, d, B=1.0, V=1.5, delta=0.1):
    k = K(d, V, delta)
    if a == 0: return 1.0
    if a == 1: return 1.0 if k >= B else 0.0
    r = min(k / B, 1.0)
    return r ** (a / (a - 1))

A = np.arange(1, 13); D = np.arange(0, 31)
Z = np.array([[Q(a, d) for d in D] for a in A])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
im = ax[0].imshow(Z, origin="lower", aspect="auto", extent=[D[0]-.5, D[-1]+.5, A[0]-.5, A[-1]+.5], vmin=0, vmax=1, cmap="viridis")
ax[0].set(xlabel="lineages that would punish an attacker, d", ylabel="lineages that read the act as a trigger, a",
          title="(a) Survival Q(a, d);  V/B = 1.5, delta = 0.1")
fig.colorbar(im, ax=ax[0], label="Q")
dstar = next(d for d in D if K(d) >= 1.0); ax[0].axvline(dstar - .5, color="w", ls=":", lw=1)
ax[0].annotate(f"K(d) = B at d = {dstar}", (dstar, 11.3), color="w", fontsize=8)
for d, c in [(2, "#1f77b4"), (5, "#d62728"), (10, "#2ca02c")]:
    ax[1].plot(A, [Q(a, d) for a in A], "o-", color=c, label=f"d = {d} punishers (K/B = {K(d):.2f})")
ax[1].set(xlabel="potential strikers, a", ylabel="survival Q", ylim=(-0.02, 1.05),
          title="(b) Fewer strikers is NOT safer unless punishers deter a lone one"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("fig_interpretation.pdf"); fig.savefig("fig_interpretation.png", dpi=150)
print("K(d)>=B first at d =", dstar)
for d in (2, 5, 10):
    print(f"d={d}: Q(a=1)={Q(1,d):.2f}  Q(a=2)={Q(2,d):.2f}  Q(a=5)={Q(5,d):.2f}  Q(a=12)={Q(12,d):.2f}")
