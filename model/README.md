# Lineage dynamics model

Minimal birth-and-death simulation behind Figure 3 of *Lurker Ecology* (Lopin 2026), Section 6.8.

```
python3 lineage_dynamics.py
```

Writes `lineage_dynamics.pdf` / `.png` and prints summary numbers (`run_summary.txt` holds the last run).
Requires numpy and matplotlib. Runtime about a minute.

State: `L` lineages in the Galaxy, `n` coexisting in a representative system. Rates per Myr.
Parameters are illustrative; the figure shows which qualitative regimes are possible, not an estimate
for the Solar System. See the docstring at the top of the script for the model definition.
