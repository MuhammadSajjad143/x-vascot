"""Bootstrap confidence intervals and paired tests for t1_baselines.json and
ig_specificity.json (both with per-contract values).

T1: contracts are resampled with replacement within each model (stratified by
model), the statistic is the mean over models of the per-model mean, 10,000
replicates, percentile 95 % CI. X-VASCOT vs every other selector: the same
resample indices give a paired CI of the difference, plus a Wilcoxon signed-rank
test on the pooled per-contract differences.

Specificity: hit rate per (instruction, explained class, selector), pooled over
models, 95 % bootstrap CI over contracts, and the lift over random selection on
the same contracts.

    python -m experiments.ci_analysis
"""
import json
import numpy as np
from scipy.stats import wilcoxon
from xvascot.config import RESULTS

B = 10_000
rng = np.random.default_rng(0)


def strat_boot(groups, B=B):
    """groups: list of per-model arrays (same contracts for paired use). Returns index sets."""
    return [[rng.integers(0, len(g), size=(B, len(g))) for g in groups]]


t1 = json.loads((RESULTS / "t1_baselines.json").read_text())
out = {"t1": {}, "specificity": {}}
for mode, r in t1.items():
    pc = r["per_contract"]
    groups = pc["xvascot"]["comp"]
    idx = [rng.integers(0, len(g), size=(B, len(g))) for g in groups]
    res = {}
    for m in pc:
        res[m] = {}
        for key in ("comp", "suff"):
            arrs = [np.asarray(g) for g in pc[m][key]]
            stat = np.mean([a[ix].mean(1) for a, ix in zip(arrs, idx)], 0)
            ref = [np.asarray(g) for g in pc["xvascot"][key]]
            diff = np.mean([(a - b)[ix].mean(1) for a, b, ix in zip(ref, arrs, idx)], 0)
            flat = np.concatenate(ref) - np.concatenate(arrs)
            res[m][key] = {
                "mean": float(np.mean([a.mean() for a in arrs])),
                "ci95": [float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))],
                "xvascot_minus_this": float(np.mean([(a - b).mean() for a, b in zip(ref, arrs)])),
                "diff_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                "wilcoxon_p": (float(wilcoxon(flat).pvalue) if m != "xvascot" and np.any(flat) else None),
                "n_contracts": int(sum(len(a) for a in arrs))}
    out["t1"][mode] = res
    print(f"\n===== {mode}   (X-VASCOT minus selector; comp: >0 better for X-VASCOT; suff: <0 better)")
    for m in pc:
        c, s_ = res[m]["comp"], res[m]["suff"]
        print(f"  {m:<13} comp {c['mean']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]  "
              f"suff {s_['mean']:+.3f} [{s_['ci95'][0]:+.3f},{s_['ci95'][1]:+.3f}]"
              + ("" if m == "xvascot" else
                 f"   d.comp {c['xvascot_minus_this']:+.3f} [{c['diff_ci95'][0]:+.3f},{c['diff_ci95'][1]:+.3f}] p={c['wilcoxon_p']:.1e}"
                 f"   d.suff {s_['xvascot_minus_this']:+.3f} [{s_['diff_ci95'][0]:+.3f},{s_['diff_ci95'][1]:+.3f}] p={s_['wilcoxon_p']:.1e}"))

sp = json.loads((RESULTS / "ig_specificity.json").read_text())
for mode, r in sp.items():
    res = {}
    for key, v in r.items():
        h = np.asarray(v["hits"], float)
        bs = h[rng.integers(0, len(h), size=(B, len(h)))].mean(1)
        op, cls, meth = key.split("|")
        rnd = np.asarray(r[f"{op}|{cls}|random"]["hits"], float).mean()
        res[key] = {"hit": float(h.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "n": len(h), "random_hit": float(rnd), "lift_over_random": float(h.mean() - rnd)}
    out["specificity"][mode] = res
    print(f"\n===== specificity {mode}: hit % [95 % CI] by explained class (safe/suicidal/prodigal/greedy)")
    for op in ("ff", "f1"):
        for meth in ("xvascot", "int_grad", "grad_x_input", "xvascot_v1", "raw_attn", "saliency", "rare_corpus", "random"):
            if f"{op}|0|{meth}" not in res:
                continue
            print(f"  {op} {meth:<13}" + "  ".join(
                f"{100*res[f'{op}|{c}|{meth}']['hit']:4.0f} [{100*res[f'{op}|{c}|{meth}']['ci95'][0]:3.0f},{100*res[f'{op}|{c}|{meth}']['ci95'][1]:3.0f}]"
                for c in range(4)))
(RESULTS / "ci_analysis.json").write_text(json.dumps(out, indent=2))
