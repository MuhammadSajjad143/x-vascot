"""Data audit for Datasets II (4-class) and II-r (binary), as done for Dataset I.

* exact duplicates per class, and how many test contracts in the base-paper
  random split have an identical bytecode in the training split (seed 42);
* near-duplicate families (families_<ds>.npy) and label purity;
* a length-only classifier (does length alone predict the label?);
* evidence that the sequences are opcode-only (PUSH operands already stripped by
  the dataset authors), so every token is an instruction and no disassembly applies;
* prevalence of the instructions MAIAN's classes are defined by: real
  SELFDESTRUCT (suicidal) and CALL (prodigal / Ether-sending), per class.

    python -m experiments.audit_dataset_ii iir ii4
"""

from __future__ import annotations

import json
import sys

from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from xvascot.config import RESULTS, SEEDS
from xvascot.data import II_CLASSES, load_dataset_ii_xls
from xvascot.xai import ETH_OPCODE_MAP, hex_to_mnemonic


def audit(ds: str) -> dict:
    df = load_dataset_ii_xls(binary=(ds == "iir"))
    y = df["LABEL"].to_numpy()
    names = ["non-vulnerable", "greedy"] if ds == "iir" else II_CLASSES
    ops = df["OPCODE"].astype(str)
    out = {"dataset": ds, "rows": len(df), "distinct_bytecodes": int(ops.nunique()),
           "per_class": {}}
    toks = [s.split() for s in ops]
    L = np.array([len(t) for t in toks], float)
    sets = [set(x.lower() for x in t) for t in toks]
    for c, nm in enumerate(names):
        m = y == c
        has = lambda op: np.array([op in s for s, keep in zip(sets, m) if keep])
        has_sd, has_call = has("ff"), has("f1")
        out["per_class"][nm] = {
            "rows": int(m.sum()), "distinct": int(ops[m].nunique()),
            "dup_rate": float(1 - ops[m].nunique() / max(m.sum(), 1)),
            "mean_bytes": float(L[m].mean()),
            "pct_real_selfdestruct": float(100 * has_sd.mean()),
            "pct_call": float(100 * has_call.mean()),
            "pct_any_ether_transfer": float(100 * np.mean([bool(s & {"f1", "f2", "f4", "ff"})
                                                            for s, keep in zip(sets, m) if keep])),
        }

    # leakage in the base-paper random split (same 68/15/17 stratified split as run.prepare)
    leak = []
    for seed in SEEDS:
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.17, random_state=seed, stratify=y)
        trs = set(ops.iloc[tr])
        leak.append(float(np.mean([ops.iloc[i] in trs for i in te])))
    out["pct_test_with_identical_bytecode_in_train"] = {"mean": 100 * float(np.mean(leak)),
                                                         "per_seed": [100 * v for v in leak]}

    # length-only classifier, random split and deduplicated split
    acc = {"random_split": [], "dedup_split": []}
    for seed in SEEDS:
        for key, keep in (("random_split", np.ones(len(y), bool)),
                          ("dedup_split", ~ops.duplicated().to_numpy())):
            Lk, yk = L[keep], y[keep]
            tr, te = train_test_split(np.arange(len(yk)), test_size=0.17, random_state=seed, stratify=yk)
            clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Lk[tr, None], yk[tr])
            acc[key].append(float((clf.predict(Lk[te, None]) == yk[te]).mean()))
    out["length_only_accuracy"] = {k: {"mean": 100 * float(np.mean(v)), "std": 100 * float(np.std(v))}
                                   for k, v in acc.items()}
    out["majority_baseline"] = float(100 * np.bincount(y).max() / len(y))

    # opcode-only evidence: distinct token values, and what follows a PUSH20 (in raw
    # bytecode the next 20 bytes are an address, i.e. near-uniform random bytes)
    vals = set().union(*sets)
    undefined = sorted(v for v in vals if v not in ETH_OPCODE_MAP)
    after_push20 = Counter(t[i + 1].lower() for t in toks for i in range(len(t) - 1) if t[i].lower() == "73")
    tot = sum(after_push20.values())
    out["opcode_only_evidence"] = {
        "distinct_token_values": len(vals),
        "undefined_opcode_values": undefined,
        "after_PUSH20_top5": [(hex_to_mnemonic(k), round(100 * v / tot, 1)) for k, v in after_push20.most_common(5)],
        "after_PUSH20_distinct": len(after_push20)}

    fam = np.load(RESULTS / f"families_{ds}.npy")
    out["families"] = json.loads((RESULTS / f"families_{ds}.json").read_text())
    out["families"]["pct_rows_in_label_pure_families"] = float(100 * np.mean(
        [len(set(y[fam == f])) == 1 for f in fam]))
    return out


def main():
    res = {}
    for ds in sys.argv[1:] or ["iir", "ii4"]:
        r = audit(ds)
        res[ds] = r
        print(f"\n===== {ds}: {r['rows']} rows, {r['distinct_bytecodes']} distinct bytecodes")
        for nm, v in r["per_class"].items():
            print(f"  {nm:<15} rows {v['rows']:>5}  distinct {v['distinct']:>5}  dup {100*v['dup_rate']:5.1f}%  "
                  f"instr {v['mean_bytes']:6.0f}  SELFDESTRUCT {v['pct_real_selfdestruct']:5.1f}%  CALL {v['pct_call']:5.1f}%  "
                  f"any transfer {v['pct_any_ether_transfer']:5.1f}%")
        print(f"  test contracts with identical bytecode in train (random split): "
              f"{r['pct_test_with_identical_bytecode_in_train']['mean']:.1f}%")
        print(f"  length-only accuracy: random {r['length_only_accuracy']['random_split']['mean']:.1f}%, "
              f"dedup {r['length_only_accuracy']['dedup_split']['mean']:.1f}%  (majority {r['majority_baseline']:.1f}%)")
        print(f"  opcode-only evidence: {r['opcode_only_evidence']}")
        print(f"  rows in label-pure families: {r['families']['pct_rows_in_label_pure_families']:.1f}%")
    (RESULTS / "audit_ii.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
