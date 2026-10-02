"""Evaluate the five stage-7 checkpoints of a mode on their own test splits.

* ``*_group`` modes: family-grouped CV. Each fold's model predicts its held-out
  fold, so pooling gives one prediction per contract from a model that never saw
  its family.
* other modes: per-seed metrics, mean and std, plus each seed's confusion matrix
  (for comparison with the base paper's Figs. 12-13).

Binary (Datasets I, II-r) and 4-class (Dataset II) are both handled.

    python -m experiments.group_cv_eval disasm_group iir_full ii4_full_group
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from experiments.train import DEVICE, build_vascot, prepare, trained_seeds
from xvascot.config import CKPTS, RESULTS, SEEDS, mode_cfg


def metrics(y, p, prob, ncls):
    avg = "binary" if ncls == 2 else "macro"
    out = {"acc": float(accuracy_score(y, p)),
           "prec": float(precision_score(y, p, average=avg, zero_division=0)),
           "rec": float(recall_score(y, p, average=avg, zero_division=0)),
           "f1": float(f1_score(y, p, average=avg, zero_division=0)),
           "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
           "balanced_acc": float(balanced_accuracy_score(y, p)),
           "per_class_recall": recall_score(y, p, average=None, labels=list(range(ncls)), zero_division=0).tolist(),
           "confusion": confusion_matrix(y, p, labels=list(range(ncls))).tolist()}
    try:
        out["auc"] = float(roc_auc_score(y, prob[:, 1]) if ncls == 2 else
                           roc_auc_score(y, prob, multi_class="ovr", labels=list(range(ncls))))
    except ValueError:
        out["auc"] = None
    return out


@torch.no_grad()
def evaluate(mode):
    ncls = mode_cfg(mode).get("n_classes", 2)
    ys, ps, prs, per, rows = [], [], [], [], []
    for seed in trained_seeds(mode):
        ctx = prepare(mode, seed=seed, log=lambda *a: None)
        m = build_vascot(mode)
        m.load_state_dict(torch.load(CKPTS / f"{mode}_seedsweep_vascot_{seed}.pt",
                                     map_location=DEVICE, weights_only=True))
        m.eval()
        yt = np.asarray(ctx["yte"])
        prob = np.stack([F.softmax(m(torch.from_numpy(np.asarray(s)).unsqueeze(0).to(DEVICE)).float(), 1)[0].cpu().numpy()
                         for s in ctx["Xte"]])
        pred = prob.argmax(1)
        per.append({"seed_or_fold": seed, "n": int(len(yt)), **metrics(yt, pred, prob, ncls)})
        ys.append(yt); ps.append(pred); prs.append(prob); rows.append(np.asarray(ctx["te_rows"]))
    out = {"mode": mode, "n_classes": ncls, "per_seed": per,
           "mean": {k: float(np.mean([r[k] for r in per if r[k] is not None]))
                    for k in ("acc", "prec", "rec", "f1", "macro_f1", "balanced_acc", "auc")},
           "std": {k: float(np.std([r[k] for r in per if r[k] is not None]))
                   for k in ("acc", "prec", "rec", "f1", "macro_f1", "balanced_acc", "auc")}}
    if mode_cfg(mode).get("split") == "group":
        y, p, pr = np.concatenate(ys), np.concatenate(ps), np.concatenate(prs)
        out["pooled"] = {"n": int(len(y)), **metrics(y, p, pr, ncls)}
        # The same pooled predictions with each distinct bytecode counted once, so a
        # family of thousands of identical rows (Dataset II's 3,138-row safe template)
        # does not dominate the score. Keeps the first row of each bytecode.
        from xvascot.data import load_dataset_i, load_dataset_ii_xls
        ds = mode_cfg(mode).get("dataset", "i")
        df = load_dataset_i() if ds == "i" else load_dataset_ii_xls(binary=(ds == "iir"))
        code = df["OPCODE"].astype(str).to_numpy()[np.concatenate(rows)]
        _, first = np.unique(code, return_index=True)
        out["pooled_distinct_bytecodes"] = {"n": int(len(first)), **metrics(y[first], p[first], pr[first], ncls)}
    return out


def main():
    path = RESULTS / "group_cv_eval.json"
    allr = json.loads(path.read_text()) if path.exists() else {}
    for mode in sys.argv[1:]:
        r = evaluate(mode)
        allr[mode] = r
        print(f"\n== {mode}: mean acc {100*r['mean']['acc']:.2f} ± {100*r['std']['acc']:.2f}  "
              f"F1 {r['mean']['f1']:.3f}  macro-F1 {r['mean']['macro_f1']:.3f} ± {r['std']['macro_f1']:.3f}  "
              f"bal.acc {100*r['mean']['balanced_acc']:.2f}  AUC {r['mean']['auc']:.3f}")
        print("   seed-42 / fold-0 confusion:", r["per_seed"][0]["confusion"])
        if "pooled" in r:
            p = r["pooled"]
            print(f"   pooled over folds: acc {100*p['acc']:.2f}  macro-F1 {p['macro_f1']:.3f}  "
                  f"bal.acc {100*p['balanced_acc']:.2f}  per-class recall "
                  f"{[round(v, 3) for v in p['per_class_recall']]}  AUC {p['auc']}")
            q = r["pooled_distinct_bytecodes"]
            print(f"   pooled, distinct bytecodes (n={q['n']}): acc {100*q['acc']:.2f}  macro-F1 {q['macro_f1']:.3f}  "
                  f"bal.acc {100*q['balanced_acc']:.2f}  per-class recall {[round(v, 3) for v in q['per_class_recall']]}")
        path.write_text(json.dumps(allr, indent=2))


if __name__ == "__main__":
    main()
