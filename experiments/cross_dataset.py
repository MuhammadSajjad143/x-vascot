"""Cross-dataset experiments of the base paper (Balci et al., Sec. 6.5), full byte input.

(a) unverified -> verified: train on Dataset II-r (all rows except a 15 % validation
    split), then (a0) test zero-shot on Dataset I's 326-contract test split and
    (a1) fine-tune on Dataset I's training split and test on the same 326.
    Base paper Fig. 14 (VASCOT: TN 164, FP 41, FN 10, TP 111).
(b) verified -> unverified: train on Dataset I (its 68/15 train/val split), test on
    Dataset II-r's 2,344-contract test split. Base paper Fig. 15
    (VASCOT: TN 1314, FP 47, FN 15, TP 968).

The tokenizer is always fitted on the training data of the first stage. Splits
reuse run.prepare's stratified 68/15/17 protocol with the same seed.

    python -m experiments.cross_dataset             # Dataset I as raw bytes (as the base paper)
    python -m experiments.cross_dataset --matched   # Dataset I disassembled, so both sides are
                                        # opcode-only instruction streams (II/II-r ship
                                        # with PUSH operands already stripped)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from experiments.train import DEVICE, build_vascot, set_seed
from xvascot.config import CONFIG, RESULTS, SEEDS
from xvascot.data import build_corpus, load_dataset_i, load_dataset_ii_xls, make_tokenizer
from xvascot.train import evaluate_model, make_loader, train_model

N_SEEDS = 3


def split(y, seed):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.17, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.15 / 0.83, random_state=seed, stratify=y[tr])
    return tr, va, te


def loaders(X, y, tr, va, te, seed):
    take = lambda ix: ([X[i] for i in ix], y[ix])
    bs = CONFIG["batch_size"]
    out = {}
    if tr is not None:
        out["train"] = make_loader(*take(tr), bs, True, True, seed)
    if va is not None:
        out["val"] = make_loader(*take(va), 1, False, True, seed)
    if te is not None:
        out["test"] = make_loader(*take(te), 1, False, True, seed)
    return out


def summarise(res, label):
    cm = confusion_matrix(res["labels"], res["preds"], labels=[0, 1]).tolist()
    return {"label": label, "acc": float(res["acc"]), "prec": float(res["prec"]),
            "rec": float(res["rec"]), "f1": float(res["f1"]), "auc": float(res["auc"]),
            "confusion": cm}


def main():
    matched = "--matched" in sys.argv
    d1, d2 = load_dataset_i(), load_dataset_ii_xls(binary=True)
    if matched:
        from xvascot.data import disassemble_frame
        d1 = disassemble_frame(d1)
    outfile = RESULTS / ("cross_dataset_matched.json" if matched else "cross_dataset.json")
    y1, y2 = d1["LABEL"].to_numpy(), d2["LABEL"].to_numpy()
    out = {"a0_iir_to_i_zero_shot": [], "a1_iir_to_i_finetuned": [], "b_i_to_iir": []}
    log = lambda *a: print(*a, flush=True)
    for seed in SEEDS[:N_SEEDS]:
        tr1, va1, te1 = split(y1, seed)
        tr2, va2, te2 = split(y2, seed)

        # (a) train on all of II-r except its validation split
        tok = make_tokenizer(d2.iloc[np.r_[tr2, te2]])
        X2, _, _ = build_corpus(d2, tok, None)
        X1, _, _ = build_corpus(d1, tok, None)
        set_seed(seed)
        m = build_vascot("iir_full")
        L2 = loaders(X2, y2, np.r_[tr2, te2], va2, None, seed)
        train_model(m, L2["train"], L2["val"], CONFIG, DEVICE, f"a-train-IIr-s{seed}", log)
        L1 = loaders(X1, y1, tr1, va1, te1, seed)
        r = evaluate_model(m, L1["test"], DEVICE, f"a0 zero-shot s{seed}", log, verbose=False)
        out["a0_iir_to_i_zero_shot"].append(summarise(r, f"seed {seed}"))
        train_model(m, L1["train"], L1["val"], CONFIG, DEVICE, f"a1-finetune-I-s{seed}", log)
        r = evaluate_model(m, L1["test"], DEVICE, f"a1 fine-tuned s{seed}", log, verbose=False)
        out["a1_iir_to_i_finetuned"].append(summarise(r, f"seed {seed}"))
        log(f"[seed {seed}] (a0) {out['a0_iir_to_i_zero_shot'][-1]}\n[seed {seed}] (a1) {out['a1_iir_to_i_finetuned'][-1]}")

        # (b) train on Dataset I, test on II-r's test split
        tok = make_tokenizer(d1.iloc[tr1])
        X1, _, _ = build_corpus(d1, tok, None)
        X2, _, _ = build_corpus(d2, tok, None)
        set_seed(seed)
        m = build_vascot("disasm" if matched else "fixed")
        L1 = loaders(X1, y1, tr1, va1, None, seed)
        train_model(m, L1["train"], L1["val"], CONFIG, DEVICE, f"b-train-I-s{seed}", log)
        L2 = loaders(X2, y2, None, None, te2, seed)
        r = evaluate_model(m, L2["test"], DEVICE, f"b I->IIr s{seed}", log, verbose=False)
        out["b_i_to_iir"].append(summarise(r, f"seed {seed}"))
        log(f"[seed {seed}] (b) {out['b_i_to_iir'][-1]}")
        outfile.write_text(json.dumps(out, indent=2))

    for k, v in out.items():
        print(f"== {k}: acc {100*np.mean([r['acc'] for r in v]):.2f} ± {100*np.std([r['acc'] for r in v]):.2f}  "
              f"greedy recall {np.mean([r['rec'] for r in v]):.3f}  safe recall "
              f"{np.mean([r['confusion'][0][0] / max(1, sum(r['confusion'][0])) for r in v]):.3f}")


if __name__ == "__main__":
    main()
