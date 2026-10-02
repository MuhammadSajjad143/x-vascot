"""Audit of Dataset I (the Dataset I column of the paper's Table 2).

* PUSH-operand share of the raw byte stream;
* contracts cut at the spreadsheet cell limit (10,922 bytes = 32,767 characters
  at three characters per byte), per class;
* greedy test contracts of the base-paper random split (seed 42) that have a
  >= 0.95-similar training contract (instruction 4-gram Jaccard);
* length-only accuracy: depth-4 decision tree on contract length, mean over the
  five seeds of the random split.

    python -m experiments.audit_dataset_i
"""

from __future__ import annotations

import json

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from xvascot.config import RESULTS, SEEDS
from xvascot.data import disassemble_tokens, load_dataset_i
from xvascot.xai import disassemble_mask

CELL_LIMIT_BYTES = 32767 // 3
DIM = 1 << 20


def split(y, seed):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.17, random_state=seed, stratify=y)
    return tr, te


def main():
    df = load_dataset_i()
    y = df["LABEL"].to_numpy()
    toks = [str(s).split() for s in df["OPCODE"].values]
    L = np.array([len(t) for t in toks])
    out = {}

    masks = [disassemble_mask(t) for t in toks]
    out["operand_pct"] = 100 * (1 - sum(int(m.sum()) for m in masks) / L.sum())
    cut = L >= CELL_LIMIT_BYTES
    out["truncated_pct"] = {"safe": 100 * cut[y == 0].mean(), "greedy": 100 * cut[y == 1].mean()}

    rows, cols = [], []
    for r, t in enumerate(toks):
        ins = disassemble_tokens(t)
        grams = {hash(" ".join(ins[i:i + 4])) % DIM for i in range(len(ins) - 3)}
        rows += [r] * len(grams)
        cols += list(grams)
    A = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(df), DIM))
    inter = (A @ A.T).toarray()
    size = np.asarray(A.sum(1)).ravel()
    jac = inter / (size[:, None] + size[None, :] - inter + 1e-9)
    np.fill_diagonal(jac, 0)
    tr, te = split(y, 42)
    g = te[y[te] == 1]
    out["greedy_test_with_train_twin_pct"] = 100 * float((jac[np.ix_(g, tr)].max(1) >= 0.95).mean())

    acc = []
    for seed in SEEDS:
        tr, te = split(y, seed)
        clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(L[tr, None], y[tr])
        acc.append(float((clf.predict(L[te, None]) == y[te]).mean()))
    out["length_only_accuracy_pct"] = 100 * float(np.mean(acc))
    out["majority_pct"] = 100 * float(np.bincount(y).max() / len(y))

    print(json.dumps(out, indent=2))
    (RESULTS / "audit_i.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
