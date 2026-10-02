"""Near-duplicate contract families on Dataset I, for a leakage-free split.

Contracts are represented by their set of instruction 4-grams (after PUSH-aware
disassembly, so differing constants/addresses do not matter). Two contracts are
linked when their Jaccard similarity is >= THRESH; families are the connected
components. A random 68/15/17 split puts 95.9 % of greedy test contracts next to
a >= 0.95-similar training contract, so accuracy on that split partly measures
template recognition. ``run.prepare(split="group")`` keeps each family whole.

    python -m experiments.families            # Dataset I  -> families.npy (built with PYTHONHASHSEED=0)
    python -m experiments.families iir        # Dataset II-r -> families_iir.npy
    python -m experiments.families ii4        # Dataset II   -> families_ii4.npy
    python -m experiments.families ii4 0.7    # sensitivity: stats only -> families_ii4_t0.7.json

Datasets II/II-r use zlib.crc32 for the n-gram hash, which is stable across
processes; Dataset I's existing file was built with Python's hash() under
PYTHONHASHSEED=0 and is kept as is so its results stay valid.
"""

from __future__ import annotations

import json
import sys
import zlib

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from xvascot.config import RESULTS
from xvascot.data import disassemble_tokens, load_dataset_i, load_dataset_ii_xls

THRESH = 0.8
DIM = 1 << 20


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "i"
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else THRESH
    df = load_dataset_i() if ds == "i" else load_dataset_ii_xls(binary=(ds == "iir"))
    h = (lambda g: hash(g)) if ds == "i" else (lambda g: zlib.crc32(g.encode()))
    y = df["LABEL"].to_numpy()
    rows, cols = [], []
    for r, s in enumerate(df["OPCODE"].values):
        # Dataset I is a raw byte stream; Datasets II/II-r are already opcode-only
        ins = disassemble_tokens(str(s).split()) if ds == "i" else str(s).split()
        grams = {h(" ".join(ins[i:i + 4])) % DIM for i in range(len(ins) - 3)}
        rows += [r] * len(grams)
        cols += list(grams)
    A = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(df), DIM))
    inter = (A @ A.T).toarray()
    size = np.asarray(A.sum(1)).ravel()
    jac = inter / (size[:, None] + size[None, :] - inter + 1e-9)
    np.fill_diagonal(jac, 0)
    n, fam = connected_components(csr_matrix(jac >= thresh), directed=False)

    sizes = np.bincount(fam)
    mixed = sum(1 for f in range(n) if len(set(y[fam == f])) > 1)
    classes = sorted(set(y.tolist()))
    out = {"threshold": thresh, "n_families": int(n), "n_singletons": int((sizes == 1).sum()),
           "largest_families": sorted(((int(sizes[f]), {int(c): int((y[fam == f] == c).sum()) for c in classes})
                                       for f in np.argsort(-sizes)[:10]), key=lambda t: -t[0]),
           "mixed_label_families": int(mixed),
           "families_per_class": {int(c): int(len(set(fam[y == c]))) for c in classes},
           # kept for Dataset I compatibility (label 1 = greedy there)
           "greedy_families": int(len(set(fam[y == 1]))), "safe_families": int(len(set(fam[y == 0])))}
    suffix = "" if ds == "i" else f"_{ds}"
    if thresh != THRESH:              # sensitivity run: never overwrite the split's families
        suffix += f"_t{thresh}"
    else:
        np.save(RESULTS / f"families{suffix}.npy", fam)
    (RESULTS / f"families{suffix}.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
