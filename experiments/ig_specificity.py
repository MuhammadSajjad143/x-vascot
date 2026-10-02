"""Signature specificity (semantic_tests T4) for every model-based selector of
t1_baselines.py, incl. signed Integrated Gradients and gradient x input: among test
contracts predicted c that contain the opcode, share whose top-5 % holds it when
explaining c. Class-specific evidence = high for the class the opcode defines and
low elsewhere.

    python -m experiments.ig_specificity ii4_full ii4_full_dedup ii4_full_group
"""
import json, sys
from collections import Counter
import numpy as np, torch
from experiments.train import DEVICE, build_vascot, prepare, trained_seeds
from xvascot.config import CKPTS, RESULTS, TOPK_RATIO
from xvascot.xai import adaptive_k
from experiments.t1_baselines import model_scores, probs

MAX_N = 40
OPS = ("f1", "ff")
path = RESULTS / "ig_specificity.json"
allres = json.loads(path.read_text()) if path.exists() else {}
for mode in sys.argv[1:]:
    acc = {}
    for seed in trained_seeds(mode):
        ctx = prepare(mode, seed=seed, log=lambda *a: None)
        m = build_vascot(mode)
        m.load_state_dict(torch.load(CKPTS / f"{mode}_seedsweep_vascot_{seed}.pt", map_location=DEVICE, weights_only=True)); m.eval()
        word, y = ctx["tok"].word_index, np.asarray(ctx["yte"])
        freq = Counter(t for s_ in ctx["Xtr"] for t in np.asarray(s_).tolist() if t)
        rng = np.random.default_rng(seed)
        for cls in range(4):
            n = 0
            for i in np.where(y == cls)[0]:
                seq = np.asarray(ctx["Xte"][i], dtype=np.int64)
                if int(probs(m, seq).argmax()) != cls:
                    continue
                nonpad = seq > 0; toks = seq[nonpad]
                present = [op for op in OPS if word.get(op) in set(toks.tolist())]
                if not present:
                    continue
                k = adaptive_k(len(toks), TOPK_RATIO)
                scs = {mt: v[nonpad] for mt, v in model_scores(m, seq, cls).items()}
                scs["rare_corpus"] = -np.array([freq.get(int(t), 0) for t in toks], float) + rng.random(len(toks)) * 1e-3
                scs["random"] = rng.random(len(toks))
                for meth, sc in scs.items():
                    top = np.argsort(sc)[::-1][:k]
                    for op in present:
                        acc.setdefault(f"{op}|{cls}|{meth}", []).append(int(np.isin(np.where(toks == word[op])[0], top).any()))
                n += 1
                if n >= MAX_N:
                    break
    res = {k: {"hit": float(np.mean(v)), "n": len(v), "hits": v} for k, v in acc.items()}
    allres[mode] = res
    path.write_text(json.dumps(allres, indent=2))
    print(f"===== {mode}  hit rate of opcode in top-5 % when explaining class (n)")
    meths = sorted({k.split("|")[2] for k in res})
    for op in OPS:
        print(f"  {op}: " + " | ".join(f"class {c}" for c in range(4)))
        for meth in meths:
            print(f"    {meth:<13}" + "  ".join(f"{100*res[f'{op}|{c}|{meth}']['hit']:5.0f}%({res[f'{op}|{c}|{meth}']['n']:>3})"
                                         if f"{op}|{c}|{meth}" in res else "    -    " for c in range(4)), flush=True)
