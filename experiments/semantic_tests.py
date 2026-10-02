"""Semantic tests on Dataset II / II-r, where MAIAN's classes have instruction-level
signatures (audit_dataset.py: real SELFDESTRUCT in 81 % of suicidal contracts vs
1-14 % elsewhere; CALL in 22 % of greedy vs 97 % of safe).

Datasets II/II-r ship opcode-only sequences (PUSH operands stripped by the dataset
authors), so on their ``full`` input the token ``ff`` IS a SELFDESTRUCT instruction
and ``f1`` IS a CALL.

T1 signature, suicidal (ii4): for suicidal contracts predicted suicidal that
   contain SELFDESTRUCT, explain the suicidal logit and ask
   (a) does X-VASCOT's top-5 % contain a SELFDESTRUCT (hit rate) vs random top-5 %,
       raw attention and gradient-only saliency; percentile rank of the best one;
   (b) does deleting ONLY the SELFDESTRUCT instructions remove the verdict, vs
       deleting the same number of random instructions?
T2 signature, prodigal (ii4): the same with CALL (f1) for prodigal verdicts.
T3 greedy counterfactual (iir and ii4): take safe contracts predicted safe and
   delete every instruction that can transfer Ether (CALL f1, CALLCODE f2,
   DELEGATECALL f4, SELFDESTRUCT ff) - MAIAN's greedy check looks for exactly these.
   Caveats: zero-value calls are removed too, and whether the contract can still
   RECEIVE Ether (MAIAN's other greedy condition) cannot be checked on opcode-only
   input. Does P(greedy) rise, vs deleting the same number of random instructions?
T4 specificity control for T1/T2 (ii4): for EVERY class c (safe included), among
   contracts predicted c that contain the opcode, how often is the opcode in the
   top-5 % when explaining c? A hit rate that is equally high for classes the opcode
   says nothing about means the hit is not class-specific evidence.

    python -m experiments.semantic_tests ii4_full iir_full [ii4_full_dedup ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from experiments.train import DEVICE, build_vascot, prepare, trained_seeds
from xvascot.config import CKPTS, CONFIG, RESULTS, SEEDS, TOPK_RATIO
from xvascot.xai import XVASCOT, adaptive_k

MAX_N = 60
RELEASE = ("f1", "f2", "f4", "ff")     # CALL, CALLCODE, DELEGATECALL, SELFDESTRUCT


class Saliency(XVASCOT):
    @staticmethod
    def _attn_vector(attn):
        return torch.ones(attn.shape[-1], device=attn.device)


@torch.no_grad()
def probs(model, seq):
    return F.softmax(model(torch.from_numpy(seq).unsqueeze(0).to(DEVICE)).float(), 1)[0].cpu().numpy()


def signature(ctx, model, cls, op_hex, rng):
    tok = ctx["tok"].word_index
    op_id = tok.get(op_hex)
    if op_id is None or op_id >= CONFIG["num_words"]:
        return {"error": f"{op_hex} not in vocabulary"}
    xv = XVASCOT(model, ctx["tok"], CONFIG, DEVICE, "l2"); xv.target = cls
    sal = Saliency(model, ctx["tok"], CONFIG, DEVICE, "l2", score="attn_grad"); sal.target = cls
    y = np.asarray(ctx["yte"])
    xv1 = XVASCOT(model, ctx["tok"], CONFIG, DEVICE, "l2", score="attn_grad"); xv1.target = cls
    rec = {m: {"hit": [], "pct_rank": []} for m in ("xvascot", "xvascot_v1", "raw_attn", "saliency", "random")}
    dele = {"signature": {"d": [], "flip": []}, "random_same_count": {"d": [], "flip": []}}
    n_used = 0
    for i in np.where(y == cls)[0]:
        seq = np.asarray(ctx["Xte"][i], dtype=np.int64)
        nonpad = seq > 0
        pos = np.where(nonpad)[0]
        sig = np.where(seq[nonpad] == op_id)[0]
        p0 = probs(model, seq)
        if len(sig) == 0 or int(p0.argmax()) != cls:
            continue
        n_used += 1
        k = adaptive_k(len(pos), TOPK_RATIO)
        x = torch.from_numpy(seq).unsqueeze(0)
        scores = {"xvascot": xv.importance(x, True)[nonpad],
                  "xvascot_v1": xv1.importance(x, True)[nonpad],
                  "raw_attn": xv.importance(x, False)[nonpad],
                  "saliency": sal.importance(x, True)[nonpad],
                  "random": rng.random(len(pos))}
        for m, sc in scores.items():
            order = np.argsort(sc)[::-1]
            rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
            rec[m]["hit"].append(int(np.any(rank[sig] < k)))
            rec[m]["pct_rank"].append(float(rank[sig].min() / len(order)))
        for name, kill in (("signature", pos[sig]),
                           ("random_same_count", rng.choice(pos, size=len(sig), replace=False))):
            d = seq.copy(); d[kill] = 0
            p1 = probs(model, d)
            dele[name]["d"].append(float(p0[cls] - p1[cls]))
            dele[name]["flip"].append(int(p1.argmax() != cls))
        if n_used >= MAX_N:
            break
    out = {"n": n_used,
           "hit_rate_top5pct": {m: float(np.mean(v["hit"])) if v["hit"] else None for m, v in rec.items()},
           "best_signature_percentile_rank": {m: float(np.median(v["pct_rank"])) if v["pct_rank"] else None
                                              for m, v in rec.items()},
           "deletion": {m: {"dconf": float(np.mean(v["d"])) if v["d"] else None,
                            "flip": float(np.mean(v["flip"])) if v["flip"] else None} for m, v in dele.items()}}
    if n_used and np.any(np.array(dele["signature"]["d"]) - np.array(dele["random_same_count"]["d"])):
        out["wilcoxon_sig_vs_random_p"] = float(wilcoxon(np.array(dele["signature"]["d"]),
                                                         np.array(dele["random_same_count"]["d"]),
                                                         alternative="greater").pvalue)
    return out


def greedy_counterfactual(ctx, model, greedy_cls, rng):
    tok = ctx["tok"].word_index
    ids = [tok[h] for h in RELEASE if h in tok and tok[h] < CONFIG["num_words"]]
    y = np.asarray(ctx["yte"])
    rows = {"remove_release": [], "remove_random": [], "flip_release": [], "flip_random": []}
    n = 0
    for i in np.where(y == 0)[0]:
        seq = np.asarray(ctx["Xte"][i], dtype=np.int64)
        nonpad = seq > 0
        pos = np.where(nonpad)[0]
        rel = pos[np.isin(seq[nonpad], ids)]
        p0 = probs(model, seq)
        if len(rel) == 0 or int(p0.argmax()) != 0:
            continue
        n += 1
        for name, kill in (("release", rel), ("random", rng.choice(pos, size=len(rel), replace=False))):
            d = seq.copy(); d[kill] = 0
            p1 = probs(model, d)
            rows[f"remove_{name}"].append(float(p1[greedy_cls] - p0[greedy_cls]))
            rows[f"flip_{name}"].append(int(p1.argmax() == greedy_cls))
        if n >= MAX_N:
            break
    a, b = np.array(rows["remove_release"]), np.array(rows["remove_random"])
    return {"n": n,
            "dP_greedy_remove_release": float(a.mean()) if n else None,
            "dP_greedy_remove_random": float(b.mean()) if n else None,
            "to_greedy_rate_release": float(np.mean(rows["flip_release"])) if n else None,
            "to_greedy_rate_random": float(np.mean(rows["flip_random"])) if n else None,
            "wilcoxon_p": float(wilcoxon(a, b, alternative="greater").pvalue) if n and np.any(a - b) else None}


def specificity(ctx, model, n_classes, rng, ops=("f1", "ff")):
    """Signature hit rate per (opcode, explained class) over correctly predicted contracts."""
    from collections import Counter
    word = ctx["tok"].word_index
    xv = XVASCOT(model, ctx["tok"], CONFIG, DEVICE, "l2")
    freq = Counter(t for s in ctx["Xtr"] for t in np.asarray(s).tolist() if t)
    y = np.asarray(ctx["yte"])
    out = {}
    for op in ops:
        op_id = word.get(op)
        res = {}
        for cls in range(n_classes):
            hits = {"xvascot": [], "rare_corpus": [], "random": []}
            for i in np.where(y == cls)[0]:
                seq = np.asarray(ctx["Xte"][i], dtype=np.int64)
                nonpad = seq > 0
                toks = seq[nonpad]
                sig = np.where(toks == op_id)[0]
                if len(sig) == 0 or int(probs(model, seq).argmax()) != cls:
                    continue
                xv.target = cls
                k = adaptive_k(len(toks), TOPK_RATIO)
                sc = {"xvascot": xv.importance(torch.from_numpy(seq).unsqueeze(0), True)[nonpad],
                      "rare_corpus": -np.array([freq.get(int(t), 0) for t in toks], float)
                                     + rng.random(len(toks)) * 1e-3,
                      "random": rng.random(len(toks))}
                for m, v in sc.items():
                    hits[m].append(int(np.isin(sig, np.argsort(v)[::-1][:k]).any()))
                if len(hits["xvascot"]) >= MAX_N:
                    break
            res[cls] = {m: (float(np.mean(v)) if v else None) for m, v in hits.items()}
            res[cls]["n"] = len(hits["xvascot"])
        out[op] = res
    return out


def main():
    path = RESULTS / "semantic_tests.json"     # v3: X-VASCOT v2 (cig) scoring
    allres = json.loads(path.read_text()) if path.exists() else {}
    for mode in sys.argv[1:]:
        res = {}
        for seed in trained_seeds(mode):
            ctx = prepare(mode, seed=seed, log=lambda *a: None)
            m = build_vascot(mode)
            m.load_state_dict(torch.load(CKPTS / f"{mode}_seedsweep_vascot_{seed}.pt",
                                         map_location=DEVICE, weights_only=True))
            m.eval()
            rng = np.random.default_rng(seed)
            r = {}
            if mode.startswith("ii4"):
                r["T1_suicidal_selfdestruct"] = signature(ctx, m, 1, "ff", rng)
                r["T2_prodigal_call"] = signature(ctx, m, 2, "f1", rng)
                r["T3_greedy_counterfactual"] = greedy_counterfactual(ctx, m, 3, rng)
                r["T4_specificity"] = specificity(ctx, m, 4, rng)
            else:
                r["T3_greedy_counterfactual"] = greedy_counterfactual(ctx, m, 1, rng)
            res[seed] = r
            print(f"\n[{mode} seed {seed}]")
            for t, v in r.items():
                print(f"  {t}: {json.dumps(v)}", flush=True)
        allres[mode] = res
        path.write_text(json.dumps(allres, indent=2))


if __name__ == "__main__":
    main()
