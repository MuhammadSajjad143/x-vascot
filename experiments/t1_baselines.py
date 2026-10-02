"""T1 with standard attribution baselines: comprehensiveness AND sufficiency.

Ten selectors pick k = ceil(0.05 L) positions per contract:

  xvascot      X-VASCOT v2: contrastive Integrated Gradients (xai.py, score="cig")
  xvascot_v1   head/query-mean received attention x ||dz/de||_2 (original Eq. 2)
  raw_attn     received attention alone (= attention rollout for a 1-layer model)
  saliency     ||dz/de||_2 alone
  grad_x_input sum_d e * dz/de (signed; positive = supports the explained class)
  int_grad     Integrated Gradients on z_c, zero-embedding (= deletion) baseline, 16 midpoint steps
  pool_attn    VASCOT's own attention-pooling weights
  rare_corpus  tokens rarest in the training split
  rare_local   tokens rarest within this contract
  frequent     tokens most frequent in the training split
  random

All gradients are taken w.r.t. the token embedding before positional encoding,
per window, and windows are merged by element-wise max (as X-VASCOT does).

Sample: test contracts whose true class is vulnerable (>0) and that the model
predicts correctly, first MAX_N per model; the explained class is the true class.

  comprehensiveness  p0(c) - p(c | top-k zeroed)          higher = better
  sufficiency        p0(c) - p(c | only top-k kept)       lower  = better
  verdict_removed    share whose argmax leaves c after deletion

Reported as the mean over models of the per-model mean, with the std across models.

    python t1_baselines.py disasm iir_full iir_full_dedup iir_full_group ii4_full ...
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from experiments.train import DEVICE, build_vascot, prepare, trained_seeds
from xvascot.config import CKPTS, RESULTS, TOPK_RATIO
from xvascot.xai import adaptive_k, window_signed_attribution

MAX_N = 60
IG_STEPS = 16
METHODS = ("xvascot", "xvascot_v1", "raw_attn", "saliency", "grad_x_input", "int_grad", "pool_attn",
           "rare_corpus", "rare_local", "frequent", "random")


@torch.no_grad()
def probs(model, seq):
    return F.softmax(model(torch.from_numpy(seq).unsqueeze(0).to(DEVICE)).float(), 1)[0].cpu().numpy()


def _window_attributions(model, ids, cls):
    """All model-based scores for one window -> dict of (W,) arrays."""
    emb0 = model.embedding(ids).detach()
    emb = emb0.clone().requires_grad_(True)
    x = model.pos_enc(emb)
    h, attn = model.encoder(x, return_attn=True)
    pool_w = F.softmax(torch.tanh(model.pool.attn(h)), dim=1)            # (1, W, 1)
    z = model.classify((pool_w * h).sum(1))[0, cls]
    model.zero_grad(set_to_none=True)
    z.backward()
    g = emb.grad.detach()[0]
    a = attn.detach()[0].mean(0).mean(0)
    out = {"raw_attn": a, "saliency": g.norm(dim=-1), "xvascot_v1": a * g.norm(dim=-1),
           "grad_x_input": (g * emb0[0]).sum(-1), "pool_attn": pool_w.detach()[0, :, 0]}
    out["int_grad"] = window_signed_attribution(model, emb0, cls, False, IG_STEPS)
    out["xvascot"] = window_signed_attribution(model, emb0, cls, True, IG_STEPS)
    return {k: v.float().cpu().numpy() for k, v in out.items()}


def model_scores(model, seq, cls):
    W = model.config["max_length"]
    L = len(seq)
    x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
    merged = {}
    for s in model.window_starts(L):
        win = _window_attributions(model, x[:, s:s + W], cls)
        for k, v in win.items():
            cur = merged.setdefault(k, np.full(L, -np.inf))
            seg = cur[s:s + len(v)]
            cur[s:s + len(v)] = np.maximum(seg, v[:len(seg)])
    return merged


def run_mode(mode):
    per_model = {m: {"comp": [], "suff": [], "flip": []} for m in METHODS}
    contracts = {m: {"comp": [], "suff": [], "flip": []} for m in METHODS}   # per model, per contract
    for seed in trained_seeds(mode):
        ctx = prepare(mode, seed=seed, log=lambda *a: None)
        model = build_vascot(mode)
        model.load_state_dict(torch.load(CKPTS / f"{mode}_seedsweep_vascot_{seed}.pt",
                                         map_location=DEVICE, weights_only=True))
        model.eval()
        rng = np.random.default_rng(seed)
        freq = Counter(t for s in ctx["Xtr"] for t in np.asarray(s).tolist() if t)
        y = np.asarray(ctx["yte"])
        acc = {m: {"comp": [], "suff": [], "flip": []} for m in METHODS}
        n = 0
        for i in np.where(y > 0)[0]:
            seq = np.asarray(ctx["Xte"][i], dtype=np.int64)
            cls = int(y[i])
            p0 = probs(model, seq)
            if int(p0.argmax()) != cls:
                continue
            nonpad = seq > 0
            pos = np.where(nonpad)[0]
            toks = seq[nonpad]
            k = adaptive_k(len(pos), TOPK_RATIO)
            tie = rng.random(len(toks)) * 1e-3
            local = Counter(toks.tolist())
            fr = np.array([freq.get(int(t), 0) for t in toks], float)
            sc = {k_: v[nonpad] for k_, v in model_scores(model, seq, cls).items()}
            sc.update(rare_corpus=-fr + tie, frequent=fr + tie, random=rng.random(len(toks)),
                      rare_local=-np.array([local[int(t)] for t in toks], float) + tie)
            for m in METHODS:
                top = pos[np.argsort(sc[m])[::-1][:k]]
                d = seq.copy(); d[top] = 0
                keep = np.zeros_like(seq); keep[top] = seq[top]
                pd, pk = probs(model, d), probs(model, keep)
                acc[m]["comp"].append(float(p0[cls] - pd[cls]))
                acc[m]["suff"].append(float(p0[cls] - pk[cls]))
                acc[m]["flip"].append(int(pd.argmax() != cls))
            n += 1
            if n >= MAX_N:
                break
        for m in METHODS:
            for key in ("comp", "suff", "flip"):
                per_model[m][key].append(float(np.mean(acc[m][key])) if n else float("nan"))
                contracts[m][key].append(acc[m][key])
        print(f"[{mode}] seed {seed}: n={n}  " + "  ".join(
            f"{m} {per_model[m]['comp'][-1]:+.3f}" for m in METHODS), flush=True)
    return {"n_models": len(trained_seeds(mode)), "max_n_per_model": MAX_N, "per_contract": contracts,
            **{m: {key: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "per_model": v}
                   for key, v in d.items()} for m, d in per_model.items()}}


def main():
    path = RESULTS / "t1_baselines.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    for mode in sys.argv[1:]:
        r = run_mode(mode)
        allres[mode] = r
        path.write_text(json.dumps(allres, indent=2))
        print(f"\n===== {mode}  ({r['n_models']} models)\n"
              f"  {'method':<13}{'comprehensiveness':>20}{'sufficiency':>18}{'verdict removed':>18}")
        for m in METHODS:
            d = r[m]
            print(f"  {m:<13}{d['comp']['mean']:+10.3f} ± {d['comp']['std']:.3f}"
                  f"{d['suff']['mean']:+10.3f} ± {d['suff']['std']:.3f}"
                  f"{100*d['flip']['mean']:10.1f}% ± {100*d['flip']['std']:.1f}")


if __name__ == "__main__":
    main()
