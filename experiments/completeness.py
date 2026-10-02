"""Completeness of X-VASCOT (contrastive IG, S=16) over the T1 sample.

For every window of every T1 contract (first 60 correctly classified vulnerable test
contracts per model), compare the sum of attributions with the exact margin change
m_c(E) - m_c(0) between the window and the all-deleted window.

    python -m experiments.completeness disasm iir_full_dedup iir_full_group ii4_full_dedup ii4_full_group
"""
import json, sys
import numpy as np, torch
import torch.nn.functional as F
from experiments.train import DEVICE, build_vascot, prepare, trained_seeds
from xvascot.config import CKPTS, RESULTS
from xvascot.xai import window_signed_attribution, target_logit

MAX_N = 60
out = {}
for mode in sys.argv[1:]:
    err, rel = [], []
    for seed in trained_seeds(mode):
        ctx = prepare(mode, seed=seed, log=lambda *a: None)
        m = build_vascot(mode)
        m.load_state_dict(torch.load(CKPTS / f"{mode}_seedsweep_vascot_{seed}.pt", map_location=DEVICE, weights_only=True)); m.eval()
        y = np.asarray(ctx["yte"]); n = 0
        for i in np.where(y > 0)[0]:
            seq = np.asarray(ctx["Xte"][i], dtype=np.int64); cls = int(y[i])
            x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if int(m(x).argmax(1)) != cls: continue
            W = m.config["max_length"]
            for s in m.window_starts(len(seq)):
                emb0 = m.embedding(x[:, s:s + W]).detach()
                phi = window_signed_attribution(m, emb0, cls, True, 16).detach()
                with torch.no_grad():
                    d = (target_logit(m.classify(m.encode_embedded(emb0)), cls, True)
                         - target_logit(m.classify(m.encode_embedded(torch.zeros_like(emb0))), cls, True)).item()
                err.append(abs(phi.sum().item() - d)); rel.append(abs(phi.sum().item() - d) / max(abs(d), 1e-6))
            n += 1
            if n >= MAX_N: break
    out[mode] = {"windows": len(err), "mean_abs_err": float(np.mean(err)), "median_rel_err": float(np.median(rel)),
                 "mean_rel_err": float(np.mean(rel))}
    print(mode, out[mode], flush=True)
(RESULTS / "completeness.json").write_text(json.dumps(out, indent=2))
allw = sum(v["windows"] for v in out.values())
print("ALL windows", allw, "mean abs err", sum(v["mean_abs_err"] * v["windows"] for v in out.values()) / allw,
      "median rel err (mean of modes)", np.mean([v["median_rel_err"] for v in out.values()]))
