"""Stage 5 - X-VASCOT against SHAP GradientExplainer.

Three corrections over the notebook's version of this experiment:

1. The notebook's "X-VASCOT" column came from ``get_grad_scores_hookfree``, which
   is plain embedding-gradient saliency with no attention weighting at all - it
   skipped ``return_attn=True`` to dodge a hook conflict with SHAP. That is not
   the proposed method. Here the real Config D explainer is used, and SHAP gets
   its own deep copy of the model so its hooks can never touch ours.

2. SHAP ran on CPU because of memory limits. It runs on the GPU here.

3. The notebook compared SHAP at 256 tokens against X-VASCOT at 2048, which
   confounds the method with the sequence length. Both are measured at matched
   lengths, and X-VASCOT is additionally reported on the full sequence, where
   its actual advantage lies.
"""

from __future__ import annotations

import copy
import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xvascot.config import CONFIG, TOPK_RATIO
from xvascot.xai import XVASCOT, adaptive_k, attention_entropy, topk_coverage


class PostEmbeddingModel(nn.Module):
    """VASCOT from the embedding onward, so SHAP attributes to token embeddings.

    Token ids are discrete, so SHAP cannot perturb them directly; attributing to
    the embedding vectors is the standard workaround and is what the notebook
    did too.
    """

    def __init__(self, vascot):
        super().__init__()
        m = copy.deepcopy(vascot)
        self.config = m.config
        self.window_agg = m.window_agg
        self.pos_enc, self.encoder, self.pool = m.pos_enc, m.encoder, m.pool
        self.drop1, self.dense, self.drop2, self.out = m.drop1, m.dense, m.drop2, m.out
        self.window_starts = m.window_starts

    def _head(self, pooled):
        return self.out(self.drop2(F.relu(self.dense(self.drop1(pooled)))))

    def forward(self, emb):
        W = self.config["max_length"]
        L = emb.shape[1]
        if L <= W:
            return F.softmax(self._head(self.pool(self.encoder(self.pos_enc(emb)))), 1)
        pooled = [self.pool(self.encoder(self.pos_enc(emb[:, s:s + W])))
                  for s in self.window_starts(L)]
        # Match the deployed model's window aggregation, so SHAP explains the
        # same function X-VASCOT does.
        if self.window_agg == "mean_prob":
            return torch.stack([F.softmax(self._head(p), 1) for p in pooled], 0).mean(0)
        return F.softmax(self._head(torch.stack(pooled, 0).mean(0)), 1)


def _slice(seq: np.ndarray, seq_len: int | None) -> np.ndarray:
    """Take the trailing ``seq_len`` tokens, matching Keras' truncating='pre'."""
    return seq if seq_len is None else seq[-seq_len:]


def stage5(mode, ctx, log, device, n_samples: int = 20,
           lengths=(256, 2048, None), nsamples_shap: int = 64):
    log("\n" + "=" * 70 + "\nSTAGE 5  X-VASCOT vs SHAP\n" + "=" * 70)
    import shap

    from experiments.train import load_vascot
    log(f"  shap {shap.__version__}  device={device}")

    vascot = load_vascot(mode)
    xv = XVASCOT(vascot, ctx["tok"], CONFIG, device,
                 grad_reduce=ctx["mc"]["grad_reduce"])

    y = np.asarray(ctx["yte"])
    vuln = np.where(y == 1)[0][:n_samples]
    log(f"  n={len(vuln)} vulnerable contracts, "
        f"lengths={['full' if l is None else l for l in lengths]}")

    emb_layer = copy.deepcopy(vascot.embedding).to(device).eval()
    results = {}

    for seq_len in lengths:
        tag = "full" if seq_len is None else str(seq_len)
        log(f"\n  --- sequence length: {tag} ---")

        # Background distribution: non-vulnerable training contracts.
        # GradientExplainer interpolates between background and input, so the
        # two must share a shape. At a fixed seq_len one background serves every
        # contract; on full sequences each contract needs its own, matched to
        # its length.
        bg_idx = np.where(np.asarray(ctx["ytr"]) == 0)[0][:8]
        bg_seqs = [np.asarray(ctx["Xtr"][i], dtype=np.int64) for i in bg_idx]

        def make_bg(target_len: int) -> torch.Tensor:
            """Embed the background contracts, pre-padded/cut to ``target_len``."""
            arr = np.zeros((len(bg_seqs), target_len), dtype=np.int64)
            for r, s in enumerate(bg_seqs):
                s = s[-target_len:]
                arr[r, -len(s):] = s
            with torch.no_grad():
                return emb_layer(torch.from_numpy(arr).to(device))

        post = PostEmbeddingModel(vascot).to(device).eval()
        fixed_len = seq_len is not None
        bg_emb = make_bg(seq_len) if fixed_len else None
        shap_ent, shap_cov, shap_t = [], [], []
        xv_ent, xv_cov, xv_t = [], [], []
        ok = []

        explainer = None
        if fixed_len:
            try:
                explainer = shap.GradientExplainer(post, bg_emb)
            except Exception as e:                                # pragma: no cover
                log(f"    SHAP init failed: {type(e).__name__}: {str(e)[:120]}")

        for n, idx in enumerate(vuln):
            seq = _slice(np.asarray(ctx["Xte"][idx], dtype=np.int64), seq_len)
            nonpad = seq > 0
            if nonpad.sum() == 0:
                continue
            k = adaptive_k(int(nonpad.sum()), TOPK_RATIO)

            if not fixed_len:
                # rebuild a length-matched background for this contract
                try:
                    explainer = shap.GradientExplainer(post, make_bg(len(seq)))
                except Exception as e:
                    log(f"    SHAP init failed at L={len(seq)}: "
                        f"{type(e).__name__}: {str(e)[:90]}")
                    explainer = None

            # ── SHAP ─────────────────────────────────────────────────────────
            if explainer is not None:
                try:
                    with torch.no_grad():
                        x_emb = emb_layer(torch.from_numpy(seq).unsqueeze(0).to(device))
                    torch.cuda.synchronize()
                    t0 = time.time()
                    sv = explainer.shap_values(x_emb, nsamples=nsamples_shap)
                    torch.cuda.synchronize()
                    dt = time.time() - t0

                    arr = sv[1] if isinstance(sv, list) and len(sv) == 2 else (
                        sv[0] if isinstance(sv, list) else sv)
                    arr = np.asarray(arr)
                    while arr.ndim > 3:          # (1, L, D, n_out) -> pick vuln output
                        arr = arr[..., 1]
                    s = np.linalg.norm(arr.reshape(arr.shape[-3:])[0]
                                       if arr.ndim == 3 else arr, axis=-1)
                    s = np.abs(s)[nonpad]
                    shap_ent.append(attention_entropy(s))
                    shap_cov.append(topk_coverage(s, min(k, len(s))))
                    shap_t.append(dt)
                except Exception as e:
                    log(f"    SHAP [{n+1}] idx={idx} failed: "
                        f"{type(e).__name__}: {str(e)[:90]}")
                    continue

            # ── X-VASCOT Config D on the same tokens ────────────────────────
            x = torch.from_numpy(seq).unsqueeze(0)
            torch.cuda.synchronize()
            t0 = time.time()
            s_xv = xv.importance(x, use_grad=True)[nonpad]
            torch.cuda.synchronize()
            xv_t.append(time.time() - t0)
            xv_ent.append(attention_entropy(s_xv))
            xv_cov.append(topk_coverage(s_xv, min(k, len(s_xv))))
            ok.append(int(idx))

            if (n + 1) % 5 == 0:
                log(f"    {n+1}/{len(vuln)} done")

            del x
            gc.collect()
            torch.cuda.empty_cache()

        def agg(v):
            return float(np.mean(v)) if v else None

        entry = {
            "n": len(ok),
            "shap": {"entropy": agg(shap_ent), "coverage": agg(shap_cov),
                     "time_s": agg(shap_t)},
            "xvascot": {"entropy": agg(xv_ent), "coverage": agg(xv_cov),
                        "time_s": agg(xv_t)},
        }
        if entry["shap"]["time_s"] and entry["xvascot"]["time_s"]:
            entry["speedup"] = entry["shap"]["time_s"] / max(entry["xvascot"]["time_s"], 1e-9)
        results[tag] = entry

        s_, x_ = entry["shap"], entry["xvascot"]
        log(f"    {'metric':<16}{'X-VASCOT':>14}{'SHAP':>14}")
        for key, name in (("entropy", "entropy"), ("coverage", "coverage"),
                          ("time_s", "time (s)")):
            xs = f"{x_[key]:.4f}" if x_[key] is not None else "-"
            ss = f"{s_[key]:.4f}" if s_[key] is not None else "-"
            log(f"    {name:<16}{xs:>14}{ss:>14}")
        if "speedup" in entry:
            log(f"    speedup: {entry['speedup']:.1f}x")

        del post, explainer
        gc.collect()
        torch.cuda.empty_cache()

    from experiments.train import save
    save("stage5_shap", results, mode)
    return results
