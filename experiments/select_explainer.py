"""Choose X-VASCOT v2's scoring rule on VALIDATION splits only (test sets untouched).

Candidates: attn_grad (v1), ig, cig (contrastive IG), gxi, cgxi (contrastive grad x input).
Selection data: validation contracts of Dataset I (disasm, 5 seeds) and Dataset II
(ii4_full_dedup, 5 seeds), correctly predicted vulnerable, <= 40 per model.

Criterion, fixed before running:
  primary    mean over the two datasets of (comprehensiveness - sufficiency)
  tie-break  (within 0.01) signature specificity on Dataset II:
             hit(own class) - mean hit(other classes), for SELFDESTRUCT/suicidal and CALL/prodigal

    python -m experiments.select_explainer
"""
import json
import numpy as np, torch
from experiments.train import DEVICE, build_vascot, prepare, trained_seeds
from xvascot.config import CKPTS, CONFIG, RESULTS, TOPK_RATIO
from xvascot.xai import XVASCOT, adaptive_k
from experiments.t1_baselines import probs

CANDS = ("attn_grad", "ig", "cig", "gxi", "cgxi")
MAX_N = 40
SIG = {"ff": 1, "f1": 2}
res = {}
for mode in ("disasm", "ii4_full_dedup"):
    per = {c: {"comp": [], "suff": []} for c in CANDS}
    hits = {c: {} for c in CANDS}
    for seed in trained_seeds(mode):
        ctx = prepare(mode, seed=seed, log=lambda *a: None)
        m = build_vascot(mode)
        m.load_state_dict(torch.load(CKPTS / f"{mode}_seedsweep_vascot_{seed}.pt", map_location=DEVICE, weights_only=True)); m.eval()
        xvs = {c: XVASCOT(m, ctx["tok"], CONFIG, DEVICE, "l2", score=c) for c in CANDS}
        word, y = ctx["tok"].word_index, np.asarray(ctx["yva"])
        acc = {c: {"comp": [], "suff": []} for c in CANDS}
        cnt = {}
        for i in np.random.default_rng(seed).permutation(len(y)):
            cls = int(y[i]); seq = np.asarray(ctx["Xva"][i], dtype=np.int64)
            p0 = probs(m, seq)
            if int(p0.argmax()) != cls:
                continue
            need_del = cls > 0 and cnt.get("del", 0) < MAX_N
            sig_ops = [op for op in SIG if mode.startswith("ii4") and word.get(op) in set(seq.tolist())
                       and cnt.get((op, cls), 0) < MAX_N]
            if not need_del and not sig_ops:
                continue
            nonpad = seq > 0; pos = np.where(nonpad)[0]; k = adaptive_k(len(pos), TOPK_RATIO)
            x = torch.from_numpy(seq).unsqueeze(0)
            for c, xv in xvs.items():
                xv.target = cls
                order = np.argsort(xv.importance(x)[nonpad])[::-1][:k]
                if need_del:
                    top = pos[order]
                    d = seq.copy(); d[top] = 0
                    kp = np.zeros_like(seq); kp[top] = seq[top]
                    acc[c]["comp"].append(float(p0[cls] - probs(m, d)[cls]))
                    acc[c]["suff"].append(float(p0[cls] - probs(m, kp)[cls]))
                for op in sig_ops:
                    hit = int(np.isin(np.where(seq[nonpad] == word[op])[0], order).any())
                    hits[c].setdefault(f"{op}|{cls}", []).append(hit)
            if need_del: cnt["del"] = cnt.get("del", 0) + 1
            for op in sig_ops: cnt[(op, cls)] = cnt.get((op, cls), 0) + 1
        for c in CANDS:
            for key in ("comp", "suff"):
                per[c][key].append(float(np.mean(acc[c][key])))
        print(f"[{mode}] seed {seed}: " + "  ".join(f"{c} {per[c]['comp'][-1]-per[c]['suff'][-1]:+.3f}" for c in CANDS), flush=True)
    res[mode] = {c: {"comp": float(np.mean(per[c]["comp"])), "suff": float(np.mean(per[c]["suff"])),
                     "comp_minus_suff": float(np.mean(per[c]["comp"]) - np.mean(per[c]["suff"]))} for c in CANDS}
    if mode.startswith("ii4"):
        for c in CANDS:
            spec = {}
            for op, own in SIG.items():
                rate = {cl: np.mean(hits[c].get(f"{op}|{cl}", [np.nan])) for cl in range(4)}
                spec[op] = float(rate[own] - np.nanmean([rate[cl] for cl in range(4) if cl != own]))
                res[mode][c][f"hit_{op}_per_class"] = [None if np.isnan(rate[cl]) else float(rate[cl]) for cl in range(4)]
            res[mode][c]["specificity"] = float(np.mean(list(spec.values())))

score = {c: float(np.mean([res[md][c]["comp_minus_suff"] for md in res])) for c in CANDS}
best = max(score.values())
tied = [c for c in CANDS if best - score[c] <= 0.01]
chosen = max(tied, key=lambda c: res["ii4_full_dedup"][c]["specificity"])
out = {"per_mode": res, "primary_score": score, "tied_within_0.01": tied, "chosen": chosen}
(RESULTS / "select_explainer.json").write_text(json.dumps(out, indent=2))
print("\n candidate   comp-suff(I)  comp-suff(II)  primary  specificity(II)")
for c in CANDS:
    print(f" {c:<10} {res['disasm'][c]['comp_minus_suff']:+12.3f} {res['ii4_full_dedup'][c]['comp_minus_suff']:+13.3f} "
          f"{score[c]:+8.3f} {res['ii4_full_dedup'][c]['specificity']:+12.3f}")
print(f"\nCHOSEN: {chosen}  (tied: {tied})")
