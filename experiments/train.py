"""Train and evaluate VASCOT (and the LSTM baseline) for one data mode.

The paper uses stage 7, which trains one model per seed (random and
deduplicated splits) or per fold (unseen-family splits):

    python -m experiments.train --mode disasm --stages 7
    python -m experiments.train --mode ii4_full_group --stages 7 --no-lstm
    python -m experiments.train --mode iir_full --stages 7 --seeds 42,123,456 --no-lstm

Modes are defined in xvascot/config.py. Stages 1-6 come from the original
X-VASCOT study (single seed-42 model, ablation, SHAP comparison, case study) and
are kept for completeness; every stage writes its JSON file to results/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

# Required by cuBLAS for deterministic GEMMs; must be set before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from xvascot.config import (CKPTS, CONFIG, LOGS, MODES, RESULTS, SEED, SEEDS,
                            TOPK_RATIO, mode_cfg)
from xvascot.data import build_corpus, load_dataset_i, make_tokenizer
from xvascot.models import LSTMModel, VASCOTModel
from xvascot.train import evaluate_model, make_loader, train_model
from xvascot.xai import (XVASCOT, adaptive_k, attention_entropy, deletion_test,
                         topk_coverage)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tee:
    """Write progress to stdout and to a log file at once."""

    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    # Same seed previously gave 94.79% vs 96.32% for the same model; pin the
    # kernels that can be pinned (warn_only: a few CUDA backward ops have no
    # deterministic implementation and will still vary slightly).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def save(name: str, obj, mode: str):
    path = RESULTS / f"{mode}_{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return path


def strip(res: dict) -> dict:
    """Metrics only - drop the large prediction arrays before serialising."""
    return {k: v for k, v in res.items() if k not in ("preds", "labels", "probs")}


# ── shared setup ─────────────────────────────────────────────────────────────
def prepare(mode: str, seed: int = SEED, log=print):
    mc = mode_cfg(mode)
    variable = mc["truncate_to"] is None

    ds = mc.get("dataset", "i")
    log(f"loading Dataset {ds.upper()}  (mode={mode}, windowing={'on' if variable else 'off'}, "
        f"disassemble={mc.get('disassemble', False)}, dedup={mc.get('dedup', False)})")
    if ds == "i":
        df = load_dataset_i()
    else:
        from xvascot.data import load_dataset_ii_xls
        df = load_dataset_ii_xls(binary=(ds == "iir"))
    if mc.get("dedup", False):
        from xvascot.data import dedup_frame
        n0 = len(df)
        df = dedup_frame(df)
        log(f"  deduplicated: {n0} -> {len(df)} distinct bytecodes")
    if mc.get("disassemble", False) and ds != "i":
        raise ValueError("Datasets II/II-r are already opcode-only (PUSH operands stripped by the "
                         "dataset authors); PUSH-aware disassembly would delete real instructions")
    if mc.get("disassemble", False):
        from xvascot.data import disassemble_frame
        before = df["OPCODE"].str.count(" ").mean() + 1
        df = disassemble_frame(df)
        after = df["OPCODE"].str.count(" ").mean() + 1
        log(f"  disassembled: {before:.0f} -> {after:.0f} tokens per contract "
            f"({100*(1-after/before):.1f}% were PUSH operands, not instructions)")
    # Split first, then fit the vocabulary on the training contracts only, so
    # validation/test token frequencies never influence the 150-token vocabulary.
    y_all = df["LABEL"].to_numpy().astype(np.int64)
    idx = np.arange(len(y_all))
    if mc.get("split") == "group":
        # 5-fold family-grouped cross-validation: near-duplicate families
        # (families.py) never straddle train/val/test. The five SEEDS index the
        # five folds, so a stage-7 sweep predicts every contract exactly once,
        # each time with a model that never saw its family.
        from sklearn.model_selection import StratifiedGroupKFold
        fam = np.load(RESULTS / ("families.npy" if ds == "i" else f"families_{ds}.npy"))
        fold = SEEDS.index(seed) if seed in SEEDS else 0
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        tr_i, te_i = list(outer.split(idx, y_all, fam))[fold]
        # validation: a family-disjoint ~15 % of the training pool that holds
        # both classes and not the single largest family (463 contracts)
        biggest = np.bincount(fam).argmax()
        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
        for a, b in inner.split(tr_i, y_all[tr_i], fam[tr_i]):
            if biggest not in fam[tr_i[b]] and len(set(y_all[tr_i[b]])) == len(set(y_all)):
                break
        tr_i, va_i = tr_i[a], tr_i[b]
    else:
        tr_i, te_i = train_test_split(idx, test_size=0.17, random_state=seed,
                                      stratify=y_all)
        tr_i, va_i = train_test_split(tr_i, test_size=0.15 / 0.83, random_state=seed,
                                      stratify=y_all[tr_i])
    tok = make_tokenizer(df.iloc[tr_i])
    X, y, lengths = build_corpus(df, tok, truncate_to=mc["truncate_to"])
    log(f"  contracts={len(y)}  vocab(train)={len(tok.word_index)}  "
        f"token length mean={lengths.mean():.0f} max={lengths.max()}")

    def take(ix):
        return ([X[i] for i in ix] if variable else X[ix]), y[ix]

    (Xtr, ytr), (Xva, yva), (Xte, yte) = take(tr_i), take(va_i), take(te_i)
    log(f"  train={len(ytr)} val={len(yva)} test={len(yte)} "
        f"(test class counts: {dict(sorted(Counter(yte.tolist()).items()))})")

    bs = CONFIG["batch_size"]
    # VASCOT applies no key_padding_mask, so padded positions take part in both
    # self-attention and attention pooling: a contract's prediction shifts with
    # however much padding its batch happens to carry (19.3% of fixed-mode
    # predictions flip; see padding_sensitivity.py). Variable-length modes are
    # therefore evaluated one contract at a time, which adds no padding at all
    # and is the only padding-free measurement available without changing the
    # architecture. Repro mode is unaffected - every sequence is exactly 2048.
    eval_bs = 1 if variable else bs
    loaders = {
        "train": make_loader(Xtr, ytr, bs, True, variable, seed),
        "val":   make_loader(Xva, yva, eval_bs, False, variable, seed),
        "test":  make_loader(Xte, yte, eval_bs, False, variable, seed),
    }
    return dict(mc=mc, variable=variable, tok=tok, loaders=loaders,
                Xte=Xte, yte=yte, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, lengths=lengths,
                te_rows=te_i)


def model_config(mode: str) -> dict:
    """CONFIG with the number of output classes this mode needs (2, or 4 for Dataset II)."""
    return {**CONFIG, "output_dim": mode_cfg(mode).get("n_classes", 2)}


def trained_seeds(mode: str) -> list[int]:
    """Seeds (folds) of SEEDS that have a stage-7 checkpoint for this mode."""
    return [s for s in SEEDS if (CKPTS / f"{mode}_seedsweep_vascot_{s}.pt").exists()]


def build_vascot(mode: str) -> VASCOTModel:
    agg = "mean_pooled" if mode == "repro" else "mean_prob"
    return VASCOTModel(model_config(mode), window_agg=agg).to(DEVICE)


def load_vascot(mode: str) -> VASCOTModel:
    m = build_vascot(mode)
    m.load_state_dict(torch.load(CKPTS / f"{mode}_vascot_seed{SEED}.pt",
                                 map_location=DEVICE, weights_only=True))
    return m.to(DEVICE).eval()


# ── stages ───────────────────────────────────────────────────────────────────
def stage1(mode, ctx, log):
    log("\n" + "=" * 70 + "\nSTAGE 1  training\n" + "=" * 70)
    out = {}
    set_seed(SEED)
    lstm = LSTMModel(CONFIG["num_words"], CONFIG["embedding_dim"],
                     CONFIG["hidden_dim"], model_config(mode)["output_dim"],
                     CONFIG["dropout"]).to(DEVICE)
    t0 = time.time()
    train_model(lstm, ctx["loaders"]["train"], ctx["loaders"]["val"], CONFIG,
                DEVICE, "LSTM", log)
    out["lstm"] = strip(evaluate_model(lstm, ctx["loaders"]["test"], DEVICE,
                                       "LSTM baseline", log))
    out["lstm"]["train_seconds"] = time.time() - t0
    torch.save(lstm.state_dict(), CKPTS / f"{mode}_lstm_seed{SEED}.pt")
    del lstm
    torch.cuda.empty_cache()

    set_seed(SEED)
    vas = build_vascot(mode)
    t0 = time.time()
    train_model(vas, ctx["loaders"]["train"], ctx["loaders"]["val"], CONFIG,
                DEVICE, "VASCOT", log)
    res = evaluate_model(vas, ctx["loaders"]["test"], DEVICE, "VASCOT", log)
    out["vascot"] = strip(res)
    out["vascot"]["train_seconds"] = time.time() - t0
    torch.save(vas.state_dict(), CKPTS / f"{mode}_vascot_seed{SEED}.pt")
    np.save(RESULTS / f"{mode}_vascot_preds.npy", res["preds"])
    np.save(RESULTS / f"{mode}_vascot_labels.npy", res["labels"])
    np.save(RESULTS / f"{mode}_vascot_probs.npy", res["probs"])
    save("stage1_classification", out, mode)
    return out


def stage2(mode, ctx, log):
    log("\n" + "=" * 70 + "\nSTAGE 2  prediction agreement\n" + "=" * 70)
    vas = load_vascot(mode)
    ref = np.load(RESULTS / f"{mode}_vascot_preds.npy")
    got = []
    with torch.no_grad():
        for X_b, _ in ctx["loaders"]["test"]:
            got.extend(vas(X_b.to(DEVICE)).argmax(1).cpu().numpy())
    got = np.array(got)
    agree = bool((got == ref).all())
    log(f"  identical predictions: {agree}  ({int((got==ref).sum())}/{len(ref)})")
    log("  note: agreement is guaranteed by construction - the explainer runs "
        "after the forward pass and never touches the weights.")
    out = {"agreement": agree, "n": len(ref),
           "n_matching": int((got == ref).sum())}
    save("stage2_agreement", out, mode)
    return out


def _explainer(mode, model, ctx, grad_reduce=None):
    return XVASCOT(model, ctx["tok"], CONFIG, DEVICE,
                   grad_reduce=grad_reduce or ctx["mc"]["grad_reduce"])


def stage3(mode, ctx, log):
    log("\n" + "=" * 70 + "\nSTAGE 3  ablation (explanation quality)\n" + "=" * 70)
    vas = load_vascot(mode)
    xv = _explainer(mode, vas, ctx)
    n = ctx["mc"]["n_ablation"]
    vuln = np.where(np.asarray(ctx["yte"]) == 1)[0][:n]
    log(f"  measuring on n={len(vuln)} vulnerable contracts")

    acc = {c: {"entropy": [], "coverage": [], "k": []} for c in ("C", "Cp", "D")}
    for j, i in enumerate(vuln):
        seq = np.asarray(ctx["Xte"][i], dtype=np.int64)
        x = torch.from_numpy(seq).unsqueeze(0)
        nonpad = seq > 0
        L = int(nonpad.sum())
        ka = adaptive_k(L, TOPK_RATIO)

        raw = xv.importance(x, use_grad=False)[nonpad]
        grad = xv.importance(x, use_grad=True)[nonpad]

        # C: raw attention, fixed k=10   C': raw attention, adaptive k
        for cfg, sc, k in (("C", raw, 10), ("Cp", raw, ka), ("D", grad, ka)):
            acc[cfg]["entropy"].append(attention_entropy(sc))
            acc[cfg]["coverage"].append(topk_coverage(sc, min(k, len(sc))))
            acc[cfg]["k"].append(min(k, len(sc)))
        if (j + 1) % 10 == 0:
            log(f"    {j+1}/{len(vuln)}  L={L} k={ka}")

    out = {c: {m: float(np.mean(v)) for m, v in d.items()} for c, d in acc.items()}
    out["n"] = len(vuln)
    log("\n  cfg  attention     k-rule      entropy   coverage   avg k")
    for c, name, krule in (("C", "raw", "fixed 10"), ("Cp", "raw", "adaptive"),
                           ("D", "grad", "adaptive")):
        o = out[c]
        log(f"  {c:<4} {name:<12} {krule:<11} {o['entropy']:7.4f}  "
            f"{o['coverage']:8.4f}  {o['k']:6.1f}")
    save("stage3_ablation", out, mode)
    return out


def stage4(mode, ctx, log):
    log("\n" + "=" * 70 + "\nSTAGE 4  faithfulness deletion test\n" + "=" * 70)
    vas = load_vascot(mode)
    xv = _explainer(mode, vas, ctx)
    n = ctx["mc"]["n_faithful"]
    out = {}
    for cfg, use_grad, fk in (("C", False, 10), ("Cp", False, None), ("D", True, None)):
        log(f"  config {cfg}  (grad={use_grad}, k={'10' if fk else 'adaptive'})")
        mean, drops, flip, extra = deletion_test(xv, ctx["Xte"], ctx["yte"],
                                                 n_samples=n, use_grad=use_grad,
                                                 fixed_k=fk, ratio=TOPK_RATIO, log=log)
        out[cfg] = {"delta_conf": mean, "flip_rate": flip, "n": len(drops),
                    **extra, "drops": drops}
        log(f"    -> dconf={mean:+.4f}  flip={flip*100:.1f}% "
            f"(of {extra['n_pred_vuln']} predicted vulnerable; "
            f"old any-nonvuln rate {extra['flip_rate_any_nonvuln']*100:.1f}%)")
    save("stage4_faithfulness", out, mode)
    return out


def stage6(mode, ctx, log):
    log("\n" + "=" * 70 + "\nSTAGE 6  case study\n" + "=" * 70)
    vas = load_vascot(mode)
    xv = _explainer(mode, vas, ctx)
    preds = np.load(RESULTS / f"{mode}_vascot_preds.npy")
    labels = np.load(RESULTS / f"{mode}_vascot_labels.npy")

    picks = {
        "true_positive":  np.where((labels == 1) & (preds == 1))[0],
        "true_negative":  np.where((labels == 0) & (preds == 0))[0],
        "false_positive": np.where((labels == 0) & (preds == 1))[0],
    }
    out = {}
    for name, idxs in picks.items():
        if not len(idxs):
            log(f"  {name}: none available")
            continue
        i = int(idxs[0])
        seq = np.asarray(ctx["Xte"][i], dtype=np.int64)
        r = xv.explain(torch.from_numpy(seq).unsqueeze(0), ratio=TOPK_RATIO)
        from collections import Counter
        fam = Counter(m for m, _ in r["top"])
        out[name] = {
            "test_index": i, "pred": r["pred"], "conf": r["conf"],
            "L": r["L"], "k": r["k"], "entropy": r["entropy"],
            "coverage": r["coverage"],
            "top15": [(m, s) for m, s in r["top"][:15]],
            "top_families": fam.most_common(10),
        }
        log(f"\n  {name}: idx={i} pred={r['pred']} conf={r['conf']*100:.1f}% "
            f"L={r['L']} k={r['k']}")
        for rank, (m, s) in enumerate(r["top"][:10], 1):
            log(f"    {rank:2d}. {m:<16} {s:.6f}")
    save("stage6_case_study", out, mode)
    return out


# Set from the command line (--seeds, --no-lstm); defaults reproduce the
# original five-seed LSTM + VASCOT sweep.
STAGE7_SEEDS = list(SEEDS)
STAGE7_LSTM = True


def stage7(mode, ctx, log):
    log("\n" + "=" * 70 + "\nSTAGE 7  multi-seed statistical run\n" + "=" * 70)
    out = {"lstm": [], "vascot": [], "seeds": list(STAGE7_SEEDS)}
    for si, seed in enumerate(STAGE7_SEEDS):
        log(f"\n--- seed {seed} ({si+1}/{len(STAGE7_SEEDS)}) ---")
        c = prepare(mode, seed=seed, log=lambda *a: None)
        if not STAGE7_LSTM:
            out["lstm"].append(None)
        else:
            _train_lstm7(mode, c, seed, out, log)

        set_seed(seed)
        vas = build_vascot(mode)
        train_model(vas, c["loaders"]["train"], c["loaders"]["val"], CONFIG,
                    DEVICE, f"VASCOT-s{seed}", log)
        out["vascot"].append(strip(evaluate_model(vas, c["loaders"]["test"],
                                                  DEVICE, f"VASCOT-s{seed}", log,
                                                  verbose=False)))
        log(f"  VASCOT acc={out['vascot'][-1]['acc']*100:.2f}%")
        # Separate namespace: stage 1's checkpoint is the one every XAI stage
        # loads, so the multi-seed sweep must not overwrite it (seed 42 appears
        # in both and would otherwise clobber it with a different model).
        torch.save(vas.state_dict(), CKPTS / f"{mode}_seedsweep_vascot_{seed}.pt")
        del vas
        torch.cuda.empty_cache()

    log("\n  metric        LSTM                VASCOT")
    summary = {}
    lstm_rows = [r for r in out["lstm"] if r is not None]
    for key, name in (("acc", "accuracy"), ("prec", "precision"),
                      ("rec", "recall"), ("f1", "F1 (vuln)"), ("auc", "AUC-ROC")):
        lm = np.mean([r[key] for r in lstm_rows]) if lstm_rows else float("nan")
        ls = np.std([r[key] for r in lstm_rows]) if lstm_rows else float("nan")
        vm = np.mean([r[key] for r in out["vascot"]])
        vs = np.std([r[key] for r in out["vascot"]])
        summary[key] = {"lstm_mean": lm, "lstm_std": ls,
                        "vascot_mean": vm, "vascot_std": vs}
        log(f"  {name:<12}  {lm:.4f} +/- {ls:.4f}    {vm:.4f} +/- {vs:.4f}")
    out["summary"] = summary
    save("stage7_seeds", out, mode)
    return out


def _train_lstm7(mode, c, seed, out, log):
    set_seed(seed)
    lstm = LSTMModel(CONFIG["num_words"], CONFIG["embedding_dim"],
                     CONFIG["hidden_dim"], model_config(mode)["output_dim"],
                     CONFIG["dropout"]).to(DEVICE)
    train_model(lstm, c["loaders"]["train"], c["loaders"]["val"], CONFIG,
                DEVICE, f"LSTM-s{seed}", log)
    out["lstm"].append(strip(evaluate_model(lstm, c["loaders"]["test"],
                                            DEVICE, f"LSTM-s{seed}", log,
                                            verbose=False)))
    log(f"  LSTM   acc={out['lstm'][-1]['acc']*100:.2f}%")
    del lstm
    torch.cuda.empty_cache()


STAGES = {1: stage1, 2: stage2, 3: stage3, 4: stage4, 6: stage6, 7: stage7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--shap-nsamples", type=int, default=64,
                    help="SHAP GradientExplainer sample budget (notebook used "
                         "the library default of 200)")
    ap.add_argument("--seeds", default=None,
                    help="stage 7 only: comma-separated subset of SEEDS (default all five)")
    ap.add_argument("--no-lstm", action="store_true",
                    help="stage 7 only: train VASCOT only")
    args = ap.parse_args()
    global STAGE7_SEEDS, STAGE7_LSTM
    if args.seeds:
        STAGE7_SEEDS = [int(x) for x in args.seeds.split(",")]
    STAGE7_LSTM = not args.no_lstm

    log = Tee(LOGS / f"{args.mode}.log")
    log(f"\n{'#'*70}\n# X-VASCOT re-run | mode={args.mode} | device={DEVICE}")
    if torch.cuda.is_available():
        log(f"# GPU: {torch.cuda.get_device_name(0)}")
    log(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'#'*70}")

    wanted = (sorted([*STAGES, 5]) if args.stages == "all"
              else [int(s) for s in args.stages.split(",")])

    ctx = prepare(args.mode, log=log)
    for s in wanted:
        if s == 5:
            from experiments.shap_compare import stage5
            # repro inputs are already a fixed 2048, so "full" would duplicate it
            lengths = (256, 2048) if args.mode == "repro" else (256, 2048, None)
            stage5(args.mode, ctx, log, DEVICE,
                   n_samples=ctx["mc"]["n_shap"], lengths=lengths,
                   nsamples_shap=args.shap_nsamples)
            continue
        t0 = time.time()
        STAGES[s](args.mode, ctx, log)
        log(f"\n[stage {s} done in {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
