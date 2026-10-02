"""Part C: which Solidity source code do X-VASCOT's selected instructions come from?

For each test contract the verified source and the runtime source map are taken
from Sourcify (no API key). The dataset stores the on-chain runtime bytecode (cut
at 10,922 bytes by an Excel cell limit), so instruction i of the dataset is
instruction i of the source map. Slither supplies exact function boundaries and
Ether semantics. Every model position is then classified as:

  eth_send      inside a function that can (transitively) send Ether
  payable       inside a payable function / receive / fallback (and not eth_send)
  other_custom  the contract's own remaining code
  std_library   standard or library code (SafeMath, Context, Ownable, ERC20,
                Address, Uniswap interfaces, libraries)
  compiler      compiler-generated (source-map file -1 or outside every function)
  metadata      bytes past the end of the source map (CBOR metadata / data), not code

We report, for greedy and safe contracts, the share of X-VASCOT's top-5 % in each
category against the category's share of the contract (enrichment) and against
random selection. Disassembled regime (every model position is an instruction).

    python -m experiments.source_localisation disasm 42 60
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.request
from collections import Counter, defaultdict

import numpy as np
import torch

from experiments.train import DEVICE, build_vascot, prepare
from xvascot.config import CKPTS, CONFIG, RESULTS
from xvascot.data import load_dataset_i
from xvascot.xai import XVASCOT

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "sourcify"
WORK = ROOT / "cache" / "slither_work"
from solc_select.constants import ARTIFACTS_DIR as SOLC  # solc-select keeps binaries inside the venv
STD = ("SafeMath", "Context", "Ownable", "IERC20", "ERC20", "IERC20Metadata", "Address",
       "IUniswapV2Factory", "IUniswapV2Router01", "IUniswapV2Router02", "IUniswapV2Pair",
       "SafeERC20", "Strings", "ReentrancyGuard", "ERC721", "IERC721")
CATS = ("eth_send", "payable", "other_custom", "std_library", "compiler", "metadata")


def sourcify(addr):
    f = CACHE / f"{addr.lower()}.json"
    if not f.exists():
        with urllib.request.urlopen(f"https://sourcify.dev/server/v2/contract/1/{addr}?fields=all", timeout=60) as r:
            f.write_text(r.read().decode())
    return json.loads(f.read_text())


def disasm_positions(hexbytes):
    """(opcode byte string) for each instruction, PUSH-aware."""
    out, i = [], 0
    while i < len(hexbytes):
        b = int(hexbytes[i], 16)
        out.append(hexbytes[i])
        i += (1 + (b - 0x5f)) if 0x60 <= b <= 0x7f else 1
    return out


def decode_sourcemap(sm):
    ents, prev = [], ["-1", "-1", "-1", "-", "0"]
    for e in sm.split(";"):
        parts = e.split(":")
        cur = [parts[k] if k < len(parts) and parts[k] != "" else prev[k] for k in range(5)]
        ents.append((int(cur[0]), int(cur[1]), int(cur[2])))
        prev = cur
    return ents


def function_ranges(d, addr):
    """Slither function spans per source path, with Ether semantics."""
    from slither import Slither
    comp = d["compilation"]
    ver = comp["compilerVersion"].split("+")[0].lstrip("v")
    os.system(f"solc-select install {ver} > /dev/null 2>&1")
    wd = WORK / addr.lower()
    wd.mkdir(parents=True, exist_ok=True)
    for p, v in d["sources"].items():
        fp = wd / p
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(v["content"])
    main = comp.get("fullyQualifiedName", "").split(":")[0]
    target = wd / main if main and (wd / main).exists() else next(wd.rglob("*.sol"))
    cwd = os.getcwd()
    os.chdir(wd)
    try:
        sl = Slither(str(target.relative_to(wd)), solc=str(SOLC / f"solc-{ver}" / f"solc-{ver}"),
                     solc_disable_warnings=True)
    finally:
        os.chdir(cwd)
    spans = defaultdict(list)            # path -> [(start, end, category)]
    for c in sl.contracts:
        std = c.is_library or c.name in STD or c.is_interface
        for f in c.functions_and_modifiers_declared:
            sm = f.source_mapping
            if sm is None or sm.start is None:
                continue
            path = sm.filename.relative if hasattr(sm.filename, "relative") else str(sm.filename)
            path = str(pathlib.Path(path).as_posix()).lstrip("./")
            if std:
                cat = "std_library"
            else:
                try:
                    send = f.can_send_eth()
                except Exception:
                    send = False
                cat = "eth_send" if send else ("payable" if getattr(f, "payable", False) else "other_custom")
            spans[path].append((sm.start, sm.start + sm.length, cat))
    return spans


def classify(ents, ids_to_path, spans):
    out = []
    for s, l, f in ents:
        if f == -2:
            out.append("metadata")
            continue
        if f < 0 or f not in ids_to_path:
            out.append("compiler")
            continue
        path = ids_to_path[f]
        best = None
        for a, b, cat in spans.get(path, []):
            if a <= s < b and (best is None or b - a < best[1] - best[0]):
                best = (a, b, cat)
        out.append(best[2] if best else "compiler")
    return out


def main():
    mode, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    df = load_dataset_i().reset_index(drop=True)
    ctx = prepare(mode, seed=seed, log=lambda *a: None)
    ck = CKPTS / f"{mode}_seedsweep_vascot_{seed}.pt"
    m = build_vascot(mode)
    m.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True))
    m.eval()
    xv = XVASCOT(m, ctx["tok"], CONFIG, DEVICE, "l2")
    # recover dataset row index of each test contract (prepare keeps df order)
    y_all = df["LABEL"].to_numpy()
    te_rows = ctx["te_rows"] if "te_rows" in ctx else None
    assert te_rows is not None, "prepare must expose te_rows"
    rng = np.random.default_rng(0)
    word = ctx["tok"].word_index

    agg = {c: {"xv": Counter(), "rand": Counter(), "base": Counter(), "n": 0, "fail": Counter()} for c in ("greedy", "safe")}
    per = []
    yte = np.asarray(ctx["yte"])
    for cls, lab in (("greedy", 1), ("safe", 0)):
        for j in np.where(yte == lab)[0][:n]:
            row = int(te_rows[j])
            addr = df.Address[row]
            try:
                d = sourcify(addr)
                ents = decode_sourcemap(d["runtimeBytecode"]["sourceMap"])
                ids_to_path = {v["id"]: p for p, v in d["sourceIds"].items()}
                spans = function_ranges(d, addr)
            except Exception as e:
                agg[cls]["fail"][type(e).__name__] += 1
                continue
            instr = disasm_positions(str(df.OPCODE[row]).split())
            # model positions are the instructions whose token survives the vocab cut
            keep = [i for i, t in enumerate(instr) if word.get(t, 10**9) < CONFIG["num_words"]]
            seq = np.asarray(ctx["Xte"][j], dtype=np.int64)
            if len(keep) != int((seq > 0).sum()):
                agg[cls]["fail"][f"alignment({len(keep)} vs {int((seq > 0).sum())})"] += 1
                continue
            ents = ents + [(-1, -1, -2)] * max(0, len(instr) - len(ents))
            cats = np.array(classify([ents[i] for i in keep], ids_to_path, spans))
            sc = xv.importance(torch.from_numpy(seq).unsqueeze(0), True)[seq > 0]
            k = max(1, int(np.ceil(0.05 * len(sc))))
            top = np.argsort(sc)[::-1][:k]
            rnd = rng.choice(len(sc), size=k, replace=False)
            agg[cls]["xv"].update(cats[top].tolist())
            agg[cls]["rand"].update(cats[rnd].tolist())
            agg[cls]["base"].update(cats.tolist())
            agg[cls]["n"] += 1
            per.append({"row": row, "cls": cls, "address": addr,
                        "top_share": {c: float((cats[top] == c).mean()) for c in CATS},
                        "base_share": {c: float((cats == c).mean()) for c in CATS}})
            print(f"{cls} {addr[:10]} n_instr={len(keep)} top: " +
                  " ".join(f"{c}={100*(cats[top]==c).mean():.0f}%/{100*(cats==c).mean():.0f}%" for c in CATS), flush=True)

    res = {"mode": mode, "seed": seed, "per_contract": per, "summary": {}}
    for cls, a in agg.items():
        tb, tx, tr = sum(a["base"].values()), sum(a["xv"].values()), sum(a["rand"].values())
        res["summary"][cls] = {
            "n": a["n"], "failures": dict(a["fail"]),
            "share_top_xvascot": {c: a["xv"][c] / max(tx, 1) for c in CATS},
            "share_top_random": {c: a["rand"][c] / max(tr, 1) for c in CATS},
            "share_base": {c: a["base"][c] / max(tb, 1) for c in CATS},
            "enrichment_xvascot": {c: (a["xv"][c] / max(tx, 1)) / max(a["base"][c] / max(tb, 1), 1e-9) for c in CATS},
        }
        s = res["summary"][cls]
        print(f"\n== {cls}: {a['n']} contracts analysed, failures {dict(a['fail'])}")
        for c in CATS:
            print(f"   {c:<13} X-VASCOT {100*s['share_top_xvascot'][c]:5.1f}%   random {100*s['share_top_random'][c]:5.1f}%   "
                  f"base {100*s['share_base'][c]:5.1f}%   enrichment {s['enrichment_xvascot'][c]:.2f}x")
    out = RESULTS / f"source_localisation_{mode}_{seed}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
