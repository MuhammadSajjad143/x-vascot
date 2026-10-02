"""The X-VASCOT explanation module and its evaluation metrics.

Implements the paper's pipeline:

    Eq. (1)  a      = mean over heads of the attention received by each position
    Eq. (2)  a~_i   = a_i * || dz/de_i ||_2
    Eq. (3)  k      = ceil(0.05 * L)

Three things the notebook did differently are parameterised rather than
hard-coded, so repro and fixed mode can both be run:

``grad_reduce='mean_abs'``  reproduces the notebook's ``.abs().mean(-1)``.
``grad_reduce='l2'``        is the l2 norm the paper's Eq. (2) actually states.

Windowed contracts combine per-window importance by element-wise maximum, as
Sec. IV-D describes, so a pattern straddling a window boundary is not lost. The
notebook averaged attention across windows instead, but never exercised the path
because its inputs were always exactly one window long.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

# ── EVM opcode table (Ethereum Yellow Paper) ─────────────────────────────────
ETH_OPCODE_MAP = {
    "00": "STOP", "01": "ADD", "02": "MUL", "03": "SUB", "04": "DIV", "05": "SDIV",
    "06": "MOD", "07": "SMOD", "08": "ADDMOD", "09": "MULMOD", "0a": "EXP",
    "0b": "SIGNEXTEND", "10": "LT", "11": "GT", "12": "SLT", "13": "SGT",
    "14": "EQ", "15": "ISZERO", "16": "AND", "17": "OR", "18": "XOR", "19": "NOT",
    "1a": "BYTE", "1b": "SHL", "1c": "SHR", "1d": "SAR", "20": "SHA3",
    "30": "ADDRESS", "31": "BALANCE", "32": "ORIGIN", "33": "CALLER",
    "34": "CALLVALUE", "35": "CALLDATALOAD", "36": "CALLDATASIZE",
    "37": "CALLDATACOPY", "38": "CODESIZE", "39": "CODECOPY", "3a": "GASPRICE",
    "3b": "EXTCODESIZE", "3c": "EXTCODECOPY", "3d": "RETURNDATASIZE",
    "3e": "RETURNDATACOPY", "3f": "EXTCODEHASH", "40": "BLOCKHASH",
    "41": "COINBASE", "42": "TIMESTAMP", "43": "NUMBER", "44": "DIFFICULTY",
    "45": "GASLIMIT", "46": "CHAINID", "47": "SELFBALANCE", "50": "POP",
    "51": "MLOAD", "52": "MSTORE", "53": "MSTORE8", "54": "SLOAD", "55": "SSTORE",
    "56": "JUMP", "57": "JUMPI", "58": "PC", "59": "MSIZE", "5a": "GAS",
    "5b": "JUMPDEST", "a0": "LOG0", "a1": "LOG1", "a2": "LOG2", "a3": "LOG3",
    "a4": "LOG4", "f0": "CREATE", "f1": "CALL", "f2": "CALLCODE", "f3": "RETURN",
    "f4": "DELEGATECALL", "f5": "CREATE2", "fa": "STATICCALL", "fd": "REVERT",
    "fe": "INVALID", "ff": "SELFDESTRUCT",
}
for _i in range(0x60, 0x80):
    ETH_OPCODE_MAP[f"{_i:02x}"] = f"PUSH{_i - 0x5f}"
for _i in range(0x80, 0x90):
    ETH_OPCODE_MAP[f"{_i:02x}"] = f"DUP{_i - 0x7f}"
for _i in range(0x90, 0xa0):
    ETH_OPCODE_MAP[f"{_i:02x}"] = f"SWAP{_i - 0x8f}"


def hex_to_mnemonic(tok) -> str:
    return ETH_OPCODE_MAP.get(str(tok).lower(), f"UNK({tok})")


def disassemble_mask(tokens) -> np.ndarray:
    """Mark which byte positions are real instructions rather than PUSH data.

    EVM PUSH1..PUSH32 (0x60-0x7f) carry 1..32 immediate bytes that are operands,
    not instructions. The dataset stores raw bytes, so a naive per-byte reading
    mislabels every one of those operands as an opcode. Returns a boolean array,
    True where the byte is genuinely an instruction.
    """
    mask = np.zeros(len(tokens), dtype=bool)
    i = 0
    while i < len(tokens):
        try:
            b = int(tokens[i], 16)
        except (ValueError, TypeError):
            i += 1
            continue
        mask[i] = True
        i += (1 + (b - 0x5f)) if 0x60 <= b <= 0x7f else 1
    return mask


# ── Metrics ──────────────────────────────────────────────────────────────────
def attention_entropy(scores: np.ndarray) -> float:
    """Shannon entropy (nats) of the normalised importance vector; lower = focused."""
    s = np.asarray(scores, dtype=np.float64)
    s = s / (s.sum() + 1e-9)
    s = s[s > 0]
    return float(-np.sum(s * np.log(s + 1e-9)))


def topk_coverage(scores: np.ndarray, k: int) -> float:
    """Share of total importance mass held by the top-k positions."""
    s = np.asarray(scores, dtype=np.float64)
    return float(np.sort(s)[::-1][:k].sum() / (s.sum() + 1e-9))


def adaptive_k(L: int, ratio: float = 0.05) -> int:
    return max(1, math.ceil(ratio * L))


# ── Signed attributions (X-VASCOT v2) ────────────────────────────────────────
SCORES = ("attn_grad", "ig", "cig", "gxi", "cgxi")
IG_STEPS = 16


def target_logit(logits: torch.Tensor, cls: int, contrastive: bool) -> torch.Tensor:
    """z_c, or the contrastive margin z_c - mean_{c' != c} z_c' ("why c rather than the others")."""
    if not contrastive:
        return logits[:, cls]
    others = [j for j in range(logits.shape[1]) if j != cls]
    return logits[:, cls] - logits[:, others].mean(1)


def window_signed_attribution(model, emb0: torch.Tensor, cls: int, contrastive: bool,
                              steps: int | None = IG_STEPS) -> torch.Tensor:
    """Signed per-position attribution for one window, (W,).

    ``steps=None``: gradient x input at the embedding. Otherwise Integrated Gradients
    from the all-zero embedding with a midpoint Riemann sum. The zero embedding is
    exactly what a padded (deleted) position contributes (padding_idx=0, positional
    encoding added afterwards), so the baseline equals the deletion state and the
    attributions sum to approximately f(x) - f(everything deleted).
    """
    if steps is None:
        e = emb0.clone().requires_grad_(True)
        f = target_logit(model.classify(model.encode_embedded(e)), cls, contrastive).sum()
        g, = torch.autograd.grad(f, e)
        return (emb0[0] * g[0]).sum(-1)
    alphas = (torch.arange(steps, device=emb0.device, dtype=emb0.dtype) + 0.5) / steps
    e = (emb0 * alphas.view(-1, 1, 1)).requires_grad_(True)            # (steps, W, d)
    f = target_logit(model.classify(model.encode_embedded(e)), cls, contrastive).sum()
    g, = torch.autograd.grad(f, e)
    return (emb0[0] * g.mean(0)).sum(-1)


# ── The explanation module ───────────────────────────────────────────────────
class XVASCOT:
    """Post-hoc explainer wrapping a frozen VASCOT checkpoint.

    The underlying weights are never modified; the only added computation is one
    backward pass per window.
    """

    def __init__(self, model, tokenizer, config, device,
                 grad_reduce: str = "l2", score: str = "cig"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.grad_reduce = grad_reduce
        self.idx2token = tokenizer.index_word
        # class whose logit is explained; 1 (vulnerable) for every binary model,
        # set per contract for the 4-class Dataset II (suicidal 1 / prodigal 2 / greedy 3)
        self.target = 1
        # Default "cig" = X-VASCOT v2, chosen on validation splits (select_explainer.py).
        # attn_grad: v1, received attention x ||dz_c/de||_2 (paper Eq. 2, unsigned)
        # ig / cig: Integrated Gradients on z_c / on the contrastive margin (signed)
        # gxi / cgxi: gradient x input on z_c / on the contrastive margin (signed)
        if score not in SCORES:
            raise ValueError(f"unknown score {score!r}; expected one of {SCORES}")
        self.score = score

    # ── gradient reduction over the embedding dimension ──────────────────────
    def _reduce(self, grad: torch.Tensor) -> torch.Tensor:
        if self.grad_reduce == "l2":
            return grad.norm(dim=-1)               # paper Eq. (2)
        if self.grad_reduce == "mean_abs":
            return grad.abs().mean(-1)             # notebook's coding of it
        raise ValueError(f"unknown grad_reduce {self.grad_reduce!r}")

    @staticmethod
    def _attn_vector(attn: torch.Tensor) -> torch.Tensor:
        """(B, H, L, L) -> (L,): mean over heads, then over queries.

        Entry (i, j) is the weight placed on key j while building query i, so
        averaging over the query axis gives the attention each position received.
        """
        return attn.squeeze(0).mean(0).mean(0)

    def importance(self, x: torch.Tensor, use_grad: bool = True) -> np.ndarray:
        """Per-position importance over the full (possibly windowed) sequence.

        Returns an array of length L. Window results are merged by element-wise
        maximum over the overlapping region.
        """
        model, W = self.model, self.config["max_length"]
        model.eval()
        x = x.to(self.device)
        L = x.shape[1]
        # signed scores can be negative, so windows start from -inf before the max-merge
        scores = np.full(L, -np.inf if (use_grad and self.score != "attn_grad") else 0.0)
        # Algorithm 1: ã is computed per window, each window's vulnerability logit
        # z differentiated w.r.t. that window's embeddings, then windows merged by
        # element-wise max. (Differentiating the aggregated mean-probability
        # output instead was tried and reverted: it weights windows by
        # p(1-p), favours the least-certain windows, and roughly halved the
        # deletion effect - see FIXES.md F6.)
        for s in model.window_starts(L):
            ids = x[:, s:s + W]
            if use_grad and self.score != "attn_grad":
                emb0 = model.embedding(ids).detach()
                win = window_signed_attribution(
                    model, emb0, self.target, self.score.startswith("c"),
                    None if self.score.endswith("gxi") else IG_STEPS).detach().float().cpu().numpy()
            elif use_grad:
                emb = model.embedding(ids).detach().requires_grad_(True)
                pooled, attn = model.encode_embedded(emb, return_attn=True)
                z = model.classify(pooled)[0, self.target]
                model.zero_grad(set_to_none=True)
                z.backward()
                a = self._attn_vector(attn).detach()
                g = self._reduce(emb.grad.detach()).squeeze(0)
                win = (a * g).float().cpu().numpy()
            else:
                with torch.no_grad():
                    _, attn = model.encode_embedded(model.embedding(ids),
                                                    return_attn=True)
                win = self._attn_vector(attn).float().cpu().numpy()
            seg = scores[s:s + len(win)]
            scores[s:s + len(win)] = np.maximum(seg, win[:len(seg)])

        return scores

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[int, float]:
        self.model.eval()
        logit = self.model(x.to(self.device))
        pred = int(logit.argmax(1).item())
        conf = float(F.softmax(logit.float(), dim=1)[0, pred].item())
        return pred, conf

    def explain(self, x: torch.Tensor, ratio: float = 0.05,
                use_grad: bool = True, fixed_k: int | None = None) -> dict:
        """Full explanation for one contract: prediction, ranked opcodes, metrics."""
        pred, conf = self.predict(x)
        scores = self.importance(x, use_grad=use_grad)

        ids = x.squeeze(0).cpu().numpy()
        nonpad = ids > 0
        s_nonpad = scores[nonpad]
        toks = [self.idx2token.get(int(t), "PAD") for t in ids[nonpad]]
        mnems = [hex_to_mnemonic(t) for t in toks]

        L = int(nonpad.sum())
        k = fixed_k if fixed_k else adaptive_k(L, ratio)
        k = min(k, L)
        order = np.argsort(s_nonpad)[::-1][:k]

        return {
            "pred": pred, "conf": conf, "L": L, "k": k,
            "scores": s_nonpad, "tokens": toks, "mnemonics": mnems,
            "top_idx": order,
            "top": [(mnems[i], float(s_nonpad[i])) for i in order],
            "entropy": attention_entropy(s_nonpad),
            "coverage": topk_coverage(s_nonpad, k),
        }


# ── Faithfulness deletion test ───────────────────────────────────────────────
def deletion_test(xv: XVASCOT, X, y, n_samples: int = 50, use_grad: bool = True,
                  fixed_k: int | None = None, ratio: float = 0.05, log=print):
    """Zero the selected opcodes, re-run inference, measure the confidence drop.

    A faithful explanation should cost the model confidence when removed; a high
    flip rate means the removed opcodes were carrying the prediction.

    Returns ``(mean_drop, drops, flip_rate, extra)``. ``flip_rate`` is computed
    over contracts originally predicted vulnerable; ``extra`` holds that count
    and the old definition (any post-deletion non-vulnerable prediction).
    """
    model, device = xv.model, xv.device
    model.eval()
    vuln = np.where(np.asarray(y) == 1)[0][:n_samples]
    drops, flips, flips_any = [], [], []

    for n, idx in enumerate(vuln):
        seq = np.asarray(X[idx], dtype=np.int64)
        x = torch.from_numpy(seq).unsqueeze(0)

        with torch.no_grad():
            logit_orig = model(x.to(device))
            conf_orig = F.softmax(logit_orig.float(), 1)[0, 1].item()
            pred_orig = int(logit_orig.argmax(1).item())

        nonpad = seq > 0
        pos = np.where(nonpad)[0]
        scores = xv.importance(x, use_grad=use_grad)[nonpad]
        k = fixed_k if fixed_k else adaptive_k(int(nonpad.sum()), ratio)
        k = min(k, len(scores))
        kill = pos[np.argsort(scores)[::-1][:k]]

        deleted = seq.copy()
        deleted[kill] = 0
        with torch.no_grad():
            logit_del = model(torch.from_numpy(deleted).unsqueeze(0).to(device))
            conf_del = F.softmax(logit_del.float(), 1)[0, 1].item()

        drops.append(conf_orig - conf_del)
        pred_del = int(logit_del.argmax(1).item())
        flips_any.append(1 if pred_del == 0 else 0)
        # A flip is a prediction that *changes* from vulnerable to non-vulnerable;
        # contracts already predicted non-vulnerable cannot flip.
        if pred_orig == 1:
            flips.append(1 if pred_del == 0 else 0)
        if (n + 1) % 10 == 0:
            log(f"    deletion {n+1}/{len(vuln)}  mean drop so far {np.mean(drops):+.4f}")

    flip_rate = float(np.mean(flips)) if flips else float("nan")
    extra = {"n_pred_vuln": len(flips),
             "flip_rate_any_nonvuln": float(np.mean(flips_any))}
    return float(np.mean(drops)), drops, flip_rate, extra
