"""Central configuration for the X-VASCOT re-run.

Three input regimes are supported (the paper's Truncated / Full / Disassembled):

``repro``  Sequences are cut to the trailing 2048 tokens with Keras'
           ``truncating='pre'`` semantics, reproducing the constrained original
           setting. The explainer is the paper's Eq. (2) (l2 norm), as in every
           other regime, and the ablation runs on the 50 contracts the paper
           reports.

``fixed``  Full sequences are kept and the sliding window actually runs.

``disasm`` As ``fixed``, with PUSH operands removed before tokenisation.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]        # repository root
DATA_DIR  = ROOT / "data"
RESULTS   = ROOT / "results"
FIGURES   = ROOT / "figures"
CKPTS     = ROOT / "checkpoints"
LOGS      = ROOT / "logs"

DATASET_I   = DATA_DIR / "Opcode 1915.xlsx"      # verified 2022, Dataset I
# Legacy file from the original X-VASCOT code (greedy-only); not used in the paper.
DATASET_II  = DATA_DIR / "Opcode 13801.xlsx"

for _d in (RESULTS, FIGURES, CKPTS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# ── Model / training hyperparameters (base paper Table 3) ────────────────────
CONFIG = {
    "num_words":            150,      # vocabulary cap - distinct EVM opcodes
    "max_length":           2048,     # transformer context window
    "embedding_dim":        128,
    "batch_size":           4,
    "epochs":               50,
    "learning_rate":        5e-4,
    "num_heads":            4,
    "ff_dim":               1024,     # 4 * 256
    "dropout":              0.2,
    "hidden_dim":           256,      # LSTM only
    "output_dim":           2,
    "sliding_window_ratio": 0.75,     # step = 0.75 * W  ->  25% overlap
    "patience":             7,
}

SEED  = 42
SEEDS = [42, 123, 456, 789, 2024]

# ── Experiment sizes ─────────────────────────────────────────────────────────
TOPK_RATIO = 0.05          # k = ceil(0.05 * L)

MODES = {
    "repro": {
        "truncate_to":  CONFIG["max_length"],   # hard cut, no windowing
        "grad_reduce":  "l2",                   # paper Eq. (2), all regimes
        "n_ablation":   50,                     # paper Table II reports n=50
        "n_faithful":   50,
        "n_shap":       20,
        "disassemble":  False,
    },
    "fixed": {
        "truncate_to":  None,                   # keep full sequence, window it
        "grad_reduce":  "l2",                   # paper Eq. (2)
        "n_ablation":   50,                     # paper reports n=50
        "n_faithful":   50,
        "n_shap":       20,
        "disassemble":  False,
    },
    # As ``fixed``, but the byte stream is disassembled first so PUSH operands
    # are not mistaken for instructions. Tests whether the SELFDESTRUCT finding
    # survives once the phantom ``ff`` bytes are removed.
    "disasm": {
        "truncate_to":  None,
        "grad_reduce":  "l2",
        "n_ablation":   50,
        "n_faithful":   50,
        "n_shap":       20,
        "disassemble":  True,
    },
}
# Family-aware variants: same regimes, but no near-duplicate family
# (families.py, 4-gram Jaccard >= 0.8) is shared between train, val and test.
MODES["disasm_group"] = {**MODES["disasm"], "split": "group"}
MODES["fixed_group"] = {**MODES["fixed"], "split": "group"}

# Datasets II (4-class) and II-r (binary) from the base paper, Opcode 16469.xls.
# Names: {iir|ii4}_{full|disasm}[_dedup|_group]
#   full   raw bytes, sliding window (base-paper protocol); disasm: PUSH operands removed
#   (none) stratified random split over rows, duplicates included (as in the base paper)
#   _dedup one row per distinct bytecode, then the random split
#   _group 5-fold family-grouped CV (families_<dataset>.npy)
DATASET_II_XLS = DATA_DIR / "Opcode 16469.xls"
for _ds, _ncls in (("iir", 2), ("ii4", 4)):
    # Datasets II/II-r ship opcode-only sequences (PUSH operands already stripped),
    # so their "full" input is the instruction stream and there is no disasm mode.
    for _inp, _base in (("full", "fixed"),):
        for _sp in ("", "_dedup", "_group"):
            MODES[f"{_ds}_{_inp}{_sp}"] = {
                **MODES[_base], "dataset": _ds, "n_classes": _ncls,
                "dedup": _sp == "_dedup",
                **({"split": "group"} if _sp == "_group" else {}),
            }


def mode_cfg(mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    return MODES[mode]
