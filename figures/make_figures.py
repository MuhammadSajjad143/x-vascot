"""Figures 2 and 3 of the paper, drawn from the JSON files in ../results.

Figure 1 (the protocol diagram) is fig_framework.tex. Compile it with pdflatex
and convert the PDF to PNG (e.g. pdftoppm -r 600 -png).

    python figures/make_figures.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "results"
sys.path.insert(0, str(HERE.parent))

S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
W = 4.8

plt.rcParams.update({
    "font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": GRID,
    "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.45, "legend.frameon": False, "legend.fontsize": 7,
    "hatch.linewidth": 0.5, "savefig.dpi": 400, "pdf.fonttype": 42,
})


def load(name):
    return json.loads((RES / name).read_text())


def fig_heatmap():
    """Signature specificity as a heatmap: selector x explained class, per instruction and split.
    Each cell is the share of contracts CONTAINING the instruction whose top-5 % holds it."""
    from matplotlib.colors import LinearSegmentedColormap
    d = load("ig_specificity.json")
    meths = ["xvascot", "int_grad", "grad_x_input", "xvascot_v1", "raw_attn", "saliency", "rare_corpus", "random"]
    names = ["X-VASCOT", "Integrated Gradients", "Gradient \u00d7 input", "Attention \u00d7 \u2016grad\u2016",
             "Raw attention", "Saliency \u2016grad\u2016", "Rarest in corpus", "Random"]
    cls = ["Saf", "Sui", "Pro", "Gre"]
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9cc2ee", S1, "#123f75"])
    panels = [("ff", "ii4_full_dedup", "SELFDESTRUCT\ndeduplicated"), ("ff", "ii4_full_group", "SELFDESTRUCT\nunseen families"),
              ("f1", "ii4_full_dedup", "CALL\ndeduplicated"), ("f1", "ii4_full_group", "CALL\nunseen families")]
    own = {"ff": 1, "f1": 2}
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.65), sharey=True)
    for k, (ax, (op, mode, title)) in enumerate(zip(axes, panels)):
        M = np.array([[100 * d[mode][f"{op}|{c}|{m}"]["hit"] for c in range(4)] for m in meths])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ns_ = [d[mode][f"{op}|{c}|xvascot"]["n"] for c in range(4)]
        for i in range(M.shape[0]):
            for j in range(4):
                small = ns_[j] < 20
                if small:   # fewer than 20 contracts: grey the cell so it is not over-read
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#e6e5e1", edgecolor="none", zorder=2))
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.2, zorder=3,
                        color=MUTED if small else ("white" if M[i, j] > 55 else INK))
        ax.add_patch(Rectangle((own[op] - 0.5, -0.5), 1, len(meths), fill=False, edgecolor=S2, lw=1.1))
        ax.axhline(2.5, color="white", lw=1.6)
        ax.axhline(5.5, color="white", lw=1.6)
        ns = [d[mode][f"{op}|{c}|xvascot"]["n"] for c in range(4)]
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{c}\n{n}" for c, n in zip(cls, ns)], fontsize=5.2)
        ax.set_title(f"({'abcd'[k]}) {title}", fontsize=6.0, pad=3)
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(meths)))
    axes[0].set_yticklabels(names, fontsize=5.8)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(HERE / "fig_specificity.pdf", bbox_inches="tight")
    fig.savefig(HERE / "fig_specificity.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_generalisation():
    """Accuracy by split protocol, with the majority-class rate of each split as a tick."""
    g = load("group_cv_eval.json")
    s7 = {m: load(f"{m}_stage7_seeds.json")["summary"]["acc"]["vascot_mean"] for m in ("disasm",)}
    from xvascot.data import load_dataset_i, load_dataset_ii_xls
    di, dr, d4 = load_dataset_i(), load_dataset_ii_xls(True), load_dataset_ii_xls(False)

    def majority(df, dedup=False):
        if dedup:
            df = df.drop_duplicates(subset="OPCODE")
        y = df["LABEL"].to_numpy()
        return 100 * np.bincount(y).max() / len(y)

    groups = ["Dataset I\n(disassembled)", "Dataset II-r\n(opcode-only)", "Dataset II, 4-class\n(opcode-only)"]
    acc = {groups[0]: [100 * s7["disasm"], np.nan, 100 * g["disasm_group"]["pooled"]["acc"]],
           groups[1]: [100 * g["iir_full"]["mean"]["acc"], 100 * g["iir_full_dedup"]["mean"]["acc"],
                       100 * g["iir_full_group"]["pooled"]["acc"]],
           groups[2]: [100 * g["ii4_full"]["mean"]["acc"], 100 * g["ii4_full_dedup"]["mean"]["acc"],
                       100 * g["ii4_full_group"]["pooled"]["acc"]]}
    maj = {groups[0]: [majority(di), np.nan, majority(di)],
           groups[1]: [majority(dr), majority(dr, True), majority(dr)],
           groups[2]: [majority(d4), majority(d4, True), majority(d4)]}
    series = ["random split (base protocol)", "deduplicated", "family-grouped (unseen families)"]
    cols, hz = [S1, S2, S3], ["", "////", "\\\\\\\\"]

    fig, ax = plt.subplots(figsize=(W, 1.08))
    x = np.arange(len(groups))
    w = 0.8 / 3
    for i, (s, c, h) in enumerate(zip(series, cols, hz)):
        off = (i - 1) * w
        for xi, gname in zip(x, groups):
            v, m = acc[gname][i], maj[gname][i]
            if np.isnan(v):
                ax.annotate("n/a", (xi + off, 2), ha="center", va="bottom", fontsize=5.6, color=MUTED)
                continue
            ax.bar(xi + off, v, w * 0.88, color=c, hatch=h, edgecolor="white", linewidth=0.6, zorder=3,
                   label=s if gname == groups[1] else None)
            ly = m + 1.0 if v < m < v + 9 else v
            ax.annotate(f"{v:.1f}", (xi + off, ly), textcoords="offset points", xytext=(0, 1.5),
                        ha="center", va="bottom", fontsize=5.6, color=INK)
            ax.plot([xi + off - w * 0.44, xi + off + w * 0.44], [m, m], color=INK, lw=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 112)
    ax.set_ylabel("VASCOT test accuracy (%)")
    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", label=s) for s, c, h in zip(series, cols, hz)]
    handles.append(plt.Line2D([0], [0], color=INK, lw=1.0, label="majority-class rate of that split"))
    ax.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.36), fontsize=6.2,
              handlelength=1.4, columnspacing=1.0)
    fig.savefig(HERE / "fig_generalisation.pdf", bbox_inches="tight")
    fig.savefig(HERE / "fig_generalisation.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_heatmap()
    fig_generalisation()
    print("figures written to", HERE)
