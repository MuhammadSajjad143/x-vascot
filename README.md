# X-VASCOT

Code and results for the paper *Signature-Grounded Validation of Explanations for
Smart Contract Vulnerability Detection* (Muhammad Sajjad, Hammad Ali).

VASCOT is a small Transformer that classifies Ethereum bytecode as safe or
vulnerable. X-VASCOT explains its verdicts: for a contract and a class, it scores
every instruction with Integrated Gradients of the class margin, computed per
window from the "deleted" (all-padding) state, and returns the top 5 % of
instructions. The paper asks whether such explanations point at the
vulnerability or just at whatever the model uses, and tests this with four
checks (faithfulness, class specificity, reliance, source grounding) on the
three VASCOT datasets under random, deduplicated and unseen-family splits.

## Layout

```
xvascot/       model, data loading, training loop and the explainer (xai.py)
experiments/   one script per experiment in the paper
scripts/       run_all.sh, the whole pipeline in order
results/       the JSON outputs behind every number in the paper
figures/       figure script and the paper figures
data/          the two VASCOT dataset files
```

## Setup

Python 3.12 and a CUDA GPU are recommended (we used an RTX 3090).

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The two dataset files are already in `data/`.

## Running

Everything is run from the repository root as a module. A few examples:

```
# train VASCOT on Dataset I (five seeds) and on Dataset II with unseen families
python -m experiments.train --mode disasm --stages 7
python -m experiments.train --mode ii4_full_group --stages 7 --no-lstm

# faithfulness against the ten baselines, then class specificity
python -m experiments.t1_baselines disasm ii4_full_dedup ii4_full_group
python -m experiments.ig_specificity ii4_full_dedup ii4_full_group
```

`scripts/run_all.sh` runs the full pipeline in the order used for the paper.
Training takes roughly 3-4 hours on one GPU and the analyses about another hour.
Set `PYTHONHASHSEED=0` before building the Dataset I families, otherwise the
family assignment (and so the unseen-family splits) will differ.

Mode names follow `<dataset>_<input>[_split]`: `disasm` is Dataset I after
disassembly, `iir_full` and `ii4_full` are Datasets II-r and II, and the suffixes
`_dedup` and `_group` select the deduplicated and unseen-family splits.

## Which script gives which result

| Paper | Script | Output |
|---|---|---|
| Table 1 (faithfulness) | `t1_baselines.py`, `ci_analysis.py` | `t1_baselines.json`, `ci_analysis.json` |
| Fig. 2 (specificity) | `ig_specificity.py`, `ci_analysis.py` | `ig_specificity.json` |
| RQ3 (reliance, counterfactual) | `semantic_tests.py` | `semantic_tests.json` |
| RQ3 (source grounding) | `source_localisation.py` | `source_localisation_*.json` |
| Choice of scoring rule | `select_explainer.py` | `select_explainer.json` |
| Completeness check | `completeness.py` | `completeness.json` |
| Table 2 (audit) | `audit_dataset_i.py`, `audit_dataset_ii.py`, `families.py` | `audit_*.json`, `families*.json` |
| Fig. 3 (accuracy) | `train.py`, `group_cv_eval.py` | `*_stage7_seeds.json`, `group_cv_eval.json` |
| Cross-dataset | `cross_dataset.py` | `cross_dataset*.json` |

Trained checkpoints are not included (about 135 MB); `train.py` recreates them.
Source grounding downloads verified source from Sourcify and needs Slither and
`solc-select`; everything else runs offline.

## Notes

- Datasets II and II-r already store instructions without PUSH operands, so only
  Dataset I is disassembled.
- VASCOT has no padding mask, so contracts are evaluated one at a time.
- The results in `results/` come from a single machine; GPU non-determinism can
  shift retrained numbers slightly.

## Citation

If you use this code or the results, please cite the paper: Muhammad Sajjad and
Hammad Ali, *Signature-Grounded Validation of Explanations for Smart Contract
Vulnerability Detection*.
