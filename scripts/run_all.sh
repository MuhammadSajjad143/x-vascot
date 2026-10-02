#!/usr/bin/env bash
# Runs every experiment in the paper, in order. Run from the repository root.
# Training takes about 3-4 hours on one RTX 3090; the analysis steps about 1 hour.
set -euo pipefail
export PYTHONHASHSEED=0   # Dataset I families use Python's hash()

# 1. near-duplicate families (0.8 is used for the splits; 0.7 and 0.9 only for the audit)
for ds in i iir ii4; do
  python -m experiments.families $ds
  python -m experiments.families $ds 0.7
  python -m experiments.families $ds 0.9
done

# 2. dataset audit (Table 2)
python -m experiments.audit_dataset_i
python -m experiments.audit_dataset_ii iir ii4

# 3. train VASCOT under every split protocol
python -m experiments.train --mode disasm --stages 7
python -m experiments.train --mode disasm_group --stages 7
python -m experiments.train --mode fixed_group --stages 7
for ds in iir ii4; do
  python -m experiments.train --mode ${ds}_full --stages 7 --seeds 42,123,456 --no-lstm
  python -m experiments.train --mode ${ds}_full_dedup --stages 7 --no-lstm
  python -m experiments.train --mode ${ds}_full_group --stages 7 --no-lstm
done

# 4. accuracy, macro-F1 and pooled unseen-family results (Fig. 3)
python -m experiments.group_cv_eval disasm disasm_group fixed_group \
  iir_full iir_full_dedup iir_full_group ii4_full ii4_full_dedup ii4_full_group

# 5. choose the X-VASCOT scoring rule on validation data, then the four tests
python -m experiments.select_explainer
python -m experiments.t1_baselines disasm iir_full iir_full_dedup iir_full_group ii4_full ii4_full_dedup ii4_full_group
python -m experiments.semantic_tests ii4_full ii4_full_dedup ii4_full_group iir_full iir_full_dedup iir_full_group
python -m experiments.ig_specificity ii4_full ii4_full_dedup ii4_full_group
python -m experiments.ci_analysis
python -m experiments.completeness disasm iir_full_dedup iir_full_group ii4_full_dedup ii4_full_group

# 6. cross-dataset transfer (original inputs, then both as instruction sequences)
python -m experiments.cross_dataset
python -m experiments.cross_dataset --matched

# 7. source grounding on Dataset I (needs internet for Sourcify, plus slither and solc-select)
for s in 42 123 456 789 2024; do
  n=30; [ $s = 42 ] && n=60
  python -m experiments.source_localisation disasm $s $n
  python -m experiments.source_localisation disasm_group $s 40
done

# 8. figures
python figures/make_figures.py
