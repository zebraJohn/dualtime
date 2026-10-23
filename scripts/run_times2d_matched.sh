#!/bin/bash
# Standalone Times2D under the SAME protocol and tag as run_fredn_standalone.sh.
#
# WHY THIS EXISTS
# ---------------
# The rho guard in docs/superpowers/plans/2026-09-27-dualfredn.md Task 7 Step 2 is
#
#   python scripts/analyze_complementarity.py --model-a Times2D --model-b FreDN \
#       --tag 'fre_s20*'
#
# but that command CANNOT WORK as the plan wrote it. analyze_complementarity.py
# globs `results/..._{model}_*_Exp_{tag}`, and the repo's existing Times2D runs
# carry the tag `refproto_s20*`, not `fre_s20*` -- so it would find 12 FreDN runs
# and ZERO Times2D runs and silently skip every horizon.
#
# The fix is to give Times2D the matched tag. It must also get the matched
# PROTOCOL: the existing Times2D runs trained with Huber loss, while the
# DualFreDN arms train with the frequency-domain MAE. Comparing residual
# correlations across two different training objectives would confound exactly
# what the guard exists to disambiguate -- and PROGRESS.md's "遗留的诚实说明"
# table is already a list of protocol-mixing mistakes.
#
# Matching run_fre_experiment.sh exactly. Times2D ignores the --fre_* knobs, but
# they are passed anyway so the setting name construction is identical.
#
#   bash scripts/run_times2d_matched.sh          # 12 runs, ~3 h
set -u

cd "$(dirname "$0")/.." || exit 1

for seed in 2021 2022 2023; do
  for pl in 96 192 336 720; do
    echo "######## standalone Times2D  seed=$seed  pred_len=$pl ########"
    env MODELS=Times2D PRED_LENS="$pl" SEED="$seed" TAG="fre_s${seed}" \
        FRE_EMBED_SIZE=32 FRE_HIDDEN_SIZE=64 FRE_HIDDEN_LAYERS=1 FRE_DROPOUT=0.1 \
        LR=0.001 LRADJ=none LOSS=freqmae TF32=1 \
        bash scripts/run_etth1.sh
  done
done

echo
echo "Now: python scripts/analyze_complementarity.py --model-a Times2D --model-b FreDN --tag 'fre_s20*'"
