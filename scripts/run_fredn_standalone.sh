#!/bin/bash
# Standalone FreDN under the same protocol as the DualFreDN arms, so that
# scripts/analyze_complementarity.py can measure rho(Times2D, FreDN).
#
# This is the spec's rho guard (docs/superpowers/specs/2026-09-27-dualfredn-design.md
# section 8.2). It must be computed BEFORE reading the CA ablation, because it
# decides how a null CA result is attributed.
#
# The protocol deliberately matches run_fre_experiment.sh (fre_embed_size=32,
# frequency-domain MAE, constant lr) rather than FreDN's paper recipe: the
# question is whether the branch AS IT ENTERS THE DUAL is complementary to
# Times2D. For a paper-faithful FreDN reproduction see D:\pycharm\FreDN and
# scripts/eval_etth1_ckpts.sh there.
#
#   bash scripts/run_fredn_standalone.sh
set -u

cd "$(dirname "$0")/.." || exit 1

declare -A FRE_DROPOUT=( [96]=0.3   [192]=0.3  [336]=0.1   [720]=0.1 )
declare -A FRE_HLAYERS=( [96]=1     [192]=1    [336]=2     [720]=4   )
declare -A FRE_HSIZE=(   [96]=64    [192]=16   [336]=128   [720]=64  )

for seed in 2021 2022 2023; do
  for pl in 96 192 336 720; do
    echo "######## standalone FreDN  seed=$seed  pred_len=$pl ########"
    env MODELS=FreDN PRED_LENS="$pl" SEED="$seed" TAG="fre_s${seed}" \
        FRE_EMBED_SIZE=32 LR=0.001 LRADJ=none LOSS=freqmae TF32=1 \
        FRE_DROPOUT="${FRE_DROPOUT[$pl]}" \
        FRE_HIDDEN_LAYERS="${FRE_HLAYERS[$pl]}" \
        FRE_HIDDEN_SIZE="${FRE_HSIZE[$pl]}" \
        bash scripts/run_etth1.sh
  done
done

echo
echo "This is only HALF the rho guard. You also need the Times2D half:"
echo "    bash scripts/run_times2d_matched.sh"
echo "then:"
echo "    python scripts/analyze_complementarity.py --model-a Times2D --model-b FreDN --tag 'fre_s20*'"
echo
echo "Running only this script finds 12 FreDN runs and 0 Times2D runs, and the"
echo "guard reports a MISSING line per horizon (exit 1)."
