#!/bin/bash
# Is "PhaseFormer beats Times2D by 2.7%" real, or one seed's luck?
#
# THE PROBLEM
# Section 4.3 of PROGRESS.md reports PhaseFormer averaging 0.4030 against
# Times2D's 0.4142 under the reference protocol -- a 2.7% lead. But both sides
# are a SINGLE seed (2021). The Times2D-family seed spread at ETTh1-720 is
# ~0.011 MSE, which is the same size as the lead being claimed, so at one seed
# the result is not distinguishable from seed noise. It is the single biggest
# gap in the current evidence.
#
# WHAT THIS RUNS
# Both branches, all four horizons, seeds 2021/2022/2023, reference protocol
#   (constant lr=1e-3, Huber d=1.0, TF32 on).
#
# WHY EACH SEED NEEDS ITS OWN TAG
# The random seed is NOT part of the experiment setting name, so three seeds
# sharing one tag resolve to the same checkpoint/results path and silently
# overwrite each other. Hence refproto_s2021 / _s2022 / _s2023.
#
# WHY SEED 2021 IS RE-RUN RATHER THAN REUSED
# Two reasons. It keeps the tag scheme uniform, and it doubles as a determinism
# check: seed 2021 must reproduce the existing untagged `refproto` numbers
# exactly (Times2D: 0.3710/0.4062/0.4309/0.4485, PhaseFormer:
# 0.3763/0.3967/0.4120/0.4269). If it does not, something in the pipeline
# changed and nothing from this run is trustworthy. Check this FIRST.
#
# COST: PhaseFormer ~2 min/run x 12 = ~25 min. Times2D ~3 min/run x 12 = ~36 min.
# Total roughly one hour.
#
# Usage:
#   bash scripts/run_refproto_seeds.sh
#   HORIZONS=720 bash scripts/run_refproto_seeds.sh     # ~15 min
set -u

cd "$(dirname "$0")/.." || exit 1
HORIZONS="${HORIZONS:-96 192 336 720}"

for SEED in 2021 2022 2023; do
  echo
  echo "################ seed=$SEED  PhaseFormer ################"
  MODELS=PhaseFormer PRED_LENS="$HORIZONS" SEED="$SEED" TAG="refproto_s$SEED" \
    LR=0.001 LRADJ=none LOSS=huber HUBER_DELTA=1.0 TF32=1 \
    bash scripts/run_etth1.sh

  echo
  echo "################ seed=$SEED  Times2D ################"
  MODELS=Times2D PRED_LENS="$HORIZONS" SEED="$SEED" TAG="refproto_s$SEED" \
    LR=0.001 LRADJ=none LOSS=huber HUBER_DELTA=1.0 TF32=1 \
    bash scripts/run_etth1.sh
done

echo
echo "==================== STEP 1: determinism check ===================="
echo "Before anything else, confirm seed 2021 reproduced the earlier run. The"
echo "tagged seed-2021 rows must match these untagged ones:"
echo "  Times2D     refproto : 0.3710 0.4062 0.4309 0.4485"
echo "  PhaseFormer refproto : 0.3763 0.3967 0.4120 0.4269"
echo "  python build_results_table.py --tagged"
echo "If they differ, STOP -- the pipeline is not deterministic and the rest of"
echo "this run cannot be interpreted."
echo
echo "==================== STEP 2: the 3-seed answer ===================="
echo "  python scripts/analyze_complementarity.py --tag 'refproto_s20*'"
echo
echo "Each cell prints as mean (min..max) over the three seeds, and every horizon"
echo "reports how many seeds gave a positive gain. Read it like this:"
echo
echo "  gain consistently positive across 3 seeds AND mean gain > 0.011"
echo "      -> the complementarity case is real. This is the one outcome that"
echo "         would justify building Cross-Attention."
echo
echo "  gain changes sign across seeds"
echo "      -> it was noise. The single-seed 2.7% lead evaporates."
echo
echo "  gain positive but mean < 0.011"
echo "      -> real but below the noise floor: still not actionable."
echo
echo "Note this also re-tests the PhaseFormer-over-Times2D ranking itself, which"
echo "is separate from the complementarity question and currently rests on the"
echo "same single seed."
