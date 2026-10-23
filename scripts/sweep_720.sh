#!/bin/bash
# The ETTh1 pred_len=720 study.
#
# Why: at 720 the val loss barely discriminates between epochs. Among epochs
# within 1% of a model's best val loss, the TEST MSE spans ~1.8% for Times2D and
# ~5.8% for DualTimes2D -- so a single val-selected checkpoint is a high-variance
# estimate, and the reported 0.4488 (Times2D) vs 0.4642 (Dual) gap sits inside
# that band. Everything below is about deciding whether that gap is real, and
# only then about closing it.
#
# STAGE 1 (do this first): is the gap real?
#   Dual at 3 seeds. If the baseline's 0.4488 falls inside the Dual's spread,
#   there is no regression to fix and stage 2 is wasted GPU time.
#
# STAGE 2 (only if stage 1 says the gap is real): can it be closed?
#   Three independent levers, any of which can be combined.
#
# Usage:
#   bash scripts/sweep_720.sh stage1
#   bash scripts/sweep_720.sh stage2
#
# Rough cost at ~25 min/run on an RTX 4070 Laptop (8GB):
#   stage1 = 3 runs ~ 1.25 h     stage2 = 5 runs ~ 2 h
set -u

STAGE="${1:-stage1}"
cd "$(dirname "$0")/.." || exit 1

run() {  # run <tag> <env assignments...>
  local tag="$1"; shift
  echo
  echo "################ ${tag} ################"
  env "$@" TAG="$tag" PRED_LENS=720 MODELS=DualTimes2D bash scripts/run_etth1.sh
}

case "$STAGE" in
  stage1)
    # Seed sweep. seed 2021 also re-runs the existing Dual config, which is a
    # free reproducibility check: it must land on ~0.4642 again.
    run s2021 SEED=2021
    run s2022 SEED=2022
    run s2023 SEED=2023
    echo
    echo "Now: python build_results_table.py --tagged"
    echo "Look for: does 0.4488 (baseline) fall inside the Dual seed spread?"
    ;;

  stage2)
    # Each lever targets a different suspected cause.
    #  - wd:      general overfitting of the doubled parameter count
    #  - fusion:  the concat head is Linear(7200->720) = 5.18M params, the single
    #             largest block in the model (baseline's head is 2.59M)
    #  - avgk:    selection variance, the root cause identified above
    #  - all:     everything at once
    run wd01    WD=0.01
    run fus128  FUSION_DIM=128
    run avg5    CKPT_AVG_K=5
    run all     WD=0.01 FUSION_DIM=128 CKPT_AVG_K=5
    echo
    echo "Now: python build_results_table.py --tagged"
    echo
    echo "NOTE ON FAIRNESS: these tune the Dual only. If a tuned Dual beats the"
    echo "baseline, the same knobs should be swept for the baseline before"
    echo "claiming an architectural win -- otherwise you are comparing a tuned"
    echo "model against an untuned one."
    ;;

  *)
    echo "usage: bash scripts/sweep_720.sh [stage1|stage2]" >&2
    exit 1
    ;;
esac
