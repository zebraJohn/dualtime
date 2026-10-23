#!/bin/bash
# Does feature-level Cross-Attention actually help? The one question no probe
# could answer.
#
# WHERE THIS COMES FROM
# Everything up to here measured things from the OUTSIDE of a dual model:
#   * residual correlation rho = 0.9377 (3 seeds, 12/12 runs)
#   * a TEST-FITTED fixed blend of the two branches' outputs gains +0.0070
# Both bound what a fixed linear combination can do. Neither can bound what a
# jointly trained model does, because CA lets one branch's latent modulate how
# the other reads its own features -- something no blend of final outputs can
# express. Three attempts at a frozen-feature probe failed to close that gap
# (see PROGRESS.md section 5); only building it can.
#
# PREREQUISITE -- DO NOT SKIP
# Run this at latent_dim=64, not the paper's 6. The phase branch enters CA
# through Linear(latent_dim -> d_model); at 6 that is a rank-<=6 map into a
# 64-wide space, so attention queries and keys live in a 6-dimensional subspace.
# That is a structural limit, not a tuning one, and it would produce a null
# result that means nothing. At latent_dim=64 the adapter is an Identity (the
# model prints this at build time) and the bottleneck is gone.
#   PF_LATENT_DIM=64 PF_HEADS=4 (heads must divide latent_dim)
#
# PROTOCOL: constant lr=1e-3 + MSE, deliberately NOT Huber.
# The reference protocol's Huber loss is what destabilised standalone Times2D
# (seed spread 0.0937 at pl=720 -- PROGRESS.md section 4.2). Huber at delta=1.0
# on RevIN-normalised data degenerates to 0.5*d^2, halving the gradient, which
# makes the early-stopping point seed-dependent. There is no reason to import
# that instability into a model we are trying to measure.
#
# STAGES -- run stage1 first, it is the go/no-go
#   stage1  1 seed  x 2 arms x 4 horizons =  8 runs  ~1.5 h
#   stage2  2 more seeds, same           = 16 runs  ~3.2 h
# Only run stage2 if stage1 shows a signal. Spending 5 h before looking is how
# you end up unable to tell a null result from a bug.
#
# Usage:
#   bash scripts/run_ca_experiment.sh stage1
#   bash scripts/run_ca_experiment.sh stage2
#   HORIZONS=720 bash scripts/run_ca_experiment.sh stage1   # ~25 min
set -u

cd "$(dirname "$0")/.." || exit 1
STAGE="${1:-stage1}"
HORIZONS="${HORIZONS:-96 192 336 720}"

# Shared knobs. latent_dim=64 is the prerequisite; heads must divide it.
COMMON="PF_LATENT_DIM=64 PF_HEADS=4 LR=0.001 LRADJ=none LOSS=mse TF32=1"

run_arm() {  # run_arm <seed> <ca 0|1>
  local seed="$1" ca="$2"
  local tag="ca${ca}_s${seed}"
  local extra=""
  [ "$ca" = "1" ] && extra="CROSS_ATTN=1 CA_LAYERS=1 CA_HEADS=4"
  echo
  echo "######## seed=$seed  cross_attn=$ca  (tag=$tag) ########"
  env MODELS=DualPhaseFormer PRED_LENS="$HORIZONS" SEED="$seed" TAG="$tag" \
      $COMMON $extra bash scripts/run_etth1.sh
}

if [ "$STAGE" = "stage1" ]; then
  run_arm 2021 0
  run_arm 2021 1
else
  for s in 2022 2023; do
    run_arm "$s" 0
    run_arm "$s" 1
  done
fi

echo
echo "==================== HOW TO READ THIS ===================="
echo "  python build_results_table.py --tagged"
echo
echo "Two questions, in order. Answer them separately -- they are not the same,"
echo "and the model has per-branch prediction terms (alpha*Y_patch + beta*Y_phase)"
echo "that contribute to the first but cancel out of the second."
echo
echo "1) DOES THE DUAL BEAT A SINGLE BRANCH?"
echo "     DualPhaseFormer (ca0 or ca1) vs Times2D and vs PhaseFormer."
echo "     Reference: the test-fitted ORACLE for blending the two branches'"
echo "     outputs is +0.0070 over the best single branch. That is an upper bound"
echo "     on any fixed blend; a trained fusion should not be expected to reach it."
echo
echo "2) DOES CROSS-ATTENTION ADD ANYTHING?"
echo "     ca1 vs ca0. THIS is what the whole exercise was for."
echo
echo "DECISION RULES (set before looking, so the result cannot be rationalised):"
echo "  ca1 beats ca0 by > 0.011 (the ETTh1-720 seed floor) AND consistently"
echo "      -> CA works. Then, and only then, consider adding MoE / Memory."
echo "  ca1 ~= ca0"
echo "      -> CA adds nothing. The +0.0070 oracle headroom is not reachable"
echo "         through feature interaction. Report it and stop."
echo "  ca1 worse than ca0"
echo "      -> matches v1, which also saw CA hurt at 3 of 4 horizons."
echo
echo "ALSO CHECK: inspect alpha and beta in the logs. If the optimiser drives them"
echo "toward 0, it is telling you the fused path is worse than the branch"
echo "predictions -- i.e. the fusion itself (CA or not) is not earning its place."
