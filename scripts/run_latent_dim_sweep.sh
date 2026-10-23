#!/bin/bash
# Step 1 of phase2d-v2.md (section 8.2): the PhaseFormer latent_dim sweep.
#
# WHY THIS EXISTS
# The Phase branch's latent Z is [B, C, period_len, latent_dim]. At latent_dim=6
# the dual model would have to cross-attend over a rank-6 subspace projected up
# to d_model=64, so its Q/K would be nearly degenerate -- the single most likely
# way for Step 5 to "run, produce gradients, and do nothing". This sweep answers
# whether a wider latent costs accuracy BEFORE any dual-branch machinery is
# built on top of it, which is much cheaper than discovering it afterwards.
#
# 4 horizons x 3 values = 12 runs at ~8.6 min each, so roughly 1.7 h.
#
# A CONFOUND, STATED UP FRONT
# PhaseFormer-test's ETTh1-720 lat32 reference run also used phase_layers=3.
# This sweep holds phase_layers=1 so latent_dim is the only variable, so the
# lat32 row here will NOT reproduce that run. Do not compare the two directly.
#
# phase_attn_heads must divide latent_dim (CrossPhaseRoutingLayer asserts it).
# 6/32/64 use heads 1/4/4: heads=6 on latent_dim=6 would mean head_dim=1 and is
# not a meaningful comparison against the wider configs.
#
# Usage:
#   bash scripts/run_latent_dim_sweep.sh              # all four horizons
#   HORIZONS=720 bash scripts/run_latent_dim_sweep.sh # just one
set -u

cd "$(dirname "$0")/.." || exit 1
HORIZONS="${HORIZONS:-96 192 336 720}"

run_one() {  # run_one <latent_dim> <heads>
  echo
  echo "################ latent_dim=$1  heads=$2  layers=1 ################"
  PF_LATENT_DIM="$1" PF_HEADS="$2" PF_LAYERS=1 \
    MODELS=PhaseFormer PRED_LENS="$HORIZONS" TAG="lat$1" \
    bash scripts/run_etth1.sh
}

run_one 6  1
run_one 32 4
run_one 64 4

echo
echo "==================== VERDICT ===================="
echo "The three sweeps are tagged _lat6 / _lat32 / _lat64. Inspect with:"
echo "  python build_results_table.py --tagged"
echo
echo "Rule for picking one: the largest latent_dim that does not LOSE accuracy"
echo "against lat6. latent_dim=64 makes the Feature Adapter an identity"
echo "(d_model=64) and removes the rank bottleneck outright, so prefer it if it"
echo "is not worse."
echo
echo "Remember the noise floor before reading anything into small gaps:"
echo "PhaseTimes2D/DualTimes2D seed spread on ETTh1-720 is ~0.011 MSE. A single"
echo "seed per config cannot resolve a difference smaller than that -- run the"
echo "chosen config at 3 seeds before treating the ranking as real."
echo
echo "If a wider latent HURTS, that is itself the Step 5 finding: the dual"
echo "branch has to be built around a narrow Z, and the two-layer MLP adapter in"
echo "phase2d-v2.md section 8.2 (not a single Linear) becomes mandatory."
