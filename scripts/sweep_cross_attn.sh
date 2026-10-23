#!/bin/bash
# Stage 2: bidirectional cross-attention (Dual-Times2D.md sections 14 / 33).
#
# READ THIS FIRST. The section-32 gate in the design doc says cross-attention
# should only be attempted once the Dual is *stably better* than single
# Times2D. Measured on ETTh1 that is not the case:
#
#   pred    Times2D    Phase     Dual
#     96     0.3670    0.3624    0.3653
#    192     0.4036    0.4039    0.4025
#    336     0.4231    0.4225    0.4225
#    720     0.4488    0.4495    0.4642   (0.4535 with ckpt_avg_k=5)
#
# The Phase branch alone is never worse than the Dual, so any gain CA produces
# is measured against a fusion that is not yet earning its place. Run this
# anyway if you want the ablation, but read the verdict step below before
# concluding anything.
#
# Scaling reference: at pred 720 the Dual's own seed spread is 0.011 MSE
# (0.4533..0.4642 across 3 seeds). A CA effect smaller than that is not
# distinguishable from seed noise at one seed per config.
#
# Usage:
#   bash scripts/sweep_cross_attn.sh          # all horizons, CA on vs off
#   bash scripts/sweep_cross_attn.sh 720      # just one horizon
set -u

cd "$(dirname "$0")/.." || exit 1
HORIZONS="${1:-96 192 336 720}"

for PL in $HORIZONS; do
  echo
  echo "################ pred_len=${PL}: Dual WITHOUT cross-attention ################"
  TAG="pl${PL}noca" PRED_LENS="$PL" MODELS=DualTimes2D SEED=2021 \
    bash scripts/run_etth1.sh

  echo
  echo "################ pred_len=${PL}: Dual WITH cross-attention ################"
  TAG="pl${PL}ca1" PRED_LENS="$PL" MODELS=DualTimes2D SEED=2021 CROSS_ATTN=1 \
    bash scripts/run_etth1.sh
done

echo
echo "==================== VERDICT ===================="
echo "Compare the _pl<N>noca and _pl<N>ca1 rows:"
echo "  python build_results_table.py --tagged"
echo
echo "CA is only evidence FOR the module if the improvement clears the seed"
echo "spread (~0.011 at 720). If it is smaller, or if CA helps at some horizons"
echo "and hurts at others, that is noise, not a mechanism."
echo
echo "If CA does not clear the bar, the honest next step is to fix the fusion"
echo "(per-branch heads + learned weighting) BEFORE stacking more modules --"
echo "CA on top of a fusion that already underperforms a single branch cannot"
echo "isolate what CA contributed."
