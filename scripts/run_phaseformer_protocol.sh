#!/bin/bash
# Step 1 of phase2d-v2.md (sections 7.1 / 7.3): why is our PhaseFormer worse
# than the paper's?
#
# THE OBSERVED PROBLEM
# With the unified protocol (lr=1e-4, OneCycleLR, MSE, no TF32) PhaseFormer
# scores 0.3864 / 0.4108 / 0.4377 / 0.4567 on ETTh1 -- WORSE than Times2D at
# every horizon, average 0.4229 vs 0.4106 (+3.0%). Meanwhile the reference
# reproduction gets 0.4086, and the paper claims 0.4030. The gap between our
# protocol and the reference reproduction (0.0143) is LARGER than the ETTh1-720
# seed noise floor (0.011), and the runs had converged (val loss flat by epoch
# 90), so this is not simply "needs more epochs".
#
# Reading PhaseFormer-test/src, the reference also differed in three more ways
# beyond the lr VALUE:
#   * CONSTANT lr -- run_etth1.py sets lr_schedule_config.type="type3", but
#     DefaultPLModule.configure_optimizers only handles "cos", so the scheduler
#     was silently skipped. No schedule was applied at all.
#   * Huber loss (delta=1.0), not MSE.
#   * TF32 enabled (torch.set_float32_matmul_precision("medium")).
#
# THREE ARMS, one variable at a time
#   1. lr=1e-4, OneCycle, MSE       (untagged) -- the unified-protocol baseline
#   2. lr=1e-3, OneCycle, MSE       (lr1e3)    -- isolates the lr VALUE
#   3. lr=1e-3, constant, Huber, TF32 (refproto) -- the full reference protocol
#
# Arm 2 vs 1 tells you whether the rate alone matters. Arm 3 vs 2 tells you
# whether the schedule/loss/precision matter. If arm 3 lands near 0.4086 while
# arms 1-2 do not, the unified protocol is what is handicapping the Phase
# branch, and phase2d-v2.md section 14 has to be revisited BEFORE Step 2 --
# otherwise the complementarity analysis is run on a deliberately weakened
# branch and means nothing.
#
# PhaseFormer is a ~1-3K parameter model (1,278 at pred_len=720), so a 100-epoch
# ETTh1 run takes minutes, not hours; all 12 runs together are well under an hour.
#
# Usage:
#   bash scripts/run_phaseformer_protocol.sh
#   HORIZONS=720 bash scripts/run_phaseformer_protocol.sh
set -u

cd "$(dirname "$0")/.." || exit 1
HORIZONS="${HORIZONS:-96 192 336 720}"

echo
echo "############ ARM 1: unified protocol  lr=1e-4 OneCycle MSE  (untagged) ############"
MODELS=PhaseFormer PRED_LENS="$HORIZONS" \
  LR=0.0001 LRADJ=TST LOSS=mse TF32=0 \
  bash scripts/run_etth1.sh

echo
echo "############ ARM 2: lr value only     lr=1e-3 OneCycle MSE  (lr1e3) ############"
MODELS=PhaseFormer PRED_LENS="$HORIZONS" TAG=lr1e3 \
  LR=0.001 LRADJ=TST LOSS=mse TF32=0 \
  bash scripts/run_etth1.sh

echo
echo "############ ARM 3: reference proto   lr=1e-3 CONSTANT Huber TF32 (refproto) ############"
MODELS=PhaseFormer PRED_LENS="$HORIZONS" TAG=refproto \
  LR=0.001 LRADJ=none LOSS=huber HUBER_DELTA=1.0 TF32=1 \
  bash scripts/run_etth1.sh

echo
echo "==================== HOW TO READ THIS ===================="
echo "  python build_results_table.py --tagged"
echo
echo "Reference points, ETTh1 average MSE over the four horizons:"
echo "  paper Table 5 ...................... 0.4030"
echo "  reference reproduction ............. 0.4086"
echo "  Times2D (our baseline) ............. 0.4106"
echo "  our PhaseFormer, unified protocol .. 0.4229   <- arm 1, already measured"
echo
echo "Noise floor: ETTh1-720 seed spread is ~0.011 MSE, and every arm here is"
echo "single-seed. Treat differences below that as unresolved until 3 seeds."
echo
echo "If ARM 3 lands near 0.4086 but arms 1-2 do not, the protocol -- not the"
echo "architecture -- is what is suppressing PhaseFormer. That invalidates the"
echo "current unified protocol as a basis for comparing the two branches, and"
echo "phase2d-v2.md section 14 must be revisited before Step 2."
echo
echo "If ARM 3 also lands near 0.42, PhaseFormer genuinely does not reach the"
echo "paper's numbers in this pipeline. Then the honest options are to report"
echo "that gap, or to reconsider the Phase branch entirely (v1's PhaseTimes2D"
echo "scored 0.4096 average and needs no new protocol at all)."
echo
echo "Do NOT start Step 2 (complementarity) on arm-1 checkpoints regardless --"
echo "they are produced by a protocol we have positive evidence is suppressing"
echo "the branch."
