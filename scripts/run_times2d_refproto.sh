#!/bin/bash
# THE DECIDING EXPERIMENT: can Times2D hold up under the reference protocol?
#
# WHY THIS BLOCKS EVERYTHING ELSE
# scripts/run_phaseformer_protocol.sh established that under the unified protocol
# (lr=1e-4, OneCycleLR, MSE, no TF32) PhaseFormer is suppressed by 0.0199 MSE --
# nearly 2x the ETTh1-720 seed noise floor. Under the reference protocol
# (constant lr=1e-3, Huber d=1.0, TF32) it scores 0.40297 average, matching the
# paper's Table 5 (0.403) to three decimals, and beats Times2D by 1.9%.
#
# But phase2d-v2.md section 14 assumes both branches are compared under ONE
# protocol, and the dual model has to TRAIN both branches under one protocol.
# Every Times2D-family number in this repo comes from the unified protocol, and
# Times2D has never been run under the reference protocol. So we do not know
# whether "adopt the reference protocol project-wide" is even available:
#
#   * If Times2D is roughly INSENSITIVE to the protocol change, one protocol
#     covers both branches and the problem disappears. Adopt it, move on.
#   * If Times2D DEGRADES under it, the two branches genuinely want different
#     protocols. A jointly-trained dual branch then needs per-branch lr /
#     schedules, or a compromise -- and that threatens Step 5 outright.
#
# 4 horizons x ~10 min = roughly 40 min. Times2D (4.3M params) is far slower
# than PhaseFormer (1,278 params), which is why this is the slow half.
#
# NOTE ON TF32: torch.set_float32_matmul_precision("medium") only governs
# matmuls. Convolutions are governed separately by torch.backends.cudnn.allow_tf32,
# which already defaults to True -- so Times2D's conv2D blocks were using TF32
# all along and this flag changes only its attention matmuls.
#
# Usage:
#   bash scripts/run_times2d_refproto.sh
#   HORIZONS=720 bash scripts/run_times2d_refproto.sh
set -u

cd "$(dirname "$0")/.." || exit 1
HORIZONS="${HORIZONS:-96 192 336 720}"

echo
echo "############ Times2D @ reference protocol ############"
echo "  lr=1e-3 CONSTANT, Huber d=1.0, TF32 on, tag=refproto"
MODELS=Times2D PRED_LENS="$HORIZONS" TAG=refproto \
  LR=0.001 LRADJ=none LOSS=huber HUBER_DELTA=1.0 TF32=1 \
  bash scripts/run_etth1.sh

echo
echo "==================== HOW TO READ THIS ===================="
echo "  python build_results_table.py --tagged"
echo
echo "Four rows matter. Average MSE over the four horizons:"
echo
echo "  Times2D     unified protocol ....... 0.4106   <- the baseline to beat"
echo "  Times2D     refproto    ............ ?        <- THIS RUN"
echo "  PhaseFormer unified protocol ....... 0.4229"
echo "  PhaseFormer refproto    ............ 0.4030   <- matches paper Table 5"
echo
echo "DIAGNOSIS:"
echo
echo "  If Times2D-refproto stays near 0.4106 (within the ~0.011 floor),"
echo "  Times2D is insensitive to the protocol. Then ONE protocol covers both"
echo "  branches: adopt the reference protocol project-wide and revisit"
echo "  section 14, and Step 2 can proceed on refproto checkpoints."
echo
echo "  If Times2D-refproto is clearly WORSE than 0.4106, the two branches want"
echo "  different protocols. That is not a dead end, but it means the dual model"
echo "  needs per-branch lr/schedule support before Step 5 -- a real design"
echo "  change, and one to price in now rather than after the fusion is built."
echo
echo "  If Times2D-refproto is clearly BETTER, the current baseline itself is"
echo "  under-trained and every comparison in this repo needs redoing."
echo
echo "Single seed per horizon -- 720's seed spread is ~0.011, so ignore any"
echo "difference smaller than that until 3 seeds are run."
