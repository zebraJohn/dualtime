#!/bin/bash
# The DualFreEformer cross-attention ablation: Times2D + full FreEformer, with
# the CA running on the CHANNEL axis instead of the B-per-channel batch axis
# every earlier dual used.
#
# V1 (FreEformer squeezed into the Times2D protocol -- see the v2 block below):
#   bash scripts/run_freeformer_experiment.sh smoke    # 1 run, for s/epoch
#   bash scripts/run_freeformer_experiment.sh stage1   # 8 runs -- go/no-go
#   bash scripts/run_freeformer_experiment.sh stage2   # 16 runs
#
# V2 (unequal width -- FreEformer at its NATIVE 512, Times2D untouched):
#   bash scripts/run_freeformer_experiment.sh uw-smoke   # 1 run, for s/epoch
#   bash scripts/run_freeformer_experiment.sh uw-stage1  # 8 runs -- go/no-go
#   bash scripts/run_freeformer_experiment.sh uw-stage2  # 16 runs
#
# THE `uw` TAGS ARE LOAD-BEARING. configs.d_model is still 64 in v2 (it is the
# PATCH branch's width), so a v2 run's setting name is byte-identical to v1's --
# without the `uw` prefix it would silently overwrite v1's checkpoints and
# results. See the v2 section below and the v2 spec's 4.2.
#
# Mirrors scripts/run_fre_experiment.sh, with two deliberate differences, both
# of which are pre-registered and both of which are in PROGRESS.md's record.
#
# ---------------------------------------------------------------------------
# DIFFERENCE 1 -- fixed epoch count, early stopping OFF
# ---------------------------------------------------------------------------
# PROGRESS.md 9.5 records a methodological problem that is INDEPENDENT of CA:
# early stopping let the two arms train for different numbers of epochs (96:
# 14 vs 11; 720: 22 vs 11), and on 720 the checkpoint choice was effectively a
# coin flip -- three seeds' best_vali spanned 0.028 while their test MSE
# spanned 0.0580. One 720 cell dragged 9.3's mean from +0.0034 to -0.0441.
#
# So both arms here run TRAIN_EPOCHS=20 with PATIENCE=20 (patience >= epochs
# means it never fires), and both arms are therefore trained identically.
#
# N=20 was chosen BEFORE any result existed: 9.3's arms stopped between 11 and
# 22 epochs. If smoke shows a horizon is clearly still descending at 20, that
# is the only reason to raise it -- and raising it must happen, and be
# committed, BEFORE any ca0/ca1 number is read. Changing it afterwards is not
# a protocol tweak, it is a different experiment.
#
# Cost of this choice: these numbers CANNOT be merged into 9.3's DualFreDN
# table, because the protocol differs. Report them as their own table and say so.
#
# ---------------------------------------------------------------------------
# DIFFERENCE 2 -- the model already mixes channels twice before the CA runs
# ---------------------------------------------------------------------------
# Spec 2.2.1, found while implementing: FreEformer's own Encoder_ori attends
# over the CHANNEL axis (upstream builds FullAttention_ablation with
# token_num=enc_in, so its softmax is over channels and its learned weight_mat
# is [1,1,enc_in,enc_in]). Measured standalone, perturbing channel 0 moves the
# other channels by 4.5e-02 against a 2.6e-01 output scale.
#
# So this model stacks THREE mixers on the channel axis: FreEformer's
# attention, its weight_mat prior, and the dual's CA. It does NOT introduce
# cross-channel information to a model that lacked it, and switching the CA
# off does NOT make the dual per-channel (that is what broke the first version
# of the control in tests/test_cross_channel.py).
#
# READ A NULL RESULT ACCORDINGLY. "CA did not help" here may mean this is the
# stack PROGRESS 11.5 predicted would not help -- "MLP and attention are doing
# the same thing on the channel axis" -- and must NOT be recorded as
# "cross-channel modelling does not work".
#
# ---------------------------------------------------------------------------
# HOW TO READ THE RESULT -- the criterion is fixed, not chosen afterwards
# ---------------------------------------------------------------------------
# Spec 8.1, unchanged from 9.3:
#   4/4 horizons the SAME SIGN and mean |delta| ABOVE the floor (ETTh1 0.011)
#     -> CA helps
#   anything else
#     -> "CA does not help", report and stop
# Picking a horizon that looks good does not count: 11.4 already names that as
# the exact thing pre-registration exists to prevent.
#
# The rho guard is deliberately NOT run (spec 8.2) -- this round skips the 11
# pre-gate. If it is ever run, it must be before these numbers are read, and it
# cannot override the criterion above.
#
# COST, MEASURED: 17.5 s/epoch at pred_len=720, seq_len=720, enc_in=7,
# batch_size=128, on cuda:0 (d_model=64), from a COMPLETED smoke that ran the
# whole pipeline through inference. Within the epoch: forward 0.114 s/step,
# backward+optimizer 0.186 s/step, 56 steps.
#
# Stage 1 is 8 runs: 8 x 20 epochs x ~17.5 s ~= 47 min, plus per-epoch
# validation and test. Budget ~1-1.5 h.
#
# An earlier interrupted run of the same config measured 24.6 s/epoch, so treat
# 17.5-25 s as the working range rather than a tight bound.
#
# d_model here is 64, not upstream's 512, and fc's input width is
# seq_len*embed_size = 720*16 = 11520 -- so upstream's published timings do not
# transfer and this number is the only useful one.
#
# V2 COST, MEASURED: 16.9 s/epoch from a completed `uw-smoke` -- the SAME
# setting, 56 steps, batch 128, but with FreEformer at its native 512 instead of
# 64. So the 8x widening is FREE in wall-clock (16.94 vs 17.49; forward 0.1139
# vs 0.114 s/step, backward 0.176 vs 0.186). The per-step cost is dominated by
# Times2D's conv/TST backbone, and the widened FreEformer matmuls
# (fre_trans_* at 5776->512, fc at 11520->512) are small next to it.
# uw-stage1 is therefore ~1 h, not the order-of-magnitude more the earlier
# version of this comment guessed.
set -u

cd "$(dirname "$0")/.." || exit 1
STAGE="${1:-stage1}"

COMMON="FE_EMBED_SIZE=16 \
FE_ATTN_ENHANCE=1 FE_ATTN_SOFTMAX=0 FE_ATTN_WEIGHT_PLUS=1 FE_ATTN_OUTSIDE_SOFTMAX=1 \
FE_PLOT_MAT=0 FE_PLOT_GRAD=0 FE_CKA=0 \
LR=0.001 LRADJ=none LOSS=freqmae TF32=1 \
TRAIN_EPOCHS=20 PATIENCE=20"

# FreEformer's NATIVE architecture. Times2D does not change -- its own official
# ETTh1 script (Times2D/scripts/times2D/longTerm/Times2D_ETTh1.sh:51-65) already
# matches this repo's d_model=64 / n_heads=16 / e_layers=3 / dropout=0.5.
# Sources for the five values below: FreEformer/scripts/ETTh1_freformer_attn1.sh
# and the recorded best config in its best_results/.
UW_ARCH="FE_D_MODEL=512 FE_D_FF=512 FE_N_HEADS=8 FE_E_LAYERS=2 FE_DROPOUT=0.2"

run_arm() {  # run_arm <seed> <ca 0|1> [uw]
  local seed="$1" ca="$2" uw="${3:-}"
  local tag="ca${ca}_s${seed}"
  local extra=""
  [ "$ca" = "1" ] && extra="CROSS_ATTN=1 CA_LAYERS=1 CA_HEADS=4"
  if [ "$uw" = "uw" ]; then
    # The tag prefix is LOAD-BEARING, not cosmetic: configs.d_model is still 64
    # in v2 (it is the PATCH branch's width), so the setting name is
    # byte-identical to v1's and an untagged run would silently overwrite v1's
    # checkpoints and results (spec 4.2).
    tag="uw${tag}"
    extra="$UW_ARCH $extra"
  fi
  echo
  echo "######## seed=$seed  cross_attn=$ca  uw=${uw:-no}  (tag=$tag) ########"
  for pl in 96 192 336 720; do
    env MODELS=DualFreEformer PRED_LENS="$pl" SEED="$seed" TAG="$tag" \
        $COMMON $extra bash scripts/run_etth1.sh
  done
}

if [ "$STAGE" = "uw-smoke" ]; then
  echo "######## UW SMOKE: 1 epoch at FreEformer's native 512, for timing ########"
  # Same env-ordering rule as the v1 smoke below: the override MUST come after
  # $COMMON or COMMON's TRAIN_EPOCHS=20 silently wins.
  env MODELS=DualFreEformer PRED_LENS=720 SEED=2021 TAG=uw-smoke \
      $COMMON $UW_ARCH TRAIN_EPOCHS=1 PATIENCE=1 \
      bash scripts/run_etth1.sh
  cat <<'EOF'

==================== UW SMOKE DONE ====================
Record s/epoch in this script's header before running uw-stage1. It is the one
number that decides how the ~24-run v2 ablation is staged, and d_model=512 is
8x the width the 17.5 s/epoch v1 figure was measured at -- expect it to be
worse, but by how much is unmeasured until this finishes.

Also confirm the log shows a real epoch and a finite mse (PROGRESS 10.5).
EOF
  exit 0
fi

if [ "$STAGE" = "smoke" ]; then
  echo "######## SMOKE: 1 epoch of DualFreEformer at pred_len=720, for timing ########"
  # The override MUST come after $COMMON. `env` applies assignments left to
  # right, so putting it first is silently overwritten by COMMON's
  # TRAIN_EPOCHS=20 PATIENCE=20 -- and the run would take 20x its advertised
  # cost while the banner above still said "1 epoch".
  env MODELS=DualFreEformer PRED_LENS=720 SEED=2021 TAG=smoke \
      $COMMON TRAIN_EPOCHS=1 PATIENCE=1 \
      bash scripts/run_etth1.sh
  cat <<'EOF'

==================== SMOKE DONE ====================
Read s/epoch out of the log, then RECORD IT in this script's header before
running stage1 -- that is the house convention (see
scripts/run_crosschannel_standalone.sh's "COST, MEASURED" block), and it is how
the next person knows whether 24 runs is an afternoon or a week.

Also confirm from the log that the run reached a real epoch and reported a
finite mse. A smoke that only checks shapes is how the hourly-weather NaN
episode got through (PROGRESS 10.5).
EOF
  exit 0
fi

if [ "$STAGE" = "stage1" ]; then
  run_arm 2021 0
  run_arm 2021 1
elif [ "$STAGE" = "uw-stage1" ]; then
  run_arm 2021 0 uw
  run_arm 2021 1 uw
elif [ "$STAGE" = "uw-stage2" ]; then
  for s in 2022 2023; do
    run_arm "$s" 0 uw
    run_arm "$s" 1 uw
  done
else
  for s in 2022 2023; do
    run_arm "$s" 0
    run_arm "$s" 1
  done
fi

if [ "$STAGE" = "uw-stage1" ] || [ "$STAGE" = "uw-stage2" ]; then
  cat <<'EOF'

==================== HOW TO READ THIS (v2) ====================
These are the UNEQUAL-WIDTH runs. FreEformer is at its native 512 and Times2D is
untouched at 64, so this is not comparable to the v1 rows in PROGRESS 12 -- a
different configuration, a different tag, its own section.

  python build_results_table.py --tagged

BEFORE reading any MSE, run the matching rho guard and finish it:

  bash scripts/run_freeformer_rho.sh all512
  python scripts/analyze_complementarity.py --model-a Times2D --model-b FreEformer --tag 'uw_s20*'

That is NOT optional and NOT the v1 rule. v1 pre-registered the guard as skipped
(spec 8.2) and the cost was PROGRESS 12.9 -- a whole extra round to work out why
the null happened. v2 spec 6.2 makes it binding: the guard runs BEFORE the CA
numbers are read, or it is hindsight.

Then read CA and rho TOGETHER, against the four pre-registered cells (spec 6.3):
  CA over floor                       -> CA helps
  CA under floor, oracle under floor  -> nothing was fusable; a clean negative
  CA under floor, oracle OVER floor   -> fusable and CA missed it; the mechanism
  CA over floor,  oracle under floor  -> contradicts the oracle bound; check the
                                         measurement before believing it

Report MSE_A and MSE_B (the two standalone models) in EVERY case -- 12.9 showed
branch-quality imbalance, not rho, was the explanatory variable, and that must
be visible in the result rather than reconstructed later.

Also check that both arms ran the same number of epochs before reading anything.
EOF
  exit 0
fi

cat <<'EOF'

==================== HOW TO READ THIS ====================
  python build_results_table.py --tagged

Then, BEFORE looking at any MSE, check from the logs that both arms of every
horizon ran the SAME number of epochs. If they did not, the ablation is not
clean and that has to be said out loud rather than worked around.

Stage 1 is seed 2021 only: 2 arms x 4 horizons = 8 runs, and it is the go/no-go.
Apply the spec 8.1 criterion to it mechanically. Stage 2 exists to confirm, not
to rescue a stage-1 miss.

Report with:
  * the protocol deviation from 9.3 (fixed epochs, no early stopping) stated
    plainly, and the two results NOT merged into one table
  * the epoch counts of both arms
  * the fact that rho was deliberately not run (spec 8.2)
  * spec 2.2.1's three-channel-axis-mixers caveat, so a null is not recorded as
    "cross-channel does not work"
EOF
