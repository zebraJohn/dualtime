#!/bin/bash
# Times2D X-shaped attention (PHAT-style PNA) ablation.
#
# Replaces each period's conv2D with period-offset x period-aligned attention,
# and optionally replaces FSDH's Inception conv too.  See
# layers/PeriodAttention.py and docs/plans for the design; the X is computed on
# the folded patch grid [H, W], not on raw timesteps.
#
# MODES
#   bash scripts/run_xattn_experiment.sh smoke    # 3 runs, 1 epoch, pl=96
#   bash scripts/run_xattn_experiment.sh stage1   # 12 runs, seed 2021
#   bash scripts/run_xattn_experiment.sh stage2   # 24 runs, seeds 2022/2023
#
# ARMS (one seeded tag per arm; the tags keep the arms from overwriting each
# other because none of the X-attn flags is in the experiment setting name):
#   xbase : X_ATTN=0 FSDH_X_ATTN=0   -- the conv baseline
#   xmain : X_ATTN=1 FSDH_X_ATTN=0   -- main branch only
#   xboth : X_ATTN=1 FSDH_X_ATTN=1   -- main + FSDH
#
# PROTOCOL
#   The default is the fixed 20-epoch, constant-lr Times2D/FreEformer protocol
#   (LR=0.001, LRADJ=none, LOSS=freqmae, TF32=1).  All three arms use it, so
#   the comparison is arm-vs-arm, not against a differently trained baseline.
#   Override LOSS/LR/LRADJ/TF32 only if you also intend to re-run xbase with the
#   same values; a tag that does not encode the protocol is a trap.
#
#   EPOCHS=20 is the default; set EPOCHS=<n> to change it.  The tags gain an
#   _e<n> suffix so different epoch budgets do not collide.
#
# EXPERIMENTS ARE RUN BY THE USER; this script only orchestrates run_etth1.sh.
# It does NOT train anything at import/source time.
set -u

cd "$(dirname "$0")/.." || exit 1

MODE="${1:-${MODE:-stage1}}"
HORIZONS="${HORIZONS:-96 192 336 720}"
EPOCHS="${EPOCHS:-20}"
LOSS="${LOSS:-freqmae}"
LR="${LR:-0.001}"
LRADJ="${LRADJ:-none}"
TF32="${TF32:-1}"
TAG_SUFFIX="${TAG_SUFFIX:-_e${EPOCHS}}"

# X-attention knobs (shared by xmain/xboth).  Override from the environment.
X_ARGS=(
  X_ATTN=1
  X_DIM="${X_DIM:-32}" X_HEADS="${X_HEADS:-4}"
  X_NEG="${X_NEG:-1}" X_MOD="${X_MOD:-1}" X_DROP="${X_DROP:-0.1}"
  X_NORM="${X_NORM:-dyt}" X_DIST="${X_DIST:-periodic}"
  X_LAMBDA="${X_LAMBDA:-0.2}" X_MU="${X_MU:-1.0}"
)
FSDH_ARGS=(
  FSDH_X_ATTN=1 FSDH_PERIOD="${FSDH_PERIOD:-24}"
  FSDH_PATCH_LEN="${FSDH_PATCH_LEN:-2}"
  FSDH_X_DIM="${FSDH_X_DIM:-32}" FSDH_X_HEADS="${FSDH_X_HEADS:-4}"
  FSDH_X_DROP="${FSDH_X_DROP:-}"
)

run_arm() {
  local seed="$1" arm="$2" pl="$3"
  local epochs="${4:-$EPOCHS}"
  local tag="${5:-${arm}_s${seed}${TAG_SUFFIX}}"
  local common=(MODELS=Times2D PRED_LENS="$pl" SEED="$seed"
                TAG="$tag"
                LR="$LR" LRADJ="$LRADJ" LOSS="$LOSS" TF32="$TF32"
                TRAIN_EPOCHS="$epochs" PATIENCE="$epochs")
  case "$arm" in
    xbase)
      env "${common[@]}" X_ATTN=0 FSDH_X_ATTN=0 \
          bash scripts/run_etth1.sh
      ;;
    xmain)
      env "${common[@]}" "${X_ARGS[@]}" FSDH_X_ATTN=0 \
          bash scripts/run_etth1.sh
      ;;
    xboth)
      env "${common[@]}" "${X_ARGS[@]}" "${FSDH_ARGS[@]}" \
          bash scripts/run_etth1.sh
      ;;
    *)
      echo "unknown arm: $arm" >&2
      exit 1
      ;;
  esac
}

case "$MODE" in
  smoke)
    echo "### X-attn smoke: 3 runs, pred_len=96, 1 epoch"
    for arm in xbase xmain xboth; do
      run_arm 2021 "$arm" 96 1 "${arm}_smoke"
    done
    ;;
  stage1)
    echo "### X-attn stage1: seed 2021, horizons=$HORIZONS, epochs=$EPOCHS"
    for arm in xbase xmain xboth; do
      for pl in $HORIZONS; do
        run_arm 2021 "$arm" "$pl"
      done
    done
    ;;
  stage2)
    echo "### X-attn stage2: seeds 2022 2023, horizons=$HORIZONS, epochs=$EPOCHS"
    for seed in 2022 2023; do
      for arm in xbase xmain xboth; do
        for pl in $HORIZONS; do
          run_arm "$seed" "$arm" "$pl"
        done
      done
    done
    ;;
  *)
    echo "usage: $0 {smoke|stage1|stage2}" >&2
    exit 1
    ;;
esac

echo
echo "Results land under results/ with tags xbase_s<seed>*, xmain_s<seed>*, xboth_s<seed>*."
echo "The X-attn arms are architecture changes; do not merge them into old tables without saying so."
