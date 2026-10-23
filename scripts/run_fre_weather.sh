#!/bin/bash
# DualFreDN (Times2D + FreDN + bidirectional cross-attention) on WEATHER.
#
# Same shape as scripts/run_fre_experiment.sh, with the dataset swapped and the
# per-dataset hyperparameters taken from the official scripts (below).
#
#   SMOKE=1 bash scripts/run_fre_weather.sh     # 1 epoch, for timing
#   SKIP_DONE=1 bash scripts/run_fre_weather.sh # resume; skips finished logs
#   bash scripts/run_fre_weather.sh             # 24 runs
#
# MEASURED COST: ~0.82 s/batch forward+backward at enc_in=21, batch=128, seq=720
# -> 281 steps/epoch -> ~250 s/epoch. Weather early-stops around epoch 17-20, so
# roughly 85 min per run, and the 24-run ablation is ~34 HOURS. That is close to
# double the standalone rho guard (12 runs per model, ~19 h), because this is 24
# runs of an 8.8-12.0M-parameter model rather than 12 runs of a 3.7M one.
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE INTERPRETING A NULL RESULT
# ---------------------------------------------------------------------------
# This script does NOT run the rho guard. So if ca1 ~= ca0 on weather, the
# result is NOT self-explanatory: it means "cross-attention changed nothing
# here", and it does NOT distinguish
#
#   (a) the two branches are still near-duplicates on weather, so there is
#       nothing to fuse -- the same death as ETTh1, and changing the dataset
#       does not rescue the approach; from
#   (b) the branches finally differ on weather and cross-attention STILL does
#       nothing -- a strictly stronger claim, that the mechanism itself is the
#       problem.
#
# ETTh1 could not separate those either (rho = 0.9602, PROGRESS.md section 9.4).
# scripts/run_weather_standalone.sh is what separates them, at ~19 h. If you
# want both answers from one overnight run, chain them:
#
#   bash scripts/run_weather_standalone.sh && bash scripts/run_fre_weather.sh
# ---------------------------------------------------------------------------
#
# Dataset values and where they come from: see the header of
# scripts/run_weather_standalone.sh -- identical here. In short:
#   PERIOD_LIST   720 360 140 70 48  = upstream's weather value, which exists
#                 only as a comment at Times2D/models/Times2D.py:78
#   PATCH_LEN     48 32 16 6 3       = the official weather script
#   FRE_*         per-horizon dropout / hidden_layers / hidden_size from
#                 FreDN/scripts/weather.sh's param_map
#   --kernel_list is NOT passed      = it is a no-op (Times2D.py:192 overwrites it)
#   FREQ=min, not 't'                = pandas 3.0.5 removed the 't' alias
#
# Protocol (LR / LRADJ / LOSS / TF32 / FRE_EMBED_SIZE) is deliberately the SAME
# as the ETTh1 runs, so the only thing that differs is the dataset.
set -u

cd "$(dirname "$0")/.." || exit 1

declare -A FRE_DROPOUT=( [96]=0.3   [192]=0.2  [336]=0.2   [720]=0.2 )
declare -A FRE_HLAYERS=( [96]=1     [192]=3    [336]=3     [720]=1   )
declare -A FRE_HSIZE=(   [96]=128   [192]=64   [336]=16    [720]=16  )

DATASET="DATA=weather DATA_ROOT=./dataset/weather/ DATA_PATH=weather.csv \
         ENC_IN=21 FREQ=min SEQ_LEN=720 LOGDIR=logs/weather"
PROTOCOL="LR=0.001 LRADJ=none LOSS=freqmae TF32=1"

# SKIP_DONE=1 resumes: a run whose log already contains a NUMERIC 'mse:' line
# finished. The pattern requires a digit after the colon on purpose -- a
# crashed run logs 'mse:nan', which a bare '^mse:' would match and then skip,
# silently reusing a broken result.
#
# Without SKIP_DONE every invocation retrains everything, which matters at ~34 h
# a go.
done_already() {  # done_already <logfile>
  [ "${SKIP_DONE:-0}" = "1" ] && [ -f "$1" ] && grep -qE '^mse:[0-9-]' "$1"
}

if [ "${SMOKE:-0}" = "1" ]; then
  echo "######## SMOKE: 1 epoch of DualFreDN on weather (ca0), for timing ########"
  env MODELS=DualFreDN PRED_LENS=96 SEED=2021 TAG=wca0_smoke \
      TRAIN_EPOCHS=1 PATIENCE=1 \
      PATCH_LEN="48 32 16 6 3" PERIOD_LIST="720 360 140 70 48" \
      FRE_EMBED_SIZE=32 FRE_DROPOUT=0.3 FRE_HIDDEN_LAYERS=1 FRE_HIDDEN_SIZE=128 \
      $DATASET $PROTOCOL bash scripts/run_etth1.sh
  echo
  echo "  grep -E 'cost time' logs/weather/DualFreDN_720_96_wca0_smoke.log"
  exit 0
fi

run_arm() {  # run_arm <seed> <ca 0|1>
  local seed="$1" ca="$2"
  local tag="ca${ca}_s${seed}"
  local extra=""
  [ "$ca" = "1" ] && extra="CROSS_ATTN=1 CA_LAYERS=1 CA_HEADS=4"
  echo
  echo "######## weather  seed=$seed  cross_attn=$ca  (tag=$tag) ########"
  for pl in 96 192 336 720; do
    local log="logs/weather/DualFreDN_720_${pl}_${tag}.log"
    if done_already "$log"; then
      echo "  skip pl=$pl (already has a result)"
      continue
    fi
    env MODELS=DualFreDN PRED_LENS="$pl" SEED="$seed" TAG="$tag" \
        PATCH_LEN="48 32 16 6 3" PERIOD_LIST="720 360 140 70 48" \
        FRE_EMBED_SIZE=32 \
        FRE_DROPOUT="${FRE_DROPOUT[$pl]}" \
        FRE_HIDDEN_LAYERS="${FRE_HLAYERS[$pl]}" \
        FRE_HIDDEN_SIZE="${FRE_HSIZE[$pl]}" \
        $DATASET $PROTOCOL $extra bash scripts/run_etth1.sh
  done
}

for s in 2021 2022 2023; do
  run_arm "$s" 0
  run_arm "$s" 1
done

cat <<'EOF'

==================== HOW TO READ THIS ====================
  python build_results_table.py --tagged
  python scripts/report_alpha_beta.py --glob '*DualFreDN*'

DECISION RULES -- unchanged from the ETTh1 run (PROGRESS.md 9.3), because the
experiment is the same experiment on different data:

  ca1 beats ca0 by > the WEATHER-720 seed floor AND consistently
      -> cross-attention works once the branches are trained on richer data.
  ca1 ~= ca0
      -> see the header. You cannot attribute this without the rho guard.
  ca1 worse than ca0
      -> matches v1 and the ETTh1 run.

The noise floor for weather is NOT 0.011 -- that is the ETTh1-720 measurement.
Measure it from the seed spread of whatever runs you have, and do not judge a
delta against the ETTh1 number.
EOF
