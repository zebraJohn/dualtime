#!/bin/bash
# Standalone Times2D + FreDN on HOURLY-RESAMPLED weather, for the rho guard.
#
# The cheap screen: the same test as scripts/run_weather_standalone.sh, on a
# 6x smaller series. ~3-4 h instead of ~19 h, for the same 24 runs.
#
#   SKIP_DONE=1 bash scripts/run_weather1h_standalone.sh
#   SMOKE=1     bash scripts/run_weather1h_standalone.sh   # 1 epoch, timing
#
# ---------------------------------------------------------------------------
# THIS IS NOT THE WEATHER BENCHMARK. READ BEFORE INTERPRETING.
# ---------------------------------------------------------------------------
# The 10-minute file is resampled to hourly AT LOAD TIME (see
# data_provider/downsample.py). seq_len and pred_len are NOT rescaled, so the
# numbers mean something different here:
#
#                     10-minute data        1-hour data (this script)
#     pred_len=96      next 16 hours         next 4 DAYS
#     pred_len=720     next 5 days           next 30 days
#     seq_len=720      5 days of context     30 days of context
#
# So the rho that comes out is "rho on hourly-subsampled weather at these
# horizons", and it is not comparable to any published weather number.
#
# It is a SCREEN, and like any screen its negative is weaker than its positive:
#   rho clearly LOWER than ETTh1's 0.9602 -> promising; confirm on the 10-minute
#       series before building anything.
#   rho still ~0.96 -> suggestive but AMBIGUOUS: you cannot tell whether the
#       branches are homogeneous on weather too, or whether the 6x coarser task
#       is what kept them together.
#
# WHY HOURLY AND NOT 7-CHANNEL
# Both cost about the same. Hourly changes the TASK (horizons move 6x);
# channel-subsetting would keep the task identical and only shrink the number of
# independent per-channel problems. Hourly was chosen for the larger cost
# saving; if the rho comes out low, the 10-minute run is the confirmation step.
#
# WHY resample().first() AND NOT iloc[::6]
# The source file has a duplicated timestamp and a 90-minute hole; index-based
# slicing drifts at both, time-based binning does not. And .first() rather than
# .mean() keeps the noise level, so rho stays comparable to ETTh1's. Details in
# data_provider/downsample.py's module docstring.
#
# WHY --data weather1h AND NOT weather
# --data lands in the experiment setting name as `ft{data}`. Reusing 'weather'
# would give these runs the SAME setting name as the 10-minute ones and silently
# overwrite their checkpoints and results.
set -u

cd "$(dirname "$0")/.." || exit 1

declare -A FRE_DROPOUT=( [96]=0.3   [192]=0.2  [336]=0.2   [720]=0.2 )
declare -A FRE_HLAYERS=( [96]=1     [192]=3    [336]=3     [720]=1   )
declare -A FRE_HSIZE=(   [96]=128   [192]=64   [336]=16    [720]=16  )

# Same 10-minute file; the resampling happens in the loader.
# FREQ=h because the resampled grid IS hourly. PERIOD_LIST is left EMPTY so
# Times2D uses its own DEFAULT_PERIOD_LIST -- at hourly resolution the timescale
# matches ETTh1, which is what that list was chosen for. Passing upstream's
# weather list ([720,360,140,70,48], a 10-minute value) here would be wrong.
DATASET="DATA=weather1h DATA_ROOT=./dataset/weather/ DATA_PATH=weather.csv \
         ENC_IN=21 FREQ=h SEQ_LEN=720 LOGDIR=logs/weather1h"
PROTOCOL="LR=0.001 LRADJ=none LOSS=freqmae TF32=1"

done_already() {  # done_already <logfile>
  [ "${SKIP_DONE:-0}" = "1" ] && [ -f "$1" ] && grep -qE '^mse:[0-9-]' "$1"
}

if [ "${SMOKE:-0}" = "1" ]; then
  echo "######## SMOKE: 1 epoch of Times2D on HOURLY weather, for timing ########"
  env MODELS=Times2D PRED_LENS=96 SEED=2021 TAG=w1h_smoke \
      TRAIN_EPOCHS=1 PATIENCE=1 \
      PATCH_LEN="48 32 16 6 3" \
      $DATASET $PROTOCOL bash scripts/run_etth1.sh
  echo
  echo "  grep -E 'cost time' logs/weather1h/Times2D_720_96_w1h_smoke.log"
  exit 0
fi

for seed in 2021 2022 2023; do
  for pl in 96 192 336 720; do
    lt="logs/weather1h/Times2D_720_${pl}_w1h_fre_s${seed}.log"
    lf="logs/weather1h/FreDN_720_${pl}_w1h_fre_s${seed}.log"

    echo
    echo "######## weather1h  Times2D  seed=$seed  pred_len=$pl ########"
    if done_already "$lt"; then
      echo "  skip (already has a result)"
    else
      env MODELS=Times2D PRED_LENS="$pl" SEED="$seed" TAG="w1h_fre_s${seed}" \
          PATCH_LEN="48 32 16 6 3" \
          $DATASET $PROTOCOL bash scripts/run_etth1.sh
    fi

    echo
    echo "######## weather1h  FreDN  seed=$seed  pred_len=$pl ########"
    if done_already "$lf"; then
      echo "  skip (already has a result)"
    else
      env MODELS=FreDN PRED_LENS="$pl" SEED="$seed" TAG="w1h_fre_s${seed}" \
          FRE_EMBED_SIZE=32 \
          FRE_DROPOUT="${FRE_DROPOUT[$pl]}" \
          FRE_HIDDEN_LAYERS="${FRE_HLAYERS[$pl]}" \
          FRE_HIDDEN_SIZE="${FRE_HSIZE[$pl]}" \
          $DATASET $PROTOCOL bash scripts/run_etth1.sh
    fi
  done
done

cat <<'EOF'

==================== NEXT ====================
python scripts/analyze_complementarity.py --model-a Times2D --model-b FreDN \
    --tag 'w1h_fre_s20*' --data weather1h

Compare against the ETTh1 measurement: rho = 0.9602 (mean), 0.9342..0.9810 per
seed at pred=96. If weather1h lands inside that band, the screen says nothing
new. Only a clear move below it is worth acting on -- and even then, confirm on
the 10-minute series, because the horizons here are 6x longer.
EOF
