#!/bin/bash
# Standalone Times2D + FreDN on WEATHER, for the rho guard.
#
# This is the minimal test PROGRESS.md section 7 prescribes for option B
# ("change the dataset"): run the two branches at 3 seeds x 4 horizons, compute
# rho, and stop if it is still ~0.94. NO dual model and NO cross-attention --
# if rho does not drop, building DualFreDN-on-weather would be wasted work.
#
#   SMOKE=1 bash scripts/run_weather_standalone.sh   # 1 epoch of Times2D, for timing
#   bash scripts/run_weather_standalone.sh           # 24 runs
#
# Then:
#   python scripts/analyze_complementarity.py --model-a Times2D --model-b FreDN \
#       --tag 'fre_s20*' --data weather --noise-floor <MEASURED>
#
# ---------------------------------------------------------------------------
# WHAT IS TAKEN FROM WHERE
# ---------------------------------------------------------------------------
# ARCHITECTURE and per-dataset hyperparameters come from the official scripts;
# the PROTOCOL stays 2dphase's, unchanged from the ETTh1 rho runs, so that the
# only thing that differs between the two rho measurements is the DATASET.
#
# Times2D, from Times2D/scripts/times2D/longTerm/Times2D_weather.sh:
#   PATCH_LEN     48 32 16 6 3     (same as ETTh1)
#   PERIOD_LIST   720 360 140 70 48
#       NOT in the weather script, because upstream HARDCODES period_list in
#       models/Times2D.py and ignores --period_list. Upstream's line 78 reads
#           #self.period_list = [720, 360, 140, 70, 48]  # weather
#       with the M4/ETT value left active on line 79. So this is the authors'
#       weather configuration, recovered from a comment -- it is the only place
#       it appears anywhere in the repo.
#   --kernel_list 5 7 11 15 is in that script but is DELIBERATELY NOT passed:
#       it is a no-op. models/Times2D.py:192 unconditionally overwrites it with
#           self.kernel_list = [(n, self.patch_len[i]) for i, n in enumerate(self.period_len)]
#       and `configs.kernel_list` is never read anywhere in the model. Upstream
#       has the same line, so the published weather numbers were produced
#       without it mattering either. Passing it would only mislead.
#
# FreDN, from FreDN/scripts/weather.sh's param_map (dropout hidden_layers
# hidden_size; that script's lr/embed_size are NOT used, see below):
#   pl=96   0.3 1 128
#   pl=192  0.2 3 64
#   pl=336  0.2 3 16
#   pl=720  0.2 1 16
#
# DELIBERATELY NOT taken from the official scripts:
#   LR       FreDN's weather script uses 0.01, Times2D's uses 1e-4 + OneCycle.
#            Using each model's own lr would mean the two branches are no longer
#            trained the same way, and the ETTh1 rho was measured with a shared
#            lr=1e-3 constant. Keep it shared. If FreDN looks undertrained at
#            1e-3 on weather, that is worth knowing -- rerun with LR=0.01 and
#            TAG=lr1e2f_s${seed} rather than silently mixing.
#   LOSS     stays freqmae for both, as in the ETTh1 rho runs.
#   FRE_EMBED_SIZE  stays 32, not the paper's 8, so the weather FreDN is the
#            same shape as the one the ETTh1 rho was measured with.
#   EPOCHS/PATIENCE stay 100/10 (2dphase), not the official weather 50/5.
# ---------------------------------------------------------------------------
set -u

cd "$(dirname "$0")/.." || exit 1

declare -A FRE_DROPOUT=( [96]=0.3   [192]=0.2  [336]=0.2   [720]=0.2 )
declare -A FRE_HLAYERS=( [96]=1     [192]=3    [336]=3     [720]=1   )
declare -A FRE_HSIZE=(   [96]=128   [192]=64   [336]=16    [720]=16  )

# FREQ=min, NOT 't'. The TSL scripts (and utils/timefeatures.py's own help text)
# say 't' for minutely, but this env has pandas 3.0.5, which removed that alias
# -- `--freq t` dies in to_offset() before training starts. 'min' is the modern
# spelling and is what the string is for anyway; these models ignore x_mark, so
# this is about the loader parsing, not about signal.
DATASET="DATA=weather DATA_ROOT=./dataset/weather/ DATA_PATH=weather.csv \
         ENC_IN=21 FREQ=min SEQ_LEN=720 LOGDIR=logs/weather"
PROTOCOL="LR=0.001 LRADJ=none LOSS=freqmae TF32=1"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "######## SMOKE: 1 epoch of Times2D on weather, for timing ########"
  env MODELS=Times2D PRED_LENS=96 SEED=2021 TAG=wsmoke \
      TRAIN_EPOCHS=1 PATIENCE=1 \
      PATCH_LEN="48 32 16 6 3" \
      PERIOD_LIST="720 360 140 70 48" \
      $DATASET $PROTOCOL bash scripts/run_etth1.sh
  echo
  echo "Read the wall-clock out of logs/weather/Times2D_720_96_wsmoke.log:"
  echo "  grep -E 'cost time' logs/weather/Times2D_720_96_wsmoke.log"
  echo "Multiply by the expected epoch count to size up the 24-run job."
  exit 0
fi

# SKIP_DONE=1 resumes: a run whose log already contains a NUMERIC 'mse:' line
# finished. The pattern requires a digit after the colon on purpose -- a
# crashed run logs 'mse:nan', which a bare '^mse:' would match and then skip,
# silently reusing a broken result.
#
# Without SKIP_DONE every invocation retrains everything, which matters at ~19 h
# a go.
done_already() {  # done_already <logfile>
  [ "${SKIP_DONE:-0}" = "1" ] && [ -f "$1" ] && grep -qE '^mse:[0-9-]' "$1"
}

for seed in 2021 2022 2023; do
  for pl in 96 192 336 720; do
    lt="logs/weather/Times2D_720_${pl}_fre_s${seed}.log"
    lf="logs/weather/FreDN_720_${pl}_fre_s${seed}.log"

    echo
    echo "######## weather  Times2D  seed=$seed  pred_len=$pl ########"
    if done_already "$lt"; then
      echo "  skip (already has a result)"
    else
      env MODELS=Times2D PRED_LENS="$pl" SEED="$seed" TAG="fre_s${seed}" \
          PATCH_LEN="48 32 16 6 3" \
          PERIOD_LIST="720 360 140 70 48" \
          $DATASET $PROTOCOL bash scripts/run_etth1.sh
    fi

    echo
    echo "######## weather  FreDN  seed=$seed  pred_len=$pl ########"
    if done_already "$lf"; then
      echo "  skip (already has a result)"
    else
      env MODELS=FreDN PRED_LENS="$pl" SEED="$seed" TAG="fre_s${seed}" \
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
    --tag 'fre_s20*' --data weather --noise-floor <MEASURED>

The noise floor MUST be measured, not assumed: the 0.011 default is ETTh1-720's.
Weather's is its own number -- get it from the spread of the three seeds you just
ran (each horizon's min..max in the MSE_A column), or from running one model at
2+ seeds and taking the spread. Passing the ETTh1 value here would make the
"is the fusion gain above noise" line wrong.

VERDICT RULE (PROGRESS.md section 7, unchanged):
  rho still ~0.94  -> option B is dead too. Each branch is near-optimal on the
                      same data, so their errors stay dominated by shared noise.
  rho clearly lower AND the oracle gain clears the new noise floor
                   -> the branches really are complementary there, and a dual
                      model is worth building on weather.
  rho clearly lower BUT the gain still does not clear the floor
                   -> they differ, but not usefully. Report and stop.
EOF
