#!/bin/bash
# Standalone CrossChannelFreq on weather, to be paired against the EXISTING
# Times2D weather runs. This is the gate for the cross-channel hypothesis.
#
#   SMOKE=1     bash scripts/run_crosschannel_standalone.sh   # 1 epoch, timing
#   SKIP_DONE=1 bash scripts/run_crosschannel_standalone.sh   # 12 runs, resumable
#
# ---------------------------------------------------------------------------
# WHAT THIS MEASURES, AND WHY IT IS THE LAST UNTRIED THING
# ---------------------------------------------------------------------------
# Three attempts to make the two branches differ have failed:
#   change the branch   -> rho 0.9602, MORE homogeneous than PhaseFormer's 0.9377
#   change the dataset  -> rho 0.9593 on weather vs 0.9602 on ETTh1, i.e. nothing
#   cross-attention     -> 5/12, mean -0.0020, PROGRESS.md 4.5
#
# Every model tried -- Times2D, PhaseFormer, FreDN -- is PER-CHANNEL: the channel
# axis is a batch axis and no information crosses between channels. Measured:
# FreDN's output for channel 1 changes by exactly 0.0 when channel 0's input is
# perturbed (tests/test_cross_channel.py, INV3-control).
#
# models/CrossChannelFreq.py is the cheapest model that does NOT have that
# property. The same test measures a 1.7e-3 effect on other channels against a
# 4.8e-2 output scale, i.e. ~3.5%.
#
# This script measures rho(Times2D, CrossChannelFreq) on weather. Compare against
# rho(Times2D, FreDN) = 0.9593 on the same data.
#
# ---------------------------------------------------------------------------
# HOW TO READ THE RESULT -- the gate is NOT rho
# ---------------------------------------------------------------------------
# The best pair this project ever measured was rho = 0.9377 with an oracle gain
# of +0.0070, against an ETTh1-720 noise floor of 0.011 -- and that did NOT clear
# the floor. So "rho went down a bit" is not success. Per the trend in the
# measured points, the gain only becomes usable somewhere around rho < 0.8.
#
#   rho stays ~0.96  -> cross-channel mixing, reduced to one axis change, does
#                       not break the homogeneity. The port of full FreEformer is
#                       NOT worth doing on this evidence; report and stop.
#   rho drops hard AND the oracle gain clears the MEASURED floor
#                    -> the per-channel assumption was the cause. Port FreEformer
#                       properly (23 config fields, 4 layer classes) and confirm.
#   rho drops BUT the gain still misses the floor
#                    -> they differ, not usefully. Report and stop.
#
# CAVEAT, stated up front: this is NOT FreEformer. FreEformer additionally changes
# the embedding, depth, width, attention mechanism and training recipe. A null
# here excludes THIS cross-channel mixer, not cross-channel modelling in general.
# That is the price of isolating one variable.
#
# COST, MEASURED: 21.4 s/epoch at seq_len=720, enc_in=21, batch=128 -- against
# 209 s for Times2D and 86 s for FreDN on the same data. At ~20 epochs early
# stopping that is ~7 min/run, so the 12 runs here are about 1.5 HOURS. No
# Times2D re-run is needed (its weather runs already carry the matching tag), so
# this is the whole cost of the gate. That is what makes it worth running before
# the ~4-hour FreEformer port and its much heavier GPU bill (d_model=512).
#
# ---------------------------------------------------------------------------
# WHY THE TAG IS 'fre_s${seed}' AND NOT SOMETHING NEW
# ---------------------------------------------------------------------------
# analyze_complementarity.py globs `results/..._{model}_*_Exp_{tag}`, so BOTH
# models must carry the SAME tag. Reusing the tag the Times2D weather runs
# already have means no Times2D re-run is needed. There is no collision: the
# model name is part of the setting name, so
#   ..._Times2D_..._Exp_fre_s2021        and
#   ..._CrossChannelFreq_..._Exp_fre_s2021
# are different directories. The LOGDIR is separate for the same reason.
set -u

cd "$(dirname "$0")/.." || exit 1

DATASET="DATA=weather DATA_ROOT=./dataset/weather/ DATA_PATH=weather.csv \
         ENC_IN=21 FREQ=min SEQ_LEN=720 LOGDIR=logs/ccfreq"
PROTOCOL="LR=0.001 LRADJ=none LOSS=freqmae TF32=1"

done_already() {  # done_already <logfile>
  [ "${SKIP_DONE:-0}" = "1" ] && [ -f "$1" ] && grep -qE '^mse:[0-9-]' "$1"
}

if [ "${SMOKE:-0}" = "1" ]; then
  echo "######## SMOKE: 1 epoch of CrossChannelFreq on weather, for timing ########"
  env MODELS=CrossChannelFreq PRED_LENS=96 SEED=2021 TAG=cc_smoke \
      TRAIN_EPOCHS=1 PATIENCE=1 \
      PATCH_LEN="48 32 16 6 3" PERIOD_LIST="720 360 140 70 48" \
      $DATASET $PROTOCOL bash scripts/run_etth1.sh
  echo
  echo "  grep -E 'cost time' logs/ccfreq/CrossChannelFreq_720_96_cc_smoke.log"
  exit 0
fi

for seed in 2021 2022 2023; do
  for pl in 96 192 336 720; do
    log="logs/ccfreq/CrossChannelFreq_720_${pl}_fre_s${seed}.log"
    echo
    echo "######## CrossChannelFreq  seed=$seed  pred_len=$pl ########"
    if done_already "$log"; then
      echo "  skip (already has a result)"
    else
      env MODELS=CrossChannelFreq PRED_LENS="$pl" SEED="$seed" TAG="fre_s${seed}" \
          PATCH_LEN="48 32 16 6 3" PERIOD_LIST="720 360 140 70 48" \
          $DATASET $PROTOCOL bash scripts/run_etth1.sh
    fi
  done
done

cat <<'EOF'

==================== NEXT ====================
python scripts/analyze_complementarity.py --model-a Times2D --model-b CrossChannelFreq \
    --tag 'fre_s20*' --data weather

Reference points, all measured in this repo:
  rho(Times2D, FreDN) on weather      = 0.9593   <- the thing to beat
  rho(Times2D, FreDN) on ETTh1        = 0.9602
  best pair ever measured (PhaseFormer, ETTh1) = 0.9377, oracle gain +0.0070,
      against a floor of 0.011 -- STILL not usable.

So measure the weather noise floor from the Times2D seed spread in the MSE_A
column before judging the gain, and remember the gate is the GAIN clearing the
floor, not rho moving.
EOF
