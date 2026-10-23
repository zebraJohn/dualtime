#!/bin/bash
# ETTh1 experiments for Dual-Times2D phase 1.
#
# Config is copied verbatim from Times2D's official ETTh1 script so results are
# directly comparable to the existing Times2D baseline (which is NOT re-run):
#   seq_len=720  batch_size=128  seed=2021  lr=1e-4  lradj=TST
#   train_epochs=100  patience=10  d_model=64  n_heads=16  e_layers=3
#
# Do NOT change batch_size: data_factory sets drop_last=True, so the reported
# MSE covers 2560/2668 test samples and a different batch size would silently
# evaluate on a different subset.
#
# Everything below can be overridden from the environment. Defaults reproduce
# the original behaviour exactly.
#
#   MODELS      models to run            (default: "PhaseTimes2D DualTimes2D PhaseFormer")
#   PRED_LENS   horizons                 (default: "96 192 336 720")
#   SEED        random seed              (default: 2021)
#   TAG         --run_tag; ALWAYS set this when varying SEED or any knob, or the
#               run will overwrite the matching untagged run's checkpoint/results
#   LR          --learning_rate          (default: 0.0001)
#               NOT part of the setting name, so a non-default LR REQUIRES a
#               TAG or the run overwrites the default-LR checkpoint/results.
#               The script refuses to start without one.
#
# Protocol knobs, same TAG rule as LR (phase2d-v2.md section 7.3):
#   LOSS        --loss                   (default: mse;  'huber' for the ref)
#   HUBER_DELTA --huber_delta            (default: 1.0; inert unless LOSS=huber)
#   LRADJ       --lradj                  (default: TST = OneCycleLR;
#                                         'none' = genuinely constant lr)
#   TF32        --tf32                   (default: 0;  1 = allow TF32 matmuls)
#
# The PhaseFormer reference protocol is exactly:
#   LR=0.001 LRADJ=none LOSS=huber HUBER_DELTA=1.0 TF32=1
# scripts/run_phaseformer_protocol.sh runs it.
#   WD          --weight_decay           (default: 0.0)
#   FUSION_DIM  --fusion_dim             (default: 0 = raw concat head)
#   CKPT_AVG_K  --ckpt_avg_k             (default: 1 = single val-best ckpt)
#   NO_FSDH     1 to disable FSDH        (default: 0)
#   CROSS_ATTN  1 to enable bidirectional cross-attention (stage 2)
#   CA_LAYERS   --ca_layers              (default: 1)
#   CA_HEADS    --ca_heads, must divide d_model (default: 4)
#
# PhaseFormer knobs (the Times2D-family models ignore all of these, and
# PhaseFormer ignores --d_model/--n_heads/--e_layers/--patch_len/--top_k above).
#
# Defaults below are the ETTh1-720 config from PhaseFormer-test, i.e. the run
# behind the 0.4342 mean this repo is comparing against. Upstream's own TSL
# script instead uses latent_dim=8, heads=4, encoder_hidden=32,
# predictor_hidden=64, attn_dropout=0.1 -- set these vars to those values to
# reproduce the published script.
#
#   PF_PERIOD_LEN  --period_len            (default: 24 = 1 day at hourly freq)
#   PF_LATENT_DIM  --latent_dim            (default: 6)
#   PF_LAYERS      --phase_layers          (default: 1)
#   PF_HEADS       --phase_attn_heads, MUST DIVIDE PF_LATENT_DIM (default: 1)
#   PF_ENC_HIDDEN  --phase_encoder_hidden  (default: 16)
#   PF_PRED_HIDDEN --predictor_hidden      (default: 32)
#   PF_ATTN_DROP   --phase_attn_dropout    (default: 0.0)
#
# PF_LATENT_DIM must be divisible by PF_HEADS -- CrossPhaseRoutingLayer asserts
# it. The defaults (6, 1) are NOT the upstream pair (8, 4), so raising PF_HEADS
# on its own aborts the run; the script checks this up front.
#
# Examples:
#   # everything: 3 models x 4 horizons = 12 runs, ~1.7 h
#   bash scripts/run_etth1.sh
#
#   # the v1 headline table only (Times2D itself is reused from the official
#   # numbers, never re-run -- see build_results_table.py's BASELINE)
#   MODELS="PhaseTimes2D DualTimes2D" bash scripts/run_etth1.sh
#
#   # seed sweep at 720 (see scripts/sweep_720.sh)
#   SEED=2022 TAG=s2022 PRED_LENS=720 MODELS=DualTimes2D bash scripts/run_etth1.sh
#
#   # regularised dual at 720
#   TAG=wd01 PRED_LENS=720 MODELS=DualTimes2D WD=0.01 bash scripts/run_etth1.sh
#
#   # PhaseFormer baseline at all four horizons (Step 1 of phase2d-v2.md)
#   MODELS=PhaseFormer TAG=pf bash scripts/run_etth1.sh
#
#   # the PhaseFormer reference protocol (constant lr + Huber + TF32)
#   LR=0.001 LRADJ=none LOSS=huber TF32=1 TAG=refproto \
#     MODELS=PhaseFormer bash scripts/run_etth1.sh
#
#   # the two Step 1 driver scripts
#   bash scripts/run_phaseformer_protocol.sh  # unified vs lr-only vs reference
#   bash scripts/run_latent_dim_sweep.sh      # latent_dim 6 vs 32 vs 64
set -u

PY="${PY:-D:/Anaconda/envs/times2d/python.exe}"
cd "$(dirname "$0")/.." || exit 1

MODELS=${MODELS:-"PhaseTimes2D DualTimes2D PhaseFormer"}
PRED_LENS=${PRED_LENS:-"96 192 336 720"}
SEED=${SEED:-2021}
TAG=${TAG:-}
WD=${WD:-0.0}
LR=${LR:-0.0001}
# Protocol knobs (phase2d-v2.md section 7.3). Defaults = original Times2D
# behaviour. The PhaseFormer reference protocol is LR=0.001 LRADJ=none
# LOSS=huber TF32=1; see scripts/run_phaseformer_protocol.sh.
LOSS=${LOSS:-mse}
HUBER_DELTA=${HUBER_DELTA:-1.0}
LRADJ=${LRADJ:-TST}
TF32=${TF32:-0}
FUSION_DIM=${FUSION_DIM:-0}
CKPT_AVG_K=${CKPT_AVG_K:-1}
NO_FSDH=${NO_FSDH:-0}
CROSS_ATTN=${CROSS_ATTN:-0}
CA_LAYERS=${CA_LAYERS:-1}
CA_HEADS=${CA_HEADS:-4}
# only for smoke-testing the wiring; leave at the defaults for real runs
TRAIN_EPOCHS=${TRAIN_EPOCHS:-100}
PATIENCE=${PATIENCE:-10}

# --- Times2D X-shaped attention (PHAT-style PNA) -----------------------------
# Off by default; see run.py --x_attn / --fsdh_x_attn.  These bind to the
# Times2D-family models only.  X_DIST is periodic (PHAT) or linear.
X_ATTN=${X_ATTN:-0}
X_DIM=${X_DIM:-32}
X_HEADS=${X_HEADS:-4}
X_NEG=${X_NEG:-1}
X_MOD=${X_MOD:-1}
X_DROP=${X_DROP:-0.1}
X_NORM=${X_NORM:-dyt}
X_DIST=${X_DIST:-periodic}
X_LAMBDA=${X_LAMBDA:-0.2}
X_MU=${X_MU:-1.0}
FSDH_X_ATTN=${FSDH_X_ATTN:-0}
FSDH_PERIOD=${FSDH_PERIOD:-24}
FSDH_PATCH_LEN=${FSDH_PATCH_LEN:-2}
FSDH_X_DIM=${FSDH_X_DIM:-32}
FSDH_X_HEADS=${FSDH_X_HEADS:-4}
FSDH_X_DROP=${FSDH_X_DROP:-}

# --- PhaseFormer knobs (see the header for what these map to) ---
PF_PERIOD_LEN=${PF_PERIOD_LEN:-24}
PF_LATENT_DIM=${PF_LATENT_DIM:-6}
PF_LAYERS=${PF_LAYERS:-1}
PF_HEADS=${PF_HEADS:-1}
PF_ENC_HIDDEN=${PF_ENC_HIDDEN:-16}
PF_PRED_HIDDEN=${PF_PRED_HIDDEN:-32}
PF_ATTN_DROP=${PF_ATTN_DROP:-0.0}

# --- FreDN knobs (FreDN and DualFreDN; the other models ignore them) ---
# fre_embed_size is the CA width when FreDN is the frequency branch of
# DualFreDN: the branch enters cross-attention through
# FeatureAdapter(2*fre_embed_size, d_model). It must satisfy
# 2*fre_embed_size == d_model, or the attention's queries and keys live in a
# low-rank subspace and a null result means nothing
# (scripts/run_ca_experiment.sh:16-22). INV6 in
# tests/test_dual_fredn_invariants.py pins this.
FRE_EMBED_SIZE=${FRE_EMBED_SIZE:-32}
FRE_HIDDEN_SIZE=${FRE_HIDDEN_SIZE:-64}
FRE_HIDDEN_LAYERS=${FRE_HIDDEN_LAYERS:-1}
FRE_DROPOUT=${FRE_DROPOUT:-0.1}

# --- FreEformer knobs (FreEformer and DualFreEformer; others ignore them) ---
# NOTE THE PREFIX: FE_ is the FreEformer branch, FRE_ is the FreDN branch. They
# are unrelated -- FE_EMBED_SIZE is the upstream FreEformer embedding width
# (16), FRE_EMBED_SIZE is FreDN's latent (32, and it must satisfy
# 2*fre_embed_size == d_model for DualFreDN).
#
# Every one of these is passed explicitly on the command line below. The
# defaults here exist only so the needs_tag guards have something honest to
# compare against -- the values actually used are always visible in the
# invocation.
#
# The three attention switches are FIXED at upstream's own attn_ablation
# settings and are NOT swept (spec 3.4): upstream's Weather script walks a 2x2x2
# grid over them, and importing that grid would change the model and a set of
# free hyper-parameters at the same time.
FE_EMBED_SIZE=${FE_EMBED_SIZE:-16}

# --- per-branch overrides for the FreEformer branch (v2) --------------------
# Empty = fall back to the shared field, i.e. v1's behaviour EXACTLY. A `uw`
# run sets all five to FreEformer's native architecture:
#   FE_D_MODEL=512 FE_D_FF=512 FE_N_HEADS=8 FE_E_LAYERS=2 FE_DROPOUT=0.2
# The five are architecture, not training recipe; lr/loss/batch_size/epochs stay
# shared because a dual has one optimizer and one output (spec 3).
#
# NOTE the collision this avoids: `--fre_dropout` was already taken by the FreDN
# branch, so these are `fe_*` (argparse) / `FE_*` (shell), matching the FE_/FRE_
# convention documented above.
FE_D_MODEL=${FE_D_MODEL:-}
FE_D_FF=${FE_D_FF:-}
FE_N_HEADS=${FE_N_HEADS:-}
FE_E_LAYERS=${FE_E_LAYERS:-}
FE_DROPOUT=${FE_DROPOUT:-}

FE_ATTN_ENHANCE=${FE_ATTN_ENHANCE:-1}
FE_ATTN_SOFTMAX=${FE_ATTN_SOFTMAX:-0}
FE_ATTN_WEIGHT_PLUS=${FE_ATTN_WEIGHT_PLUS:-1}
FE_ATTN_OUTSIDE_SOFTMAX=${FE_ATTN_OUTSIDE_SOFTMAX:-1}
# These three MUST stay 0. Upstream's plot_mat guard is `random.random() < 1`
# (always true) and its plot_grad guard is not wrapped in `not self.training`,
# so either would draw from the global RNG on every eval or every step and
# break seed reproducibility; CKA_flag calls random.uniform inside forward.
# The port raises instead of plotting (spec 9 R2).
FE_PLOT_MAT=${FE_PLOT_MAT:-0}
FE_PLOT_GRAD=${FE_PLOT_GRAD:-0}
FE_CKA=${FE_CKA:-0}

# --- Dataset knobs -------------------------------------------------------
# Defaults reproduce the ETTh1 behaviour exactly. Overriding any of the data
# ones requires a TAG for the same reason LR does: they are not all in the
# experiment setting name (only seq_len is), so a change would otherwise
# overwrite another dataset's checkpoint and results.
#
#   DATA / DATA_ROOT / DATA_PATH  feed --data / --root_path / --data_path.
#                                 DATA must be a key of data_provider's
#                                 data_dict -- note 'weather', NOT 'custom'
#                                 (this repo has no 'custom' key, unlike the
#                                 upstream FreDN scripts).
#   ENC_IN                        MUST equal the number of non-date columns in
#                                 DATA_PATH. weather=21, ETTh1=7.
#   SEQ_LEN                       lookback.
#   FREQ                          time-feature encoding frequency. These models
#                                 ignore x_mark, so this is hygiene, not signal.
#   PERIOD_LIST                   Times2D's period folding. EMPTY = leave it to
#                                 models/Times2D.py's DEFAULT_PERIOD_LIST, which
#                                 is tuned for ETTh1 at seq_len=720 and is NOT
#                                 transferable across sampling rates.
#   PATCH_LEN                     one entry per period; Times2DBackbone asserts
#                                 len(patch_len) >= len(period_list).
DATA=${DATA:-ETTh1}
DATA_ROOT=${DATA_ROOT:-./dataset/ETT-small/}
DATA_PATH=${DATA_PATH:-ETTh1.csv}
ENC_IN=${ENC_IN:-7}
SEQ_LEN=${SEQ_LEN:-720}
FREQ=${FREQ:-h}
PERIOD_LIST=${PERIOD_LIST:-}
PATCH_LEN=${PATCH_LEN:-"48 32 16 6 3"}
LOGDIR=${LOGDIR:-logs/Etth1_dual}

# --- CrossChannelFreq knobs (that model only) ---
CC_EMBED_SIZE=${CC_EMBED_SIZE:-8}
CC_HIDDEN=${CC_HIDDEN:-32}
CC_HEAD_HIDDEN=${CC_HEAD_HIDDEN:-256}
CC_DROPOUT=${CC_DROPOUT:-0.1}

# CrossPhaseRoutingLayer asserts attention_dim % num_heads == 0, and it does so
# inside the first forward pass -- so without this check a bad pair burns the
# data-loading and model-build time of every run in the loop before failing.
if [ $((PF_LATENT_DIM % PF_HEADS)) -ne 0 ]; then
  echo "ERROR: PF_LATENT_DIM=$PF_LATENT_DIM is not divisible by PF_HEADS=$PF_HEADS" >&2
  printf '       valid PF_HEADS for latent_dim=%s:' "$PF_LATENT_DIM" >&2
  for h in $(seq 1 "$PF_LATENT_DIM"); do
    [ $((PF_LATENT_DIM % h)) -eq 0 ] && printf ' %s' "$h" >&2
  done
  echo >&2
  exit 1
fi

# None of LR / LOSS / LRADJ / TF32 appear in the experiment setting name, so two
# runs differing only in one of them resolve to the same checkpoint/results path
# and silently overwrite each other. Require an explicit TAG whenever any of
# them leaves its default. This bites hardest on PhaseFormer: its reference
# protocol is constant lr=1e-3 + Huber + TF32, while the Times2D family runs
# lr=1e-4 + MSE + no TF32 (phase2d-v2.md section 7.3).
#
# HUBER_DELTA needs no guard of its own: it is inert unless LOSS=huber, and that
# already forces a TAG.
needs_tag() {  # needs_tag <name> <value> <default>
  if [ "$2" != "$3" ] && [ -z "$TAG" ]; then
    echo "ERROR: $1=$2 differs from the default $3 but TAG is empty." >&2
    echo "       $1 is not part of the experiment setting name, so this run" >&2
    echo "       would overwrite the default run's checkpoint and results." >&2
    echo "       Re-run with a TAG, e.g.:" >&2
    echo "         $1=$2 TAG=refproto MODELS=PhaseFormer bash $0" >&2
    exit 1
  fi
}
needs_tag LR "$LR" "0.0001"
needs_tag LOSS "$LOSS" "mse"
needs_tag LRADJ "$LRADJ" "TST"
needs_tag TF32 "$TF32" "0"
# All four FreDN knobs, not just fre_embed_size: none of them appears in the
# experiment setting name, so any of them changing without a TAG overwrites
# another run's checkpoint and results. HUBER_DELTA needs no guard only because
# LOSS=huber already forces a TAG; these three have no such transitive cover.
needs_tag FRE_EMBED_SIZE "$FRE_EMBED_SIZE" "32"
needs_tag FRE_HIDDEN_SIZE "$FRE_HIDDEN_SIZE" "64"
needs_tag FRE_HIDDEN_LAYERS "$FRE_HIDDEN_LAYERS" "1"
needs_tag FRE_DROPOUT "$FRE_DROPOUT" "0.1"
# FreEformer knobs, same rule and same reason: none of them reaches the
# experiment setting name, so any one changing alone overwrites another run's
# checkpoint and results.
needs_tag FE_EMBED_SIZE "$FE_EMBED_SIZE" "16"
needs_tag FE_ATTN_ENHANCE "$FE_ATTN_ENHANCE" "1"
needs_tag FE_ATTN_SOFTMAX "$FE_ATTN_SOFTMAX" "0"
needs_tag FE_ATTN_WEIGHT_PLUS "$FE_ATTN_WEIGHT_PLUS" "1"
needs_tag FE_ATTN_OUTSIDE_SOFTMAX "$FE_ATTN_OUTSIDE_SOFTMAX" "1"
needs_tag FE_PLOT_MAT "$FE_PLOT_MAT" "0"
needs_tag FE_PLOT_GRAD "$FE_PLOT_GRAD" "0"
needs_tag FE_CKA "$FE_CKA" "0"
# v2 per-branch overrides. None of these reaches the setting name -- the `nh`
# field in the name is the PATCH branch's d_model, which stays 64 -- so a change
# to any of them needs a TAG or it overwrites another run's checkpoint/results.
needs_tag FE_D_MODEL "$FE_D_MODEL" ""
needs_tag FE_D_FF "$FE_D_FF" ""
needs_tag FE_N_HEADS "$FE_N_HEADS" ""
needs_tag FE_E_LAYERS "$FE_E_LAYERS" ""
needs_tag FE_DROPOUT "$FE_DROPOUT" ""
needs_tag CC_EMBED_SIZE "$CC_EMBED_SIZE" "8"
needs_tag CC_HIDDEN "$CC_HIDDEN" "32"
needs_tag CC_HEAD_HIDDEN "$CC_HEAD_HIDDEN" "256"
needs_tag CC_DROPOUT "$CC_DROPOUT" "0.1"
# Dataset knobs. Only `data` and `model_id` reach the setting name, so each of
# these changing alone reuses an ETTh1 run's checkpoint/results path.
needs_tag DATA_ROOT "$DATA_ROOT" "./dataset/ETT-small/"
needs_tag DATA_PATH "$DATA_PATH" "ETTh1.csv"
needs_tag ENC_IN "$ENC_IN" "7"
needs_tag PERIOD_LIST "$PERIOD_LIST" ""
needs_tag PATCH_LEN "$PATCH_LEN" "48 32 16 6 3"
# X-attention knobs: none of them reaches the experiment setting name, so any
# non-default value needs a TAG or it overwrites another run's checkpoint.
needs_tag X_ATTN "$X_ATTN" "0"
needs_tag X_DIM "$X_DIM" "32"
needs_tag X_HEADS "$X_HEADS" "4"
needs_tag X_NEG "$X_NEG" "1"
needs_tag X_MOD "$X_MOD" "1"
needs_tag X_DROP "$X_DROP" "0.1"
needs_tag X_NORM "$X_NORM" "dyt"
needs_tag X_DIST "$X_DIST" "periodic"
needs_tag X_LAMBDA "$X_LAMBDA" "0.2"
needs_tag X_MU "$X_MU" "1.0"
needs_tag FSDH_X_ATTN "$FSDH_X_ATTN" "0"
needs_tag FSDH_PERIOD "$FSDH_PERIOD" "24"
needs_tag FSDH_PATCH_LEN "$FSDH_PATCH_LEN" "2"
needs_tag FSDH_X_DIM "$FSDH_X_DIM" "32"
needs_tag FSDH_X_HEADS "$FSDH_X_HEADS" "4"
needs_tag FSDH_X_DROP "$FSDH_X_DROP" ""

mkdir -p "$LOGDIR"

EXTRA=()
[ -n "$TAG" ] && EXTRA+=(--run_tag "$TAG")
[ "$WD" != "0.0" ] && EXTRA+=(--weight_decay "$WD")
[ "$FUSION_DIM" != "0" ] && EXTRA+=(--fusion_dim "$FUSION_DIM")
[ "$CKPT_AVG_K" != "1" ] && EXTRA+=(--ckpt_avg_k "$CKPT_AVG_K")
[ "$NO_FSDH" != "0" ] && EXTRA+=(--no_fsdh)
[ "$CROSS_ATTN" != "0" ] && EXTRA+=(--cross_attn 1 --ca_layers "$CA_LAYERS" --ca_heads "$CA_HEADS")
# Always passed: none of these appear in the setting name, so the defaults
# above exist only to keep the guard honest -- the value actually used is
# always explicit in the command line.
EXTRA+=(--fre_embed_size "$FRE_EMBED_SIZE" --fre_hidden_size "$FRE_HIDDEN_SIZE"
        --fre_hidden_layers "$FRE_HIDDEN_LAYERS" --fre_dropout "$FRE_DROPOUT")
# Always passed, same reasoning as the FreDN block above.
EXTRA+=(--embed_size "$FE_EMBED_SIZE" --attn_enhance "$FE_ATTN_ENHANCE"
        --attn_softmax_flag "$FE_ATTN_SOFTMAX" --attn_weight_plus "$FE_ATTN_WEIGHT_PLUS"
        --attn_outside_softmax "$FE_ATTN_OUTSIDE_SOFTMAX"
        --plot_mat_flag "$FE_PLOT_MAT" --plot_grad_flag "$FE_PLOT_GRAD"
        --CKA_flag "$FE_CKA")
# v2 per-branch overrides. Passed ONLY when set, so the default command line is
# byte-identical to v1's and the setting name (which encodes only the SHARED
# d_model) does not move. Empty means fall back to the shared field.
[ -n "$FE_D_MODEL" ]  && EXTRA+=(--fe_d_model  "$FE_D_MODEL")
[ -n "$FE_D_FF" ]     && EXTRA+=(--fe_d_ff     "$FE_D_FF")
[ -n "$FE_N_HEADS" ]  && EXTRA+=(--fe_n_heads  "$FE_N_HEADS")
[ -n "$FE_E_LAYERS" ] && EXTRA+=(--fe_e_layers "$FE_E_LAYERS")
[ -n "$FE_DROPOUT" ]  && EXTRA+=(--fe_dropout  "$FE_DROPOUT")
# Empty PERIOD_LIST means "let Times2D use its own DEFAULT_PERIOD_LIST", which is
# the ETTh1 configuration. Passing it explicitly for any other dataset.
[ -n "$PERIOD_LIST" ] && EXTRA+=(--period_list $PERIOD_LIST)
EXTRA+=(--cc_embed_size "$CC_EMBED_SIZE" --cc_hidden "$CC_HIDDEN"
        --cc_head_hidden "$CC_HEAD_HIDDEN" --cc_dropout "$CC_DROPOUT")
# X-attention: only passed when enabled, so the X_ATTN=0 command line is
# byte-identical to the baseline.  All sub-knobs are passed explicitly in that
# mode so the command line records the exact PNA configuration.
if [ "$X_ATTN" != "0" ]; then
  EXTRA+=(--x_attn "$X_ATTN" --x_attn_dim "$X_DIM" --x_attn_heads "$X_HEADS"
          --x_attn_negative "$X_NEG" --x_attn_modulation "$X_MOD"
          --x_attn_dropout "$X_DROP" --x_attn_norm "$X_NORM"
          --x_attn_distance "$X_DIST" --x_attn_lambda "$X_LAMBDA"
          --x_attn_mu "$X_MU")
fi
if [ "$FSDH_X_ATTN" != "0" ]; then
  if [ "$NO_FSDH" != "0" ]; then
    echo "WARNING: FSDH_X_ATTN=$FSDH_X_ATTN but NO_FSDH=$NO_FSDH; FSDH is disabled." >&2
  fi
  EXTRA+=(--fsdh_x_attn "$FSDH_X_ATTN" --fsdh_period "$FSDH_PERIOD"
          --fsdh_patch_len "$FSDH_PATCH_LEN" --fsdh_x_dim "$FSDH_X_DIM"
          --fsdh_x_heads "$FSDH_X_HEADS")
fi
[ -n "$FSDH_X_DROP" ] && EXTRA+=(--fsdh_x_dropout "$FSDH_X_DROP")

if [ -n "$TAG" ]; then SUFFIX="_${TAG}"; else SUFFIX=""; fi

for MODEL in $MODELS; do
  for PRED_LEN in $PRED_LENS; do
    LOG="${LOGDIR}/${MODEL}_${SEQ_LEN}_${PRED_LEN}${SUFFIX}.log"
    echo "===== ${MODEL} pred_len=${PRED_LEN} seed=${SEED} tag='${TAG}' -> ${LOG} ====="
    # -u so the log is readable while the run is in progress
    "$PY" -u run.py \
      --task_name long_term_forecast --is_training 1 \
      --root_path "$DATA_ROOT" --data_path "$DATA_PATH" --data "$DATA" \
      --features M --seq_len "$SEQ_LEN" --enc_in "$ENC_IN" --freq "$FREQ" \
      --e_layers 3 --n_heads 16 --d_model 64 --d_ff 64 \
      --dropout 0.5 --fc_dropout 0.25 \
      --patch_len $PATCH_LEN \
      --des Exp --lradj "$LRADJ" --train_epochs "$TRAIN_EPOCHS" --patience "$PATIENCE" \
      --loss "$LOSS" --huber_delta "$HUBER_DELTA" --tf32 "$TF32" \
      --top_k 5 --itr 1 --batch_size 128 --learning_rate "$LR" \
      --random_seed "$SEED" \
      --period_len "$PF_PERIOD_LEN" --latent_dim "$PF_LATENT_DIM" \
      --phase_layers "$PF_LAYERS" --phase_attn_heads "$PF_HEADS" \
      --phase_encoder_hidden "$PF_ENC_HIDDEN" --predictor_hidden "$PF_PRED_HIDDEN" \
      --phase_attn_dropout "$PF_ATTN_DROP" \
      --model_id "${DATA}_${SEQ_LEN}_${PRED_LEN}" --pred_len "${PRED_LEN}" --model "${MODEL}" \
      "${EXTRA[@]}" > "${LOG}" 2>&1
    echo "  exit=$? $(grep -E '^mse:' "${LOG}" | tail -1)"
  done
done
echo "===== done ====="
