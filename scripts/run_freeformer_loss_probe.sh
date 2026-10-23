#!/bin/bash
# Is the FreEformer branch's COLLAPSE caused by the shared training loss?
#
#   bash scripts/run_freeformer_loss_probe.sh          # 3 runs, ~20 min
#
# ---------------------------------------------------------------------------
# WHAT TRIGGERED THIS
# ---------------------------------------------------------------------------
# PROGRESS 13.9: standalone FreEformer does not merely under-perform -- it
# outputs a per-sample CONSTANT across the whole forecast window.
#
#   along-horizon std of the prediction:  FreEformer 0.0025  vs  truth 0.6549
#   its test MSE 0.7257 is WORSE than the optimal per-sample constant (0.6319),
#   while Times2D sits at 0.3660, i.e. 0.2659 BETTER than that constant.
#
# That is why 13.5 found the 8x width increase changed nothing: a constant
# predictor's quality does not depend on the model's width. And it means the CA
# ablations in 12 and 13 put a constant predictor in slot 2, which is a much
# weaker statement about cross-attention than "the branch is a bit weak".
#
# ---------------------------------------------------------------------------
# THE HYPOTHESIS
# ---------------------------------------------------------------------------
# spec 5.2 DELIBERATELY swapped the training objective for FreDN's published one
# and fixed it for the whole dual:
#
#     diff = rfft(pred, dim=1) - rfft(target, dim=1);  return diff.abs().mean()
#
# That is a frequency-domain L1. L1 targets the MEDIAN, and for the many
# high-frequency bins the target spectrum's median is near zero -- so its
# sign-valued gradient pushes the prediction flat. Times2D survives it on the
# strength of its periodic-conv inductive bias; FreEformer has only a global
# linear map from a flattened latent, so it collapses.
#
# Upstream FreEformer trains with TIME-DOMAIN L1 plus lr=1e-4, a type1 schedule
# and 30 epochs. We use freqmae plus lr=1e-3, constant, 20 epochs.
#
# ---------------------------------------------------------------------------
# WHAT THIS MEASURES -- and the read rule, fixed BEFORE the run
# ---------------------------------------------------------------------------
# Three standalone runs, identical in every way except --loss. Native
# architecture (FreEformer 512), seq_len=720, pred_len=96, seed 2021.
#
#   mse      time-domain MSE
#   huber    time-domain Huber(delta=1.0), i.e. ~L1 in the time domain
#   freqmae  the current objective (the baseline, reproduced under this tag)
#
# The read-out is NOT the MSE. It is the ALONG-HORIZON STD OF THE PREDICTION,
# because that is the quantity the collapse moved to 0.0025:
#
#   std > 0.05 (20x the collapsed value)  -> the loss was the cause. Re-run the
#                                            ablation with a time-domain loss,
#                                            and 12/13's CA null has to be
#                                            re-read: it was measured against a
#                                            constant, not against FreEformer.
#   std <~ 0.01                           -> the loss is NOT the cause. Next
#                                            suspects: lr (1e-3 constant vs
#                                            upstream 1e-4 + schedule) and
#                                            seq_len (720 here, 96 upstream).
#
# Reference points: collapsed = 0.0025, Times2D = 0.473, truth = 0.655.
# Report the std for all three, not just the winner -- a time-domain loss that
# lifts the std while making the MSE worse is still evidence about the cause.
set -u

cd "$(dirname "$0")/.." || exit 1

# FreEformer's native architecture (spec 2.1), same as the uw rounds.
UW_ARCH="FE_D_MODEL=512 FE_D_FF=512 FE_N_HEADS=8 FE_E_LAYERS=2 FE_DROPOUT=0.2"
COMMON="FE_EMBED_SIZE=16 \
FE_ATTN_ENHANCE=1 FE_ATTN_SOFTMAX=0 FE_ATTN_WEIGHT_PLUS=1 FE_ATTN_OUTSIDE_SOFTMAX=1 \
FE_PLOT_MAT=0 FE_PLOT_GRAD=0 FE_CKA=0 \
LR=0.001 LRADJ=none TF32=1 TRAIN_EPOCHS=20 PATIENCE=20"

for loss in mse huber freqmae; do
  echo
  echo "######## standalone FreEformer@512  loss=${loss}  pl=96  seed=2021 ########"
  env MODELS=FreEformer PRED_LENS=96 SEED=2021 TAG="uwloss_${loss}_s2021" \
      LOSS="$loss" $UW_ARCH $COMMON bash scripts/run_etth1.sh
done

cat <<'EOF'

==================== NOW READ THE COLLAPSE METRIC ====================
Do NOT look at the MSE first -- the whole point is the along-horizon std.

python - <<'PY'
import numpy as np, glob
base = 'results/long_term_forecast_ETTh1_720_96_FreEformer_ftETTh1_slM_ll720_pl48_dm96_nh64_el16_dl3_df1_fc64_eb1_dttimeF_True_Exp_uwloss_%s_s2021'
print(f"{'loss':>9} {'MSE':>10} {'horizon-std':>12} {'vs collapsed':>13}")
for loss in ('mse', 'huber', 'freqmae'):
    d = glob.glob(base % loss)
    if not d:
        print(f"{loss:>9}   NOT FOUND"); continue
    p = np.load(d[0] + '/pred.npy'); t = np.load(d[0] + '/true.npy')
    s = p.std(axis=1).mean()
    print(f"{loss:>9} {((p-t)**2).mean():>10.6f} {s:>12.6f} {s/0.0025:>12.1f}x")
print('\nreferences: collapsed 0.0025 | Times2D 0.473 | truth 0.655')
PY

Apply the threshold in this script's header, which was written before the run.
Both arms of the dual used freqmae, so if the loss turns out to be the cause,
the CA ablations in 12 and 13 were measured against a constant predictor and
that has to be said in their sections rather than quietly corrected here.
EOF
