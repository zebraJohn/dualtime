#!/bin/bash
# The rho guard for the DualFreEformer ablation (PROGRESS.md section 12).
#
#   bash scripts/run_freeformer_rho.sh times2d     # 12 runs, ~1 h
#   bash scripts/run_freeformer_rho.sh freeformer  # 12 runs, ~1.2 h
#   bash scripts/run_freeformer_rho.sh all         # both halves
#
# V2 (the DualFreEformer-v2 guard -- FreEformer at its NATIVE 512):
#   bash scripts/run_freeformer_rho.sh all512      # 24 runs
#
# Use `all512`, not two separate calls with a hand-picked tag. Both halves must
# carry the SAME tag, and it must not be `fe_s*` -- a FreEformer@512 run's
# setting name differs from @64 only in the `nh` field, and
# analyze_complementarity.py globs by TAG alone, so under `fe_s20*` the two
# configurations would both match and be folded into one rho. Times2D is
# unchanged between the two guards but still has to be re-run, because the tag
# is what pairs the halves.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# PROGRESS 12.6 records that the guard was pre-registered as SKIPPED, so the
# section says nothing at all about rho(Times2D, FreEformer). That is honest but
# it leaves the null unexplained: 12 can say "CA did not help" but cannot say
# whether the two branches had anything to fuse in the first place.
#
# The reason to run it now is that answering "why" with numbers FROM THIS PAIR
# requires exactly this. The argument that the branches are too homogeneous to
# leave room for CA was borrowed from 9.4 (Times2D+FreDN) and 11.3
# (Times2D+CrossChannelFreq); it was never measured here. And this pair differs
# structurally from every previous one -- FreEformer's own Encoder_ori already
# attends over the channel axis (spec 2.2.1) -- so borrowing that conclusion is
# exactly what should not be done.
#
# ---------------------------------------------------------------------------
# WHY BOTH HALVES, AND WHY NEITHER CAN BE REUSED
# ---------------------------------------------------------------------------
# analyze_complementarity.py globs `results/..._{model}_*_Exp_{tag}`, so BOTH
# models must carry the SAME tag. Times2D runs already exist under
# `fre_s20*` (scripts/run_times2d_matched.sh) -- but those used the OLD protocol
# (default TRAIN_EPOCHS=100 PATIENCE=10, i.e. early stopping), while the
# ablation this guard explains ran at a FIXED 20 epochs.
#
# Comparing residual correlations across two training protocols would confound
# the thing the guard exists to disambiguate. run_times2d_matched.sh's own
# header says the same about loss functions: "PROGRESS.md's 遗留的诚实说明 table
# is already a list of protocol-mixing mistakes". So both halves are re-run
# under the matching protocol with a fresh tag, `fe_s${seed}` -- `fe_` for the
# FreEformer round, mirroring how `fre_` was used for the FreDN round.
#
# ---------------------------------------------------------------------------
# COST, MEASURED
# ---------------------------------------------------------------------------
# Times2D standalone ETTh1: 14.1 s/epoch (read off the existing fre_s20* logs;
#   three seeds within 0.03 s of each other). 12 runs x 20 epochs ~= 56 min.
# FreEformer standalone: 17.5 s/epoch (from the DualFreEformer smoke, which is
#   the same branch at the same width). 12 runs x 20 epochs ~= 70 min.
# Budget ~2-2.5 h for both halves.
#
# ---------------------------------------------------------------------------
# HOW TO READ IT -- decide this BEFORE looking
# ---------------------------------------------------------------------------
# The guard produces rho and an ORACLE gain: the improvement from the best
# fixed mixing weight chosen with hindsight. CA is a LEARNED mixer, so it
# cannot beat the oracle. Therefore:
#
#   oracle gain BELOW the floor  -> the null is explained: there was no room
#                                   for ANY mixer to help, learned or not.
#                                   Section 12's conclusion stands and now has
#                                   its own attribution.
#   oracle gain ABOVE the floor  -> something IS fusable and CA failed to find
#                                   it. THAT is the interesting result, and the
#                                   next question becomes about the mechanism
#                                   rather than about the branches.
#
# CAVEAT, stated up front: the floor passed to analyze_complementarity.py
# (0.011 on ETTh1) was derived under the OLD early-stopping protocol. This
# script changes the protocol, so that floor is INHERITED, not re-measured.
# Treat a near-floor result as inconclusive rather than decisive; a result
# several times away from it in either direction is not affected.
set -u

cd "$(dirname "$0")/.." || exit 1
HALF="${1:-all}"

# Matches scripts/run_freeformer_experiment.sh exactly, so the standalone models
# are trained the way the dual's branches were.
PROTOCOL="LR=0.001 LRADJ=none LOSS=freqmae TF32=1 TRAIN_EPOCHS=20 PATIENCE=20"

# The FreEformer knobs, pinned as in the ablation (spec 3.4).
FE_KNOBS="FE_EMBED_SIZE=16 \
FE_ATTN_ENHANCE=1 FE_ATTN_SOFTMAX=0 FE_ATTN_WEIGHT_PLUS=1 FE_ATTN_OUTSIDE_SOFTMAX=1 \
FE_PLOT_MAT=0 FE_PLOT_GRAD=0 FE_CKA=0"

# FreEformer's NATIVE architecture, for the v2 guard. The `freeformer` half above
# inherits the Times2D protocol, which is what made its standalone score 0.727
# against Times2D's 0.372 while the same weights at 512 reproduce upstream's
# 0.47068 exactly (PROGRESS 12.10). The `freeformer512` half removes that
# handicap; Times2D is identical either way, but must be RE-RUN under the
# matching tag because analyze_complementarity.py globs by tag only.
UW_ARCH="FE_D_MODEL=512 FE_D_FF=512 FE_N_HEADS=8 FE_E_LAYERS=2 FE_DROPOUT=0.2"

run_half() {  # run_half <MODEL> <tag-prefix> <extra env...>
  local model="$1"; local prefix="$2"; shift 2
  for seed in 2021 2022 2023; do
    for pl in 96 192 336 720; do
      echo "######## standalone ${model}  seed=${seed}  pred_len=${pl} ########"
      env MODELS="$model" PRED_LENS="$pl" SEED="$seed" TAG="${prefix}${seed}" \
          $PROTOCOL "$@" bash scripts/run_etth1.sh
    done
  done
}

case "$HALF" in
  times2d)   run_half Times2D      fe_s ;;
  freeformer) run_half FreEformer  fe_s $FE_KNOBS ;;
  all)       run_half Times2D      fe_s; run_half FreEformer fe_s $FE_KNOBS ;;
  # --- v2 guard ---------------------------------------------------------
  # Both halves must carry the SAME tag, and that tag must NOT be `fe_s*`.
  # A FreEformer@512 run's setting name differs from @64 only in the `nh` field
  # (..._nh512_... vs ..._nh64_...), and analyze_complementarity.py globs by TAG
  # alone -- so under `fe_s20*` the two configurations would both match and be
  # folded into one rho. Hence the separate `uw_s*` tag, and hence Times2D has
  # to be re-run under it even though its own config is unchanged.
  times2d512)      run_half Times2D     uw_s ;;
  freeformer512)   run_half FreEformer  uw_s $FE_KNOBS $UW_ARCH ;;
  all512)          run_half Times2D     uw_s
                   run_half FreEformer  uw_s $FE_KNOBS $UW_ARCH ;;
  *) echo "usage: $0 [times2d|freeformer|all|times2d512|freeformer512|all512]" >&2; exit 2 ;;
esac

cat <<'EOF'

==================== HOW TO READ THIS ====================
Both halves must be complete before the analysis means anything -- one half
alone finds 12 runs of one model and 0 of the other, and silently produces a
MISSING line per horizon (this is why run_times2d_matched.sh exists at all).

  # v1 guard (FreEformer at 64, the Times2D protocol):
  python scripts/analyze_complementarity.py \
      --model-a Times2D --model-b FreEformer --tag 'fe_s20*'

  # v2 guard (FreEformer at its native 512):
  python scripts/analyze_complementarity.py \
      --model-a Times2D --model-b FreEformer --tag 'uw_s20*'

Use the tag matching the half you just ran -- and if both guards have been run,
check the `n` column: a tag pattern that matches BOTH configurations would
report 6 seeds per horizon instead of 3.

Then apply the criterion in the header above, which was written before the run.
Report the rho NEXT TO the ablation number, and remember the floor caveat:
0.011 came from the old protocol.
EOF
