#!/bin/bash
# The DualFreDN cross-attention ablation: Times2D + FreDN, CA off vs on.
#
# Mirrors scripts/run_ca_experiment.sh exactly, with the frequency branch swapped
# from PhaseFormer to FreDN. Same protocol (constant lr=1e-3, TF32, batch 128),
# same 3 seeds x 4 horizons x 2 arms, so the result is a row that can be dropped
# straight into PROGRESS.md section 4.5's table.
#
#   bash scripts/run_fre_experiment.sh stage1   # 8 runs, ~1.5-2 h  -- go/no-go
#   bash scripts/run_fre_experiment.sh stage2   # 16 runs, ~3-4 h
#
# BEFORE looking at the result, compute the rho guard (spec 8.2), which needs
# standalone runs of BOTH models under the matching tag:
#
#   bash scripts/run_times2d_matched.sh    # Times2D half -- REQUIRED
#   bash scripts/run_fredn_standalone.sh   # FreDN half
#   python scripts/analyze_complementarity.py --model-a Times2D --model-b FreDN \
#       --tag 'fre_s20*'
#
# Both halves are needed. analyze_complementarity.py globs
# `results/..._{model}_*_Exp_{tag}`, and the repo's pre-existing Times2D runs
# carry `refproto_s20*`, NOT `fre_s20*` -- so running only the FreDN half finds
# 12 FreDN runs and 0 Times2D runs and the guard degenerates to a MISSING line
# per horizon and exit 1.
set -u

cd "$(dirname "$0")/.." || exit 1
STAGE="${1:-stage1}"

COMMON="FRE_EMBED_SIZE=32 LR=0.001 LRADJ=none LOSS=freqmae TF32=1"

# FreDN's own ETTh1 param_map (FreDN/scripts/ETTh1.sh), indexed by pred_len:
#   dropout hidden_layers hidden_size
# These are that script's grid-search winners. Both arms use the same set, so
# the ablation stays a single variable; they are varied per horizon only so the
# frequency branch is not handicapped by one horizon's tuning.
declare -A FRE_DROPOUT=( [96]=0.3   [192]=0.3  [336]=0.1   [720]=0.1 )
declare -A FRE_HLAYERS=( [96]=1     [192]=1    [336]=2     [720]=4   )
declare -A FRE_HSIZE=(   [96]=64    [192]=16   [336]=128   [720]=64  )

run_arm() {  # run_arm <seed> <ca 0|1>
  local seed="$1" ca="$2"
  local tag="ca${ca}_s${seed}"
  local extra=""
  [ "$ca" = "1" ] && extra="CROSS_ATTN=1 CA_LAYERS=1 CA_HEADS=4"
  echo
  echo "######## seed=$seed  cross_attn=$ca  (tag=$tag) ########"
  for pl in 96 192 336 720; do
    env MODELS=DualFreDN PRED_LENS="$pl" SEED="$seed" TAG="$tag" \
        FRE_DROPOUT="${FRE_DROPOUT[$pl]}" \
        FRE_HIDDEN_LAYERS="${FRE_HLAYERS[$pl]}" \
        FRE_HIDDEN_SIZE="${FRE_HSIZE[$pl]}" \
        $COMMON $extra bash scripts/run_etth1.sh
  done
}

if [ "$STAGE" = "stage1" ]; then
  run_arm 2021 0
  run_arm 2021 1
else
  for s in 2022 2023; do
    run_arm "$s" 0
    run_arm "$s" 1
  done
fi

cat <<'EOF'

==================== HOW TO READ THIS ====================
  python build_results_table.py --tagged
  python scripts/report_alpha_beta.py --glob '*DualFreDN*'

Two questions, in order. Answer them separately -- the model has per-branch
prediction terms (alpha*Y_patch + beta*Y_freq) that contribute to the first but
cancel out of the second.

1) DOES THE DUAL BEAT A SINGLE BRANCH?
     DualFreDN (ca0 or ca1) vs Times2D and vs FreDN.
     Reference: the test-fitted ORACLE for blending the two branches' outputs
     was +0.0070 for the Times2D/PhaseFormer pair. Treat that as the scale of
     the headroom, not a target.

2) DOES CROSS-ATTENTION ADD ANYTHING?  <-- this is what the exercise is for
     ca1 vs ca0.

DECISION RULES (set before looking, so the result cannot be rationalised):
  ca1 beats ca0 by > 0.011 (the ETTh1-720 seed floor) AND consistently
      -> CA works with a frequency branch. Only then consider MoE / Memory.
  ca1 ~= ca0
      -> CA adds nothing. THEN CHECK RHO before concluding anything about CA:
         rho >= 0.93  the branch was as homogeneous as PhaseFormer (0.9377);
                      this is the same death as the PhaseFormer run, and CA is
                      not the variable.
         rho <  0.93  the branches really do differ and CA STILL does nothing.
                      That is the stronger result: the mechanism itself is the
                      problem, not the branch choice.
  ca1 worse than ca0
      -> matches v1, which also saw CA hurt at 3 of 4 horizons.

ALSO CHECK: python scripts/report_alpha_beta.py. alpha/beta start at 0.5/0.5;
values left near 0.5 mean the optimiser found the fused path no more
trustworthy than the branch predictions.
EOF
