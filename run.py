import argparse
import os
import sys

import torch
from exp.exp_long_term_forecasting import Exp_long_term_forecasting
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast

import random
import numpy as np
from multiprocessing import freeze_support
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate '
                         'predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# DLinear parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for
# each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.0, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')

parser.add_argument('--add', action='store_true', default=False, help='add')
parser.add_argument('--wo_conv', action='store_true', default=False, help='without convolution')
parser.add_argument('--serial_conv', action='store_true', default=False, help='serial convolution')

parser.add_argument('--kernel_list', type=int, nargs='+', default=[3, 7, 9], help='kernel size list')
parser.add_argument('--patch_len', type=int, nargs='+', default=[16], help='patch high')
parser.add_argument('--period_list', type=int, nargs='+', default=None,
                    help='Times2D period folding. None (the default) = use the model\'s own '
                         'DEFAULT_PERIOD_LIST, which is the ETTh1 configuration. Upstream used '
                         '[720, 360, 140, 70, 48] for weather, hardcoded in the model file.')
parser.add_argument('--stride', type=int, nargs='+', default=None, help='stride')

parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

# Dual-Times2D / Phase-Times2D ("phase view" branch)
# `period` is the fold width (what counts as one cycle); `patch_len` is the
# phase-bin width within a cycle, so tokens = ceil(period / patch_len).
# Defaults are all divisors of 720 and whole numbers of days (24h), giving
# 1/2/3/5/10-day cycles with 12 phase bins each and zero padding.
parser.add_argument('--phase_period_list', type=int, nargs='+', default=[24, 48, 72, 120, 240],
                    help='period list for the phase-view branch')
parser.add_argument('--phase_patch_len', type=int, nargs='+', default=[2, 4, 6, 10, 20],
                    help='patch length list for the phase-view branch (one per period)')
parser.add_argument('--no_fsdh', action='store_true', default=False,
                    help='disable the FSDH derivative-heatmap term (ablation)')

# PhaseFormer branch (ICLR 2026). Names and defaults are copied verbatim from
# the upstream PhaseFormer/run.py so that a config here maps 1:1 onto the
# published one; `tests/test_phaseformer_parity.py` pins the model itself.
#
# NOTE: phase_attn_heads must divide latent_dim -- CrossPhaseRoutingLayer
# asserts this. The defaults (8 and 4) satisfy it. The ETTh1 720 runs in
# PhaseFormer-test used latent_dim=6, which forces heads to 1/2/3/6.
#
# NOTE: the RevIN knobs below are a SEPARATE set from Times2D's --revin/--affine
# above. They are independent models today; when the dual branch is built the
# two must be driven off one shared RevIN instance (phase2d-v2.md section 14.3).
parser.add_argument('--period_len', type=int, default=24,
                    help='PhaseFormer: cycle length; also the number of phase bins (L)')
parser.add_argument('--latent_dim', type=int, default=8,
                    help='PhaseFormer latent dim D. The Z the feature tap exposes is '
                         '[B, C, period_len, latent_dim]; must be divisible by phase_attn_heads')
parser.add_argument('--phase_encoder_hidden', type=int, default=32,
                    help='PhaseFormer: hidden size of the phase embedding MLP (use_mlp only)')
parser.add_argument('--predictor_hidden', type=int, default=64,
                    help='PhaseFormer: hidden size of the predictor MLP (use_mlp only)')
parser.add_argument('--phase_layers', type=int, default=1,
                    help='PhaseFormer: number of cross-phase routing layers')
parser.add_argument('--phase_attn_heads', type=int, default=4,
                    help='PhaseFormer: heads in cross-phase attention (must divide latent_dim)')
parser.add_argument('--phase_attn_dropout', type=float, default=0.1,
                    help='PhaseFormer: dropout inside cross-phase attention')
parser.add_argument('--phase_attn_use_relpos', type=int, default=1,
                    help='PhaseFormer: relative positional encoding flag (currently unused upstream)')
parser.add_argument('--phase_attn_window', type=int, default=None,
                    help='PhaseFormer: optional local attention window')
parser.add_argument('--phase_attention_dim', type=int, default=None,
                    help='PhaseFormer: optional attention projection dim (kept for interface compat)')
parser.add_argument('--phase_num_routers', type=int, default=8,
                    help='PhaseFormer: number of learnable routers')
parser.add_argument('--phase_use_pos_embed', type=int, default=1,
                    help='PhaseFormer: add a learnable phase positional embedding')
parser.add_argument('--phase_pos_dropout', type=float, default=0.0,
                    help='PhaseFormer: dropout on the phase positional embedding')
parser.add_argument('--use_revin', type=int, default=1,
                    help='PhaseFormer: enable its (own) RevIN. Distinct from Times2D --revin.')
parser.add_argument('--revin_affine', type=int, default=0,
                    help='PhaseFormer RevIN affine. Distinct from Times2D --affine.')
parser.add_argument('--revin_eps', type=float, default=1e-5,
                    help='PhaseFormer RevIN epsilon')
parser.add_argument('--phase_encoder_use_mlp', type=int, default=0,
                    help='PhaseFormer: use a 2-layer MLP instead of a single Linear in the embedding')
parser.add_argument('--phase_encoder_dropout', type=float, default=0.0,
                    help='PhaseFormer: dropout in the embedding MLP')
parser.add_argument('--predictor_use_mlp', type=int, default=0,
                    help='PhaseFormer: use a 2-layer MLP instead of a single Linear in the predictor')
parser.add_argument('--predictor_dropout', type=float, default=0.0,
                    help='PhaseFormer: dropout in the predictor')

# DualPhaseFormer: the minimal Patch+Phase model with bidirectional
# cross-attention. CA is toggled with the existing --cross_attn / --ca_* flags
# so the ablation is a single flag on the same architecture.
parser.add_argument('--dual_hidden', type=int, default=128,
                    help='DualPhaseFormer: width of the fusion bottleneck. The raw '
                         'concat head (top_k*seq_len + period_len*d_model inputs) '
                         'overfits badly at seq_len=720, so both branch features are '
                         'projected to this width before fusing.')
parser.add_argument('--adapter_dropout', type=float, default=0.0,
                    help='DualPhaseFormer: dropout in the feature adapters and the '
                         'branch projection layers')

# FreDN branch (also used standalone). fre_embed_size doubles as the CA width
# when this model is the frequency branch of DualFreDN: the branch enters
# cross-attention through FeatureAdapter(2*fre_embed_size, d_model), so it must
# satisfy 2*fre_embed_size == d_model or the attention's queries and keys live
# in a low-rank subspace. See scripts/run_ca_experiment.sh's header for why that
# makes a null result uninterpretable. NOT part of the experiment setting name,
# so changing it without --run_tag overwrites another run's checkpoint.
parser.add_argument('--fre_embed_size', type=int, default=32,
                    help='FreDN: embedding width D. Upstream ETTh1 grid is {4,8,16}; 32 is '
                         'chosen so that 2*D == d_model and the CA adapter is an Identity.')
parser.add_argument('--fre_hidden_size', type=int, default=64,
                    help='FreDN: hidden width of the residual MLPs')
parser.add_argument('--fre_hidden_layers', type=int, default=1,
                    help='FreDN: number of stacked blocks in the residual MLPs')
parser.add_argument('--fre_dropout', type=float, default=0.1,
                    help='FreDN: dropout inside the residual MLPs')

# CrossChannelFreq: the frequency branch that mixes ACROSS CHANNELS, unlike every
# other model in this repo. See models/CrossChannelFreq.py for why it exists.
parser.add_argument('--cc_embed_size', type=int, default=8,
                    help='CrossChannelFreq: embedding width D')
parser.add_argument('--cc_hidden', type=int, default=32,
                    help='CrossChannelFreq: hidden width of the cross-channel MLP')
parser.add_argument('--cc_head_hidden', type=int, default=256,
                    help='CrossChannelFreq: hidden width of the output head')
parser.add_argument('--cc_dropout', type=float, default=0.1,
                    help='CrossChannelFreq: dropout')

# Regularisation / run-identification knobs (added for the ETTh1 720 study).
# All default to the original behaviour so existing results still reproduce.
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Adam weight decay (0.0 = original Times2D behaviour)')
parser.add_argument('--fusion_dim', type=int, default=0,
                    help='DualTimes2D only: if >0, project each branch to this dim before the '
                         'shared head instead of concatenating raw features. 0 = original.')
parser.add_argument('--ckpt_avg_k', type=int, default=1,
                    help='average the weights of the k best-validation checkpoints instead of '
                         'trusting a single val minimum. 1 = original behaviour.')
parser.add_argument('--run_tag', type=str, default='',
                    help='suffix appended to the experiment setting, so runs that differ only by '
                         'seed/hyperparameter do not overwrite each other. Empty = original naming.')

# Stage 2: bidirectional cross-attention between the two branches (DualTimes2D).
parser.add_argument('--cross_attn', type=int, default=0,
                    help='DualTimes2D: enable bidirectional cross-attention. 0 = stage-1 Dual.')
parser.add_argument('--ca_layers', type=int, default=1,
                    help='number of bidirectional cross-attention blocks')
parser.add_argument('--ca_heads', type=int, default=4,
                    help='attention heads (must divide d_model)')
parser.add_argument('--ca_dropout', type=float, default=0.0,
                    help='dropout inside the cross-attention blocks')

# Formers 
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default 1: value patch_embedding + temporal patch_embedding + positional patch_embedding 2: value '
                         'patch_embedding + temporal patch_embedding 3: value patch_embedding + positional patch_embedding 4: value patch_embedding')
parser.add_argument('--enc_in', type=int, default=7,
                    help='global_encoder input size')  # DLinear with --individual, use this hyperparameter as the number of
# channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
# --- FreEformer branch (models/FreEformer.py) -------------------------------
# Argument NAMES match D:\pycharm\FreEformer\run.py so a config maps 1:1 onto
# the upstream script, but four DEFAULTS deliberately do NOT match upstream's
# argparse defaults. Upstream ships embed_size=8, attn_softmax_flag=1,
# attn_weight_plus=0, attn_outside_softmax=0 (FreEformer/run.py:60,67-69); the
# values here are the ones its own Weather attn_ablation run passes
# (FreEformer/scripts/Weather_freformer_attn1.sh: `--embed_size 16` with
# e1_s0_p1_o1). Running `python run.py --model FreEformer` with no flags
# therefore builds a DIFFERENT attention configuration from upstream's
# defaults -- deliberately, but do not read this block as "upstream's defaults".
# scripts/run_etth1.sh passes every one of them explicitly, so no experiment
# depends on these.
# --factor, --output_attention, --activation and --dropout are shared with the
# Times2D family and already exist above, so they are not redeclared here.
parser.add_argument('--embed_size', type=int, default=16,
                    help='FreEformer embedding width. NOT --fre_embed_size, which is '
                         'the FreDN branch\'s latent and is unrelated')
parser.add_argument('--time_branch', type=int, default=0,
                    help='read by upstream FreEformer but unused on the attn_ablation '
                         'path; carried only so a config maps 1:1 (spec 9 R3)')
parser.add_argument('--temp_patch_len', type=int, default=16,
                    help='NO-OP: upstream FreEformer forward never reads it (spec 9 R3)')
parser.add_argument('--temp_stride', type=int, default=8,
                    help='NO-OP: upstream FreEformer forward never reads it (spec 9 R3)')
parser.add_argument('--attn_enhance', type=int, default=1,
                    help='FreEformer SF_mode: 1 uses the learned weight_mat prior')
parser.add_argument('--attn_softmax_flag', type=int, default=0,
                    help='FreEformer: softmax the weight_mat prior (and create tau)')
parser.add_argument('--attn_weight_plus', type=int, default=1,
                    help='FreEformer: add the prior instead of multiplying')
parser.add_argument('--attn_outside_softmax', type=int, default=1,
                    help='FreEformer: fold the prior in outside the softmax')
parser.add_argument('--plot_mat_flag', type=int, default=0,
                    help='MUST STAY 0. Upstream draws from the global RNG on EVERY eval '
                         '(its guard is random.random() < 1) and writes files, which '
                         'would break seed reproducibility (spec 9 R2). The port raises.')
parser.add_argument('--plot_grad_flag', type=int, default=0,
                    help='MUST STAY 0. Upstream draws from the global RNG during TRAINING '
                         '(this guard is not wrapped in `not self.training`) and writes '
                         'files (spec 9 R2). The port raises.')
parser.add_argument('--CKA_flag', type=int, default=0,
                    help='MUST STAY 0. Upstream calls random.uniform inside forward and '
                         'imports CudaCKA; the port removed the branch entirely')
# --- per-branch overrides for the FreEformer branch (v2) --------------------
# None = fall back to the shared fields above, i.e. v1's behaviour EXACTLY.
# The spec's 2.1 chose "architectures native per branch, training recipe
# shared": these five are architecture, so they are per-branch; lr/loss/
# batch_size/epochs stay shared because a dual has one optimizer and one output.
# NOTE THE PREFIX: `fe_` is the FreEformer branch, `fre_` is the FreDN branch.
# They are unrelated, and the shell side uses the same convention as FE_* / FRE_*
# (scripts/run_etth1.sh:140). `--fre_dropout` was already taken by FreDN, so
# naming these `fre_*` would have collided.
parser.add_argument('--fe_d_model', type=int, default=None,
                    help='FreEformer branch width. None -> --d_model (v1 behaviour). '
                         'Its native is 512; --d_model is the Times2D branch width.')
parser.add_argument('--fe_d_ff', type=int, default=None,
                    help='None -> --d_ff. Native 512.')
parser.add_argument('--fe_n_heads', type=int, default=None,
                    help='None -> --n_heads. Native 8; paired with fe_d_model=512 '
                         'that is 64 dims per head, against 4 if it inherits 16.')
parser.add_argument('--fe_e_layers', type=int, default=None,
                    help='None -> --e_layers. Native 2. NOTE this is a DIFFERENT '
                         'meaning from Times2D\'s e_layers (conv blocks vs attention '
                         'layers) -- they share the field by coincidence (spec 2.1).')
parser.add_argument('--fe_dropout', type=float, default=None,
                    help='None -> --dropout. Native 0.2. This value feeds '
                         'FullAttention_ablation\'s attention_dropout directly, so '
                         'inheriting the Times2D 0.5 halves the attention weights.')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of global_encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in global_encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn_dropout', type=float, default=0.05, help='attention dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse',
                    help="training loss. long_term_forecast accepts 'mse' (default, original "
                         "behaviour), 'huber' (the PhaseFormer reference protocol, phase2d-v2.md "
                         "7.3) or 'freqmae' (FreDN's frequency-domain MAE, which is the ENTIRE "
                         "objective in that paper -- its time-domain term is weighted zero).")
parser.add_argument('--huber_delta', type=float, default=1.0,
                    help="delta for --loss huber. 1.0 is the PhaseFormer reference value.")
parser.add_argument('--lradj', type=str, default='type3',
                    help="adjust learning rate. 'none' = no schedule at all, constant lr "
                         "(what the PhaseFormer reference actually ran -- see phase2d-v2.md 7.3).")
parser.add_argument('--tf32', type=int, default=0,
                    help="1 = torch.set_float32_matmul_precision('medium'), i.e. allow TF32 "
                         "matmuls. The PhaseFormer reference ran with this ON; the TSL "
                         "pipeline default (0) is off. Affects both speed and numerics.")
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# output log file
parser.add_argument('--log', type=str, default='./logs/LongForecasting/PatchTST_Electricity_336_96.log',
                    help='path of output log file')

# --- Times2D X-shaped attention (PHAT-style PNA) -----------------------------
# Replaces each period's conv2D with period-offset x period-aligned attention.
# Off by default; the baseline code path and state_dict are unchanged.
parser.add_argument('--x_attn', type=int, default=0,
                    help='Times2D: replace conv2D with X-shaped period attention')
parser.add_argument('--x_attn_dim', type=int, default=32,
                    help='PNA width (must be divisible by --x_attn_heads)')
parser.add_argument('--x_attn_heads', type=int, default=4, help='PNA heads')
parser.add_argument('--x_attn_negative', type=int, default=1,
                    help='use PHAT positive/negative offset logits')
parser.add_argument('--x_attn_modulation', type=int, default=1,
                    help='use PHAT periodic-distance Softplus modulation')
parser.add_argument('--x_attn_dropout', type=float, default=0.1,
                    help='dropout inside PNA')
parser.add_argument('--x_attn_norm', type=str, default='dyt',
                    choices=['dyt', 'layernorm', 'none'],
                    help='normalisation after the PNA residual')
parser.add_argument('--x_attn_lambda', type=float, default=0.2,
                    help='initial negative-branch weight (sigmoid)')
parser.add_argument('--x_attn_mu', type=float, default=1.0,
                    help='initial aligned-branch scale (softplus)')
parser.add_argument('--x_attn_distance', type=str, default='periodic',
                    choices=['periodic', 'linear'],
                    help='offset-axis distance; periodic is PHAT\'s circular one')
parser.add_argument('--fsdh_x_attn', type=int, default=0,
                    help='replace FSDH\'s Inception conv with PNA')
parser.add_argument('--fsdh_period', type=int, default=24,
                    help='FSDH fold period (ETTh1 hourly: 24)')
parser.add_argument('--fsdh_patch_len', type=int, default=2,
                    help='FSDH patch width within a period')
parser.add_argument('--fsdh_x_dim', type=int, default=32,
                    help='FSDH PNA width (must be divisible by --fsdh_x_heads)')
parser.add_argument('--fsdh_x_heads', type=int, default=4, help='FSDH PNA heads')
parser.add_argument('--fsdh_x_dropout', type=float, default=None,
                    help='FSDH PNA dropout; None falls back to --x_attn_dropout')


def main():
    args = parser.parse_args()

    # output
    # sys.stdout = open(args.log, 'w')

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Must be set before any matmul runs. TF32 keeps 10 mantissa bits instead of
    # fp32's 23, so it shifts results by roughly the same magnitude as the
    # ETTh1-720 seed noise -- it is a protocol knob, not a detail.
    if args.tf32:
        torch.set_float32_matmul_precision('medium')
        print('tf32: torch.set_float32_matmul_precision("medium")')

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_long_term_forecasting
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            if args.run_tag:
                setting = '{}_{}'.format(setting, args.run_tag)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.task_name,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)
        if args.run_tag:
            setting = '{}_{}'.format(setting, args.run_tag)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()




if __name__ == '__main__':
    freeze_support()
    main()