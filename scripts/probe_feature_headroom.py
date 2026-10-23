"""Does the FEATURE space hold more usable information than the OUTPUT space?

THE QUESTION THIS ANSWERS
phase2d-v2.md's Step 2 measured, in output space, that Times2D and PhaseFormer
residuals correlate at rho ~ 0.935, and that the best possible fixed linear blend
of their two predictions gains only +0.0053 MSE -- half the 0.011 noise floor.

That is NOT by itself an argument against Cross-Attention, and it would be wrong
to treat it as one. The oracle bound is a bound on LINEAR BLENDING OF OUTPUTS, a
strictly smaller hypothesis class than a joint model that lets one branch's
latent modulate how the other reads its own features. Cross-Attention could in
principle exploit structure that a convex combination of the two outputs cannot
express. v1's CA failure does not settle it either: v1's two branches were the
same architecture under a different period_list (rho ~ 0.99, near-duplicates), so
CA has never been tested on a genuinely different pair.

So: is there anything in the features for CA to combine? This script answers that
directly and cheaply, WITHOUT building the dual-branch model.

METHOD
Both checkpoints are frozen. For every (sample, channel) row we extract
    Times2D     period_token_features -> 66 tokens x d_model 64 = 4224 dims
    PhaseFormer forward_features Z    -> 24 phases  x latent 6 =  144 dims
and fit readouts to predict the RevIN-normalised future, denormalising
analytically with the same per-sample statistics RevIN used.

    probe A : Times2D features     -> y
    probe B : PhaseFormer features -> y
    probe C : [A ; B] concatenated -> y     <- the one that matters

TWO READOUTS, because each covers a different objection:

  ridge  -- linear, closed form, one lambda shared across all feature blocks.
            Cheap reference. Its weakness: a single lambda shrinks the 4224-dim
            Times2D block and the 144-dim PhaseFormer block together, which can
            MASK marginal value in the small block.

  mlp    -- nonlinear, weight decay + early stopping, so the two blocks are
            regularised independently, and it can express interactions a linear
            readout cannot. This is the stronger test and it drives the verdict
            when both are run.

Both are selected on VALIDATION and scored once on TEST.

CALIBRATION: on a synthetic target built as a pure product of one feature from
each block -- expressible by no linear readout at all -- ridge stays at the mean
predictor (~1.006) on every subset while the MLP on the concatenation recovers
it (0.0042 against a 0.0001 noise floor). So the MLP really can find
complementarity that ridge cannot, which is the whole point of running it.

READING IT
    C ~= min(A, B)            the two feature sets carry the same information.
                              Cross-Attention has nothing to combine. Do not
                              build the dual model.
    C clearly < min(A, B)     the features jointly carry information neither
                              carries alone -- CA has real headroom.

Before reading ANY of that, two gates have to pass:

  1. The SANITY line. A probe on a branch's own features must roughly reproduce
     that branch's output (Times2D's readout is itself linear from these tokens,
     so probe A should land slightly worse than Times2D -- it lacks only FSDH).
     A large gap means the features or the denormalisation are wrong and every
     other number is worthless.

  2. No probe may have its lambda pinned to an END of the penalty grid. The
     concat probe has 4368 features x pred_len outputs -- 3.1M parameters at
     pred_len=720 -- against ~50k training rows, so it is heavily
     underdetermined and genuinely needs enormous shrinkage. A lambda at the
     grid edge means the optimum lies outside and the probe is mis-regularised;
     the script flags this and refuses to give a verdict.

The default grid runs to 1e10 for exactly this reason. An earlier version
topped out at 1e4, every probe pinned to that maximum, probe A failed to
reproduce Times2D, and the whole table had to be thrown away.

Note the OUTPUT oracle (from the saved predictions) is TEST-fitted -- but it has
only ONE free parameter, so on ~14-19k test rows it is a genuine upper bound and
fair to quote. An analogous test-fitted FEATURE probe would have 3.1M parameters
and would just overfit; that comparison is deliberately not made here.

HONEST LIMITS -- read before quoting a number
  * Features are FROZEN. This is the one gap neither readout can close: a
    jointly trained model computes its features jointly, so it could in
    principle create an interaction that does not exist between the two frozen
    sets. That is precisely what Cross-Attention would add, and only building it
    settles it. Everything here bounds what a READOUT of fixed features can do.
  * The MLP is one specific nonlinearity (a 2-layer MLP over the whole feature
    vector). It is far more expressive than ridge and demonstrably finds
    products, but it is not a proof that no readout of any form could do better.
  * A result only counts if it clears the 0.011 noise floor AND the sanity gate.

Usage:
    python scripts/probe_feature_headroom.py                      # all horizons
    python scripts/probe_feature_headroom.py --pred-lens 720
    python scripts/probe_feature_headroom.py --probe ridge        # linear only, ~1s
    python scripts/probe_feature_headroom.py --probe mlp          # nonlinear only
    python scripts/probe_feature_headroom.py --tag refproto --max-train 20000
"""
import argparse
import glob
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data_provider.data_factory import data_provider  # noqa: E402
from models.Times2D import Times2DBackbone, Model as Times2DModel  # noqa: E402
from models.PhaseFormer import Model as PhaseFormerModel  # noqa: E402

REVIN_EPS = 1e-5


# --------------------------------------------------------------------------
# configuration -- must reproduce the training config exactly, or the
# checkpoints will not load.
# --------------------------------------------------------------------------
def build_args(pred_len, batch_size=128):
    args = SimpleNamespace(
        random_seed=2021, task_name='long_term_forecast', is_training=0,
        model_id=f'ETTh1_720_{pred_len}', model='Times2D', data='ETTh1',
        root_path='./dataset/ETT-small/', data_path='ETTh1.csv',
        features='M', target='OT', freq='h', checkpoints='./checkpoints/',
        seq_len=720, label_len=48, pred_len=pred_len, seasonal_patterns='Monthly',
        fc_dropout=0.25, head_dropout=0.0, add=False, wo_conv=False, serial_conv=False,
        kernel_list=[3, 7, 9], patch_len=[48, 32, 16, 6, 3], period_list=[24, 12],
        stride=None, padding_patch='end', revin=1, affine=0, subtract_last=0,
        decomposition=0, kernel_size=25, individual=0, top_k=5, num_kernels=6,
        embed_type=0, enc_in=7, dec_in=7, c_out=7,
        d_model=64, n_heads=16, e_layers=3, d_layers=1, d_ff=64,
        moving_avg=25, factor=1, distil=True, dropout=0.5, attn_dropout=0.05,
        embed='timeF', activation='gelu', output_attention=False, do_predict=False,
        num_workers=0, itr=1, train_epochs=100, batch_size=batch_size, patience=10,
        learning_rate=0.0001, des='Exp', loss='mse', lradj='TST', pct_start=0.3,
        use_amp=False, use_gpu=False, gpu=0, use_multi_gpu=False,
        devices='0,1,2,3', test_flop=False,
        phase_period_list=[24, 48, 72, 120, 240],
        phase_patch_len=[2, 4, 6, 10, 20], no_fsdh=False,
        weight_decay=0.0, fusion_dim=0, ckpt_avg_k=1, run_tag='',
        cross_attn=0, ca_layers=1, ca_heads=4, ca_dropout=0.0,
        # PhaseFormer block
        period_len=24, latent_dim=6, phase_encoder_hidden=16, predictor_hidden=32,
        phase_layers=1, phase_attn_heads=1, phase_attn_dropout=0.0,
        phase_attn_use_relpos=1, phase_attn_window=None, phase_attention_dim=None,
        phase_num_routers=8, phase_use_pos_embed=1, phase_pos_dropout=0.0,
        use_revin=1, revin_affine=0, revin_eps=REVIN_EPS,
        phase_encoder_use_mlp=0, phase_encoder_dropout=0.0,
        predictor_use_mlp=0, predictor_dropout=0.0,
    )
    return args


def find_setting(pred_len, model, tag):
    pat = (f'long_term_forecast_ETTh1_720_{pred_len}_{model}_*_Exp_{tag}' if tag
           else f'long_term_forecast_ETTh1_720_{pred_len}_{model}_*_Exp')
    hits = sorted(glob.glob(os.path.join(REPO, 'checkpoints', pat)))
    if not hits:
        return None
    if len(hits) > 1:
        print(f'  WARNING: {len(hits)} checkpoints match {model} pl={pred_len} tag={tag!r}; '
              f'using the first ({os.path.basename(hits[0])})')
    return hits[0]


def load_model(args, model_name, tag):
    setting = find_setting(args.pred_len, model_name, tag)
    if setting is None:
        raise SystemExit(f'no checkpoint for {model_name} pl={args.pred_len} tag={tag!r}')
    path = os.path.join(setting, 'checkpoint.pth')
    if model_name == 'Times2D':
        model = Times2DModel(args).float()
    elif model_name == 'PhaseFormer':
        model = PhaseFormerModel(args).float()
    else:
        raise SystemExit(f'unknown model {model_name}')
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, path


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------
def extract(model_t2d, model_pf, loader, device, max_rows=None):
    """Return per-(sample, channel) features, normalised targets and RevIN stats."""
    f_t2d, f_pf, y_norm, mu_s, sg_s = [], [], [], [], []
    n_rows = 0
    backbone = model_t2d.model          # Times2DModel wraps Times2DBackbone

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            B, T, C = batch_x.shape

            # RevIN statistics: mean/var over TIME, per (sample, channel).
            # Both branches use this same parameterisation (affine off,
            # subtract_last off), so one set of stats serves both.
            mu = batch_x.mean(dim=1)                                   # [B, C]
            var = batch_x.var(dim=1, unbiased=False)
            sigma = torch.sqrt(var + REVIN_EPS)                        # [B, C]

            # --- Times2D token features ---
            x_norm_t2d = ((batch_x - mu[:, None, :]) / sigma[:, None, :])   # [B,T,C]
            toks = backbone.period_token_features(x_norm_t2d.permute(0, 2, 1))  # [B,C,d,L_i]
            t2d = torch.cat([t.reshape(B, C, -1) for t in toks], dim=-1)   # [B,C,66*64]

            # --- PhaseFormer latent Z ---
            Z, _ = model_pf.forward_features(batch_x)                  # [B,C,L,D]
            pf = Z.reshape(B, C, -1)                                   # [B,C,24*latent]

            # --- target, in the normalised space the models predict in ---
            yt = batch_y[:, -model_pf.pred_len:, :]                    # [B,pl,C]
            yt = yt.permute(0, 2, 1)                                   # [B,C,pl]
            yn = (yt - mu[:, :, None]) / sigma[:, :, None]

            # Stays on the device: the gram matrix below is the whole cost of
            # this script, and bouncing ~1GB back to host memory to run it on
            # the CPU is the difference between seconds and ten minutes.
            f_t2d.append(t2d.reshape(B * C, -1))
            f_pf.append(pf.reshape(B * C, -1))
            y_norm.append(yn.reshape(B * C, -1))
            mu_s.append(mu.reshape(B * C))
            sg_s.append(sigma.reshape(B * C))
            n_rows += B * C
            if max_rows and n_rows >= max_rows:
                break

    X_t2d = torch.cat(f_t2d)
    X_pf = torch.cat(f_pf)
    Y = torch.cat(y_norm)
    MU = torch.cat(mu_s)
    SG = torch.cat(sg_s)
    if max_rows:
        X_t2d, X_pf, Y, MU, SG = (a[:max_rows] for a in (X_t2d, X_pf, Y, MU, SG))
    return X_t2d, X_pf, Y, MU, SG


# --------------------------------------------------------------------------
# ridge probe (primal form; d << n here, so d x d is the cheap factorisation)
# --------------------------------------------------------------------------
def ridge_prep(Xtr, Ytr):
    """Everything that does NOT depend on the ridge penalty.

    Two deliberate choices, both about the gram matrix X^T X (n*d^2, ~2e12 flops
    at n=54656 d=4368 -- the entire cost of this script):

      * hoisted out of the lambda loop, because only its DIAGONAL changes with
        lambda. Recomputing per lambda turns a ~1min job into a >10min one.
      * run on the GPU in float32. cuBLAS sgemm is orders of magnitude faster
        than the CPU here, and float32 halves a ~1GB working set. Accumulating
        the gram in float64 is not worth it: consumer GPUs run fp64 at ~1/64 of
        fp32, which would land us back at CPU speed.

    Precision: Xz is standardised to unit variance, so gram entries are ~n*rho
    and the diagonal is exactly n. float32's relative error there is ~1e-7, far
    below anything this probe is trying to resolve.

    Primal form on purpose: d (~4400) << n (~55000), so a d x d factorisation
    beats the n x n dual by orders of magnitude.
    """
    mean = Xtr.mean(dim=0, keepdim=True)
    std = Xtr.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    Ymean = Ytr.mean(dim=0, keepdim=True)
    Xz = (Xtr - mean) / std
    G = Xz.T @ Xz
    b = Xz.T @ (Ytr - Ymean)
    del Xz
    return dict(mean=mean, std=std, Ymean=Ymean, G=G, b=b)


def ridge_solve(prep, lam):
    """Least-squares weights for one penalty value, reusing the cached gram."""
    G = prep['G'].clone()
    G.diagonal().add_(lam)
    return torch.linalg.solve(G, prep['b'])


def ridge_apply(W, prep, X):
    return (X - prep['mean']) / prep['std'] @ W + prep['Ymean']


def mse_original(pred_norm, Y_norm_true, MU, SG):
    """Denormalise with the same RevIN statistics the models use, and score in
    the ORIGINAL space -- the space every reported number in this repo lives in.
    """
    pred = pred_norm * SG[:, None] + MU[:, None]
    true = Y_norm_true * SG[:, None] + MU[:, None]
    return float(((pred - true) ** 2).mean())


# --------------------------------------------------------------------------
# nonlinear (MLP) readout
# --------------------------------------------------------------------------
def mlp_probe(Xtr, Ytr, Xva, Yva, MU_va, SG_va, Xte, Yte, MU_te, SG_te,
              hidden, wds, seeds, epochs, patience, batch, device):
    """Fit one MLP per (weight_decay, seed); return the val-best configuration.

    WHY THIS EXISTS ON TOP OF THE RIDGE PROBE
    Ridge applies ONE lambda to the entire feature block, but the concat probe
    mixes 4224 Times2D dims with 144 PhaseFormer dims that plausibly want very
    different shrinkage. A single lambda can only shrink them together, so it
    can MASK marginal value in the small block -- exactly the failure mode that
    would make us wrongly declare "no complementarity". An MLP regularises the
    two blocks implicitly and separately (via weight decay + early stopping),
    and adds the nonlinearity Cross-Attention would also have.

    Selection is on VALIDATION ONLY. Weight decay and seed are chosen by val
    MSE; the test set is scored once, for the winning configuration. The val
    number is computed in the ORIGINAL space, the same metric as everything else.

    Returns (val_mse, weight_decay, seed, test_mse, epochs_run).
    """
    mean = Xtr.mean(dim=0, keepdim=True)
    std = Xtr.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    Ztr, Zva, Zte = ((X - mean) / std for X in (Xtr, Xva, Xte))
    Ymean = Ytr.mean(dim=0, keepdim=True)
    n, d = Ztr.shape
    m = Ytr.shape[1]

    best = None
    for wd in wds:
        for seed in seeds:
            torch.manual_seed(seed)
            net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                nn.Linear(hidden, m)).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
            best_val, best_state, bad, ep = float('inf'), None, 0, 0
            for ep in range(epochs):
                net.train()
                perm = torch.randperm(n, device=device)
                for i in range(0, n, batch):
                    idx = perm[i:i + batch]
                    opt.zero_grad(set_to_none=True)
                    loss = ((net(Ztr[idx]) + Ymean - Ytr[idx]) ** 2).mean()
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = mse_original(net(Zva) + Ymean, Yva, MU_va, SG_va)
                if v < best_val - 1e-7:
                    best_val, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                t = mse_original(net(Zte) + Ymean, Yte, MU_te, SG_te)
            if best is None or best_val < best[0]:
                best = (best_val, wd, seed, t, ep + 1)
            del net, opt, best_state
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-lens', type=int, nargs='+', default=[96, 192, 336, 720])
    ap.add_argument('--tag', default='refproto')
    ap.add_argument('--lambdas', type=float, nargs='+', default=None,
                    help="ridge penalties to search. Default is a log-spaced grid from "
                         "1e-2 to 1e10. It has to reach that far: the concat probe has "
                         "4368 features x pred_len outputs -- 3.1M parameters at "
                         "pred_len=720 -- against only ~50k training rows, so it is "
                         "heavily underdetermined and needs far more shrinkage than a "
                         "small grid offers. The first run used a grid topping out at "
                         "1e4, every probe pinned to that maximum, and probe A could not "
                         "reproduce Times2D's own output as a result.")
    ap.add_argument('--probe', choices=['ridge', 'mlp', 'both'], default='both',
                    help="which readouts to fit. 'mlp' closes the two gaps ridge "
                         "leaves open -- that a linear readout may understate what "
                         "is extractable, and that one shared lambda cannot shrink "
                         "a 4224-dim block and a 144-dim block independently.")
    ap.add_argument('--mlp-hidden', type=int, default=512)
    ap.add_argument('--mlp-wds', type=float, nargs='+',
                    default=[0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
                    help="weight-decay grid. This has to reach as far as the ridge "
                         "lambda grid does, for the same reason: probe A's first layer "
                         "is 4224x512 = 2.2M parameters against ~50k rows, so it wants "
                         "far more shrinkage than a small grid offers. The first "
                         "version capped this at 1e-2, every probe pinned to that "
                         "maximum, and probe A came out 19-39 percent WORSE than "
                         "Times2D's own output. A winner at either END of this list is "
                         "flagged as mis-regularised and invalidates the horizon.")
    ap.add_argument('--mlp-seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--mlp-epochs', type=int, default=150)
    ap.add_argument('--mlp-patience', type=int, default=8)
    ap.add_argument('--mlp-batch', type=int, default=512)
    ap.add_argument('--max-train', type=int, default=0,
                    help='cap training rows (0 = use all); useful on low-RAM machines')
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help="cuda by default; the gram matrix is ~2e12 flops and the "
                         "gap between CPU and GPU here is minutes vs seconds")
    cli = ap.parse_args()

    if cli.lambdas is None:
        cli.lambdas = [float(x) for x in np.logspace(-2, 10, 25)]
    cli.lambdas = sorted(cli.lambdas)

    device = torch.device(cli.device)
    if device.type == 'cpu':
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    else:
        # Let cuBLAS pick, but do not silently fall back to TF32 -- 10 mantissa
        # bits is not enough for a gram accumulated over ~55000 rows.
        torch.backends.cuda.matmul.allow_tf32 = False

    if device.type == 'cuda':
        print(f'tag={cli.tag!r}  device={device} ({torch.cuda.get_device_name(0)})')
        print(f'  VRAM total {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        print(f'tag={cli.tag!r}  device={device}')
    print('  profiles frozen, no training')
    print(f'  lambda grid: {len(cli.lambdas)} values, '
          f'{cli.lambdas[0]:.2g} .. {cli.lambdas[-1]:.2g}'
          f'{" (a probe pinned to either end is mis-regularised and gets flagged)" if len(cli.lambdas) > 1 else ""}')
    print()

    for pl in cli.pred_lens:
        t0 = time.time()
        args = build_args(pl)
        print(f'================ pred_len={pl} ================')

        m_t2d, p_t2d = load_model(args, 'Times2D', cli.tag)
        m_pf, p_pf = load_model(args, 'PhaseFormer', cli.tag)
        print(f'  checkpoints: {os.path.basename(os.path.dirname(p_t2d))}')
        print(f'               {os.path.basename(os.path.dirname(p_pf))}')
        m_t2d.to(device)
        m_pf.to(device)

        _, tr_loader = data_provider(args, 'train')
        _, va_loader = data_provider(args, 'val')
        _, te_loader = data_provider(args, 'test')

        A_tr, B_tr, Y_tr, MU_tr, SG_tr = extract(m_t2d, m_pf, tr_loader, device, cli.max_train or None)
        A_va, B_va, Y_va, MU_va, SG_va = extract(m_t2d, m_pf, va_loader, device)
        A_te, B_te, Y_te, MU_te, SG_te = extract(m_t2d, m_pf, te_loader, device)
        print(f'  rows: train={A_tr.shape[0]} val={A_va.shape[0]} test={A_te.shape[0]}')
        print(f'  dims: Times2D={A_tr.shape[1]}  PhaseFormer={B_tr.shape[1]}  '
              f'concat={A_tr.shape[1] + B_tr.shape[1]}')

        features = {
            'A: Times2D only': A_tr,
            'B: PhaseFormer only': B_tr,
            'C: concat (what CA sees)': torch.cat([A_tr, B_tr], dim=1),
        }
        val_features = {
            'A: Times2D only': A_va,
            'B: PhaseFormer only': B_va,
            'C: concat (what CA sees)': torch.cat([A_va, B_va], dim=1),
        }
        test_features = {
            'A: Times2D only': A_te,
            'B: PhaseFormer only': B_te,
            'C: concat (what CA sees)': torch.cat([A_te, B_te], dim=1),
        }
        if device.type == 'cuda':
            # The concat view at pred_len=720/seq_len=720 is ~1GB alone; report
            # what we actually reached so an OOM is diagnosable rather than cryptic.
            print(f'  VRAM peak after features: '
                  f'{torch.cuda.max_memory_allocated() / 1024**2:.0f} MB')

        A, B, C = 'A: Times2D only', 'B: PhaseFormer only', 'C: concat (what CA sees)'

        # ---- linear readout (ridge) ---------------------------------------
        test_mse, at_edge, prep = {}, {}, None
        if cli.probe in ('ridge', 'both'):
            print('\n  ridge readout  (lambda selected on VAL, scored on TEST)')
            print(f'  {"probe":<28}{"lambda":>9}{"val MSE":>10}{"TEST MSE":>10}')
            print('  ' + '-' * 70)
            for name in features:
                t_start = time.time()
                prep = ridge_prep(features[name], Y_tr)
                best = None
                for idx, lam in enumerate(cli.lambdas):
                    W = ridge_solve(prep, lam)
                    v = mse_original(ridge_apply(W, prep, val_features[name]), Y_va, MU_va, SG_va)
                    if best is None or v < best[0]:
                        best = (v, idx, W)
                v, idx, W = best
                lam = cli.lambdas[idx]
                # A lambda pinned to an END of the grid means the true optimum
                # lies outside it: the probe is mis-regularised and its TEST
                # number is not comparable to anything. That is what the first
                # run showed -- every probe chose the grid maximum 1e4, probe A
                # then failed to reproduce Times2D, and the table was junk.
                at_edge[name] = idx in (0, len(cli.lambdas) - 1)
                t = mse_original(ridge_apply(W, prep, test_features[name]), Y_te, MU_te, SG_te)
                test_mse[name] = t
                flag = '   <-- LAMBDA AT GRID EDGE' if at_edge[name] else ''
                print(f'  {name:<28}{lam:>9.3g}{v:>10.4f}{t:>10.4f}   '
                      f'({time.time()-t_start:.0f}s){flag}', flush=True)

        # ---- nonlinear readout (MLP) --------------------------------------
        mlp_val, mlp_test, mlp_cfg, mlp_edge = {}, {}, {}, {}
        if cli.probe in ('mlp', 'both'):
            print(f'\n  MLP readout  ({len(cli.mlp_wds)} weight decays x '
                  f'{len(cli.mlp_seeds)} seeds; config chosen on VAL, scored on TEST)')
            print(f'  {"probe":<28}{"wd":>9}{"seed":>6}{"val MSE":>10}{"TEST MSE":>10}')
            print('  ' + '-' * 68)
            for name in features:
                t_start = time.time()
                v, wd, seed, t, eps = mlp_probe(
                    features[name], Y_tr, val_features[name], Y_va, MU_va, SG_va,
                    test_features[name], Y_te, MU_te, SG_te,
                    cli.mlp_hidden, cli.mlp_wds, cli.mlp_seeds,
                    cli.mlp_epochs, cli.mlp_patience, cli.mlp_batch, device)
                mlp_val[name], mlp_test[name] = v, t
                mlp_cfg[name] = (wd, seed, eps)
                # Same gate as the ridge lambda: a winner at either end of the
                # grid means the true optimum is outside it and the probe is
                # mis-regularised, so its number is not comparable to anything.
                mlp_edge[name] = (len(cli.mlp_wds) > 1
                                  and wd in (cli.mlp_wds[0], cli.mlp_wds[-1]))
                flag = '   <-- WD AT GRID EDGE' if mlp_edge[name] else ''
                print(f'  {name:<28}{wd:>9.3g}{seed:>6}{v:>10.4f}{t:>10.4f}   '
                      f'({time.time()-t_start:.0f}s){flag}', flush=True)

        # Output-space oracle from the saved predictions, fitted on test.
        try:
            d_a = glob.glob(os.path.join(REPO, 'results',
                                         f'long_term_forecast_ETTh1_720_{pl}_Times2D_*_Exp_{cli.tag}'))[0]
            d_b = glob.glob(os.path.join(REPO, 'results',
                                         f'long_term_forecast_ETTh1_720_{pl}_PhaseFormer_*_Exp_{cli.tag}'))[0]
            pa = np.load(os.path.join(d_a, 'pred.npy')).ravel().astype(np.float64)
            pb = np.load(os.path.join(d_b, 'pred.npy')).ravel().astype(np.float64)
            yt = np.load(os.path.join(d_a, 'true.npy')).ravel().astype(np.float64)
            ra, rb = pa - yt, pb - yt
            m11, m22, m12 = float((ra * ra).mean()), float((rb * rb).mean()), float((ra * rb).mean())
            den = m11 - 2 * m12 + m22
            al = min(1.0, max(0.0, (m22 - m12) / den if den else 0.5))
            oracle = al * al * m11 + 2 * al * (1 - al) * m12 + (1 - al) ** 2 * m22
            print(f'\n  model outputs: Times2D={m11:.4f}  PhaseFormer={m22:.4f}'
                  f'  output-oracle={oracle:.4f} (alpha*={al:.3f}, fitted on TEST)')
            have_oracle = True
        except Exception as exc:
            print(f'\n  (could not compute the output oracle: {exc})')
            oracle, have_oracle = None, False

        # The MLP is the stronger test (nonlinear, and regularises the two
        # feature blocks independently), so it drives the verdict when present;
        # ridge is then context rather than the deciding number.
        use_mlp = bool(mlp_test)
        prim = mlp_test if use_mlp else test_mse
        kind = 'MLP' if use_mlp else 'ridge'
        c_off_grid = (mlp_edge.get(C, False) if use_mlp else at_edge.get(C, False))

        # ---- sanity check -------------------------------------------------
        # A probe on a branch's OWN features must roughly reproduce that branch's
        # output. If it cannot, the features or the denormalisation are wrong and
        # C's number means nothing either -- so this gates everything below.
        sanity_ok = True
        if have_oracle:
            print(f'\n  sanity ({kind}): probe on a branch\'s own features vs that branch\'s output')
            for lbl, pv, mv, nm in (('A', prim[A], m11, 'Times2D'),
                                    ('B', prim[B], m22, 'PhaseFormer')):
                rel = 100.0 * (pv - mv) / mv
                if rel > 10.0:
                    note = ('  <-- far worse than the model: probe underfit. Levels on '
                            'this horizon are not comparable.')
                    sanity_ok = False
                elif pv < mv - 0.005:
                    # Usually NOT leakage. Times2D's own readout is linear from
                    # these tokens and it was early-stopped (11-14 epochs under
                    # the reference protocol), whereas the probe refits that same
                    # readout on ~55k rows. A better readout beating a
                    # short-trained head is expected; a large gap is not.
                    note = '  <-- beats the model; expected if it was early-stopped, but check the gap'
                else:
                    note = '  OK'
                print(f'    probe {lbl} {pv:.4f}   vs   {nm} output {mv:.4f}   ({rel:+.1f}%){note}')

        best_single = min(prim[A], prim[B])
        gain_c = best_single - prim[C]
        print(f'\n  best single-feature probe = {best_single:.4f}   ({kind})')
        print(f'  C gain over best single   = {gain_c:+.4f}   (noise floor 0.0110)')
        if use_mlp and test_mse:
            print(f'  (ridge for reference: A={test_mse[A]:.4f} B={test_mse[B]:.4f} '
                  f'C={test_mse[C]:.4f})')
        if oracle is not None:
            print(f'  output-space oracle       = {oracle:.4f}   (TEST-fitted, but it has only')
            print(f'                                                 1 free parameter, so it is a')
            print(f'                                                 genuine upper bound)')

        print('\n  VERDICT: ', end='')
        if c_off_grid:
            knob = '--mlp-wds' if use_mlp else '--lambdas'
            which = 'weight decay' if use_mlp else 'lambda'
            print(f'INVALID -- the concat probe pinned its {which} to an END of the grid,')
            print('    so the optimum lies outside it and the probe is mis-regularised.')
            print('    Its TEST MSE is not comparable to anything. Widen '
                  f'{knob} and re-run.')
            if sanity_ok:
                print('    (For reference the sanity gate passed, so this is purely a')
                print('     hyperparameter-range problem, not a feature problem.)')
        elif not sanity_ok:
            print('INVALID -- the sanity check failed: a probe cannot reproduce its own')
            print('    branch\'s output on this horizon, so the levels are not comparable')
            print('    and C\'s number cannot be read. See the sanity lines above.')
        elif gain_c > 0.011:
            print(f'features carry information the outputs do not ({kind} readout).')
            print('    -> Cross-Attention has real headroom. Build it.')
        else:
            print(f'no complementarity visible in feature space ({kind} readout).')
            print('    A readout on the UNION of both branches\' features does not beat the')
            print('    better branch alone, so the second branch adds nothing the first did')
            print('    not already carry.')
            if sanity_ok:
                print('    With a nonlinear readout too, this closes the "ridge is only')
                print('    linear" and "one lambda cannot serve two blocks" objections.')
                print('    The one gap left is that these features are FROZEN -- a jointly')
                print('    trained model could in principle reshape them. See the header.')
            singles_off_grid = ((mlp_edge.get(A, False) or mlp_edge.get(B, False))
                                if use_mlp else
                                (at_edge.get(A, False) or at_edge.get(B, False)))
            if singles_off_grid:
                print('    CAVEAT: a single-branch probe also hit the grid edge, so at least')
                print('    one probe is still mis-regularised -- treat this as provisional.')
        print(f'\n  ({time.time() - t0:.0f}s)\n')

        # Release this horizon's tensors before the next one allocates its own;
        # the concat views are ~1GB each and VRAM is the binding constraint.
        del features, val_features, test_features, prep
        del A_tr, B_tr, Y_tr, MU_tr, SG_tr
        del A_va, B_va, Y_va, MU_va, SG_va
        del A_te, B_te, Y_te, MU_te, SG_te
        del m_t2d, m_pf
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    sys.exit(main())
