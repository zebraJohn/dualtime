"""Step 2 of phase2d-v2.md: are the patch and phase branches' errors decorrelated?

WHY THIS IS THE DECIDING MEASUREMENT
Fusion can only help if the two branches make DIFFERENT mistakes. Which branch is
individually more accurate is secondary: if the phase branch is somewhat worse,
the optimal weight simply leans on the patch branch and most of the gain survives.
But if the two residual fields are near-perfectly correlated, no weighting helps --
the fused prediction is stuck at the accuracy of one branch.

That is exactly how v1 (Dual-Times2D) died: its "phase branch" was a Times2D with
a different period_list, i.e. the same architecture, so the residuals were almost
the same and the fusion bought nothing.

So the number to look at is rho = corr(r_patch, r_phase), not the MSE ranking.

WHAT IT COMPUTES, per horizon
  rho          residual correlation, flattened over (sample, time, channel)
  alpha*       optimal fixed blend weight on branch A, by grid search
  oracle       MSE at alpha*, i.e. the best a FIXED blend can do
  gain         min(MSE_A, MSE_B) - oracle   <- the headroom a fusion head could use

READING IT
  rho > ~0.95        the branches agree on nearly everything. Fusion is hopeless;
                     stop and reconsider the phase branch, do not build the dual.
  rho ~ 0.8-0.95     some complementarity. Whether it is usable depends on whether
                     `gain` clears the seed noise floor.
  rho < ~0.8         strongly decorrelated. Fusion has real headroom.

NOISE FLOOR: ETTh1-720 seed spread is ~0.011 MSE for the Times2D family. A single
seed per configuration cannot resolve a `gain` smaller than that. Use --seeds to
average several runs once they exist.

NOTE: `oracle` is an UPPER BOUND on what a fixed linear blend achieves, computed on
the TEST set. It is optimistically biased -- the weight is fitted on the same data
it is scored on. Treat it as headroom, never as a result to report.

`--tag` goes straight into a glob, so a pattern picks up every seed of a run:
    --tag refproto        one run
    --tag 'refproto_s20*' seeds 2021+2022+2023, analysed PER SEED

With more than one seed each cell is reported as mean (min..max), and the
per-horizon line says how many seeds had a positive gain. That spread is the
whole point: a gain of +0.008 that is +0.008 in all three seeds is a real effect;
a +0.008 that ranges from -0.004 to +0.021 is not.

Usage:
    python scripts/analyze_complementarity.py
    python scripts/analyze_complementarity.py --tag refproto
    python scripts/analyze_complementarity.py --tag 'refproto_s20*'
    python scripts/analyze_complementarity.py --model-a Times2D --model-b PhaseFormer --tag refproto
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

PRED_LENS = [96, 192, 336, 720]
# ETTh1-720 seed spread, from scripts/sweep_720.sh's own measurements.
NOISE_FLOOR = 0.011


def find_run_dir(pred_len, model, tag, data='ETTh1', seq_len=720):
    """Locate results/<setting>/ for one (pred_len, model, tag) combination.

    `data` and `seq_len` mirror run.py's --data and --seq_len. Both sit in the
    experiment setting name (`ft{data}` and `ll{seq_len}`), so the glob must
    match them exactly or nothing is found. They were hardcoded to ETTh1/720
    until the weather runs needed the same analysis.
    """
    prefix = f'long_term_forecast_{data}_{seq_len}_{pred_len}_{model}_'
    pattern = os.path.join(
        REPO, 'results',
        f'{prefix}*_Exp_{tag}' if tag else f'{prefix}*_Exp')
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    if len(hits) > 1:
        # Several seeds of the same config; caller averages them.
        return hits
    return hits[0]


def load_predictions(run_dir):
    pred = np.load(os.path.join(run_dir, 'pred.npy'))
    true = np.load(os.path.join(run_dir, 'true.npy'))
    return pred, true


def collect(pred_len, model, tag, data='ETTh1', seq_len=720):
    """Return a LIST of (pred, true) pairs -- one per seed -- or None.

    Deliberately NOT averaged across seeds here. Averaging predictions first and
    then measuring the residual correlation would quantify the complementarity
    of two ENSEMBLES, which is a different and easier question than whether the
    two branches are complementary at a single seed. The caller computes the
    statistics per seed and reports their spread, which is what distinguishes a
    robust effect from a one-seed artifact.

    `tag` is passed straight to glob, so a pattern like 'refproto_s20*' picks up
    all seeds of a run.
    """
    dirs = find_run_dir(pred_len, model, tag, data, seq_len)
    if dirs is None:
        return None
    if isinstance(dirs, str):
        dirs = [dirs]
    out = [load_predictions(d) for d in dirs]
    # Every seed must cover the same test samples for the comparison to be valid.
    shapes = {p.shape for p, _ in out} | {t.shape for _, t in out}
    if len(shapes) != 1:
        raise SystemExit(f'inconsistent shapes across seeds for {model} pl={pred_len}: {shapes}')
    return out


def analyze(pred_a, true, pred_b):
    """Residual statistics for two branches on one horizon.

    The optimal blend is solved in closed form, not by grid search: the test
    tensors here are ~13M elements at pred_len=720, and sweeping 2001 candidate
    weights over that costs minutes per horizon.

        f(a) = a^2*m11 + 2a(1-a)*m12 + (1-a)^2*m22
        a*   = (m22 - m12) / (m11 - 2*m12 + m22)
        f(a*)= (m11*m22 - m12^2) / (m11 + m22 - 2*m12)

    a* is clamped to [0, 1]: outside that range the optimum is to discard one
    branch entirely, and the clamped value is what a fixed blend could actually
    use. The denominator is 0 only when the two residuals are identical, in
    which case every a gives the same MSE and a*=0.5 is a valid representative.
    """
    r_a = (pred_a - true).ravel().astype(np.float64)
    r_b = (pred_b - true).ravel().astype(np.float64)

    m11 = float(np.mean(r_a * r_a))
    m22 = float(np.mean(r_b * r_b))
    m12 = float(np.mean(r_a * r_b))
    if r_a.std() == 0 or r_b.std() == 0:
        rho = float('nan')
    else:
        rho = float(np.corrcoef(r_a, r_b)[0, 1])

    den = m11 - 2 * m12 + m22
    alpha_raw = (m22 - m12) / den if den != 0 else 0.5
    alpha = min(1.0, max(0.0, alpha_raw))
    oracle = alpha * alpha * m11 + 2 * alpha * (1 - alpha) * m12 + (1 - alpha) ** 2 * m22

    return dict(m_a=m11, m_b=m22, rho=rho,
                alpha=alpha, alpha_unclamped=alpha_raw, oracle=float(oracle),
                naive_half=float(0.25 * m11 + 0.5 * m12 + 0.25 * m22))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-a', default='Times2D', help='branch A (patch view)')
    ap.add_argument('--model-b', default='PhaseFormer', help='branch B (phase view)')
    ap.add_argument('--tag', default='refproto', help="run_tag; '' for untagged runs")
    ap.add_argument('--horizons', type=int, nargs='+', default=PRED_LENS)
    ap.add_argument('--data', default='ETTh1',
                    help="the --data value the runs were made with; it is part of the "
                         "experiment setting name, so a mismatch finds nothing")
    ap.add_argument('--seq-len', type=int, default=720,
                    help="the --seq_len the runs were made with (also in the setting name)")
    ap.add_argument('--noise-floor', type=float, default=NOISE_FLOOR,
                    help="seed spread the fusion gain is judged against. The 0.011 default "
                         "is the ETTh1-720 measurement (scripts/sweep_720.sh); another "
                         "dataset has a different spread and it must be MEASURED, not "
                         "assumed -- running the same model at 2+ seeds gives it.")
    cli = ap.parse_args()

    tag_desc = cli.tag if cli.tag else '(untagged)'
    print(f'branch A = {cli.model_a}   branch B = {cli.model_b}   tag = {tag_desc}')
    print(f'dataset = {cli.data}  seq_len = {cli.seq_len}  horizons = {cli.horizons}')
    print(f'noise floor = {cli.noise_floor:.4f}'
          f'{"  (ETTh1-720 default)" if cli.noise_floor == NOISE_FLOOR else ""}\n')

    rows = []
    for pl in cli.horizons:
        a = collect(pl, cli.model_a, cli.tag, cli.data, cli.seq_len)
        b = collect(pl, cli.model_b, cli.tag, cli.data, cli.seq_len)
        if a is None or b is None:
            missing = cli.model_a if a is None else cli.model_b
            print(f'  pl={pl}: MISSING runs for {missing} (tag={tag_desc}) -- skipped')
            continue
        if len(a) != len(b):
            print(f'  pl={pl}: {len(a)} run(s) for {cli.model_a} but {len(b)} for '
                  f'{cli.model_b} -- seed counts must match, skipped')
            continue
        per, bad = [], False
        for i, ((pa, ta), (pb, tb)) in enumerate(zip(a, b)):
            if pa.shape != pb.shape or ta.shape != pa.shape:
                print(f'  pl={pl} seed#{i}: shape mismatch {pa.shape} vs {pb.shape} -- skipped')
                bad = True
                break
            per.append(analyze(pa, ta, pb))
        if bad or not per:
            continue
        rows.append((pl, per))

    if not rows:
        print('nothing to analyse.')
        return 1

    def cell(vals, plus=False):
        fmt = (lambda v: f'{v:+.4f}') if plus else (lambda v: f'{v:.4f}')
        m = sum(vals) / len(vals)
        if len(vals) == 1:
            return fmt(m)
        return f'{fmt(m)} ({fmt(min(vals))}..{fmt(max(vals))})'

    print(f'{"pred":>5} {"n":>3}  {"MSE_A":>19} {"MSE_B":>19} {"rho":>19} {"gain":>21}  consistent')
    print('-' * 108)
    all_gain, all_rho, alphas = [], [], []
    for pl, per in rows:
        ga = [r['m_a'] for r in per]
        gb = [r['m_b'] for r in per]
        gr = [r['rho'] for r in per]
        gg = [min(r['m_a'], r['m_b']) - r['oracle'] for r in per]
        all_gain += gg
        all_rho += gr
        alphas += [r['alpha'] for r in per]
        if len(per) > 1:
            consist = f'{sum(1 for g in gg if g > 0)}/{len(gg)} gain>0'
        else:
            consist = 'single seed'
        print(f'{pl:>5} {len(per):>3}  {cell(ga):>19} {cell(gb):>19} {cell(gr):>19} '
              f'{cell(gg, plus=True):>21}  {consist}')

    print('-' * 108)
    mean_gain = sum(all_gain) / len(all_gain)
    mean_rho = sum(all_rho) / len(all_rho)
    n_seeds = len(rows[0][1])
    print(f'mean rho = {mean_rho:.4f}    mean gain = {mean_gain:+.4f}    '
          f'noise floor = {cli.noise_floor:.4f}')
    if n_seeds > 1:
        print(f'{n_seeds} seeds per horizon; each cell is mean (min..max) over seeds.')
    else:
        print('SINGLE SEED -- the gain is a single sample, not an established effect.')
        print('The Times2D-family seed spread at ETTh1-720 is ~0.011, so treat any')
        print('gain below that as unresolved and re-run with multiple seeds.')

    degenerate = [a for a in alphas if a < 0.05 or a > 0.95]
    print(f'alpha* per horizon = {[round(a, 3) for a in alphas]}')
    if degenerate:
        print(f'  WARNING: alpha* hits an endpoint on {len(degenerate)}/{len(alphas)} horizons '
              f'-- the optimum is to ignore one branch entirely, i.e. no fusion.')

    print()
    print('=' * 84)
    if mean_rho > 0.95:
        print('VERDICT: the branches are near-duplicates (mean rho > 0.95).')
        print('  Fusion cannot help. This is the v1 failure mode -- stop and reconsider')
        print('  the phase branch before building any dual-branch machinery.')
    elif mean_gain > cli.noise_floor and not degenerate:
        print(f'VERDICT: genuine headroom. mean gain {mean_gain:+.4f} exceeds the '
              f'{cli.noise_floor:.4f} floor')
        print('  and the optimum uses BOTH branches (no alpha* at an endpoint).')
        print('  -> Step 3 (prediction-level fusion) is worth building.')
    else:
        print(f'VERDICT: no usable headroom (mean gain {mean_gain:+.4f}, '
              f'floor {cli.noise_floor:.4f}).')
        if n_seeds > 1:
            print(f'  Measured over {n_seeds} seeds per horizon, so this is NOT a')
            print('  single-seed artifact: the fixed-blend headroom is genuinely below')
            print('  the noise floor, and it holds across seeds.')
        else:
            print('  Single seed -- the effect may simply sit inside the seed noise.')
            print('  Add seeds before concluding.')
    print('=' * 84)
    print('\nReminder: `oracle` fits alpha on the TEST set and scores it there, so it is an')
    print('optimistic UPPER bound on a fixed blend. Never report it as a result.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
