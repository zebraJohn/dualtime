"""Print the learned alpha / beta of Dual* checkpoints.

scripts/run_ca_experiment.sh:97 tells the reader to "inspect alpha and beta in
the logs", and docs/superpowers/specs/2026-09-27-dualfredn-design.md section
8.3 makes it a pre-registered diagnostic -- but nothing ever printed them. This
closes that gap, and works retroactively on the existing ca0/ca1 checkpoints.

The values are read straight out of the state dict, so no model needs to be
built and no hyperparameters need to be reconstructed from the setting name.
`ckpt_avg_k` averages state dicts rather than renaming them, so the keys
survive averaging unchanged.

Read: alpha and beta start at 0.5/0.5. If the optimiser leaves them near 0.5
(PhaseFormer's ca runs landed at alpha 0.44-0.48, beta 0.47-0.49) it is saying
the fused path is no more trustworthy than the branch predictions -- i.e. the
fusion is not earning its place.

Usage:
    python scripts/report_alpha_beta.py                       # every Dual* run
    python scripts/report_alpha_beta.py --glob 'ca1_s20*'
    python scripts/report_alpha_beta.py --glob '*DualFreDN*'
"""
import argparse
import glob
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# The setting name's field labels are SHIFTED BY ONE relative to their values:
# run.py's format string has 17 placeholders but 18 arguments are passed, so
# from `ft` onward every label carries the previous argument. Concretely
# `ll720` is seq_len, `pl48` is label_len, `dm192` is pred_len, `nh64` is
# d_model, and `eb1` is `factor`. Parse by label anyway -- the labels are stable
# even though they lie about what they hold.
FIELD_RE = re.compile(r'_ll(?P<seq_len>\d+)_pl(?P<label_len>\d+)_dm(?P<pred_len>\d+)'
                      r'_nh(?P<d_model>\d+)_el(?P<n_heads>\d+)_')
MODEL_RE = re.compile(r'_([A-Za-z][A-Za-z0-9]*)_ft')


def label_for(name):
    """A compact, readable label.

    The raw name is ~140 chars, and truncating it to fit dropped the --run_tag
    -- which is exactly what distinguishes seeds in a ca0/ca1 x 3-seed study.
    Parse the handful of fields that matter out instead.
    """
    tag = name.split('_Exp_', 1)[1] if '_Exp_' in name else '(untagged)'
    f = FIELD_RE.search(name)
    m = MODEL_RE.search(name)
    model = m.group(1) if m else '?'
    if not f:
        return f'{model}  tag={tag}'
    return (f'{model}  seq{f.group("seq_len")}  pred{f.group("pred_len")}  '
            f'tag={tag}')


def read_ab(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    a, b = sd.get('alpha'), sd.get('beta')
    if a is None or b is None:
        return None
    return float(a.reshape(-1)[0]), float(b.reshape(-1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoints', default=os.path.join(REPO, 'checkpoints'))
    ap.add_argument('--glob', default='*Dual*',
                    help="fnmatch on the setting directory name (default '*Dual*')")
    cli = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(cli.checkpoints, cli.glob))
                  if os.path.isdir(d))
    if not dirs:
        print(f'no checkpoint directories matched {cli.glob!r} under {cli.checkpoints}')
        return 1

    rows, skipped = [], 0
    for d in dirs:
        path = os.path.join(d, 'checkpoint.pth')
        if not os.path.exists(path):
            skipped += 1
            continue
        ab = read_ab(path)
        if ab is None:
            skipped += 1
            continue
        rows.append((os.path.basename(d), ab[0], ab[1]))

    if not rows:
        print(f'{len(dirs)} directories matched but none had alpha/beta '
              f'(skipped {skipped}) -- are these DualPhaseFormer/DualFreDN runs?')
        return 1

    print(f'{"run":<52} {"alpha":>7} {"beta":>7}')
    for name, a, b in rows:
        print(f'{label_for(name):<52} {a:7.4f} {b:7.4f}')

    if skipped:
        print(f'\n({skipped} matched directories skipped: no checkpoint.pth or no alpha/beta)')

    print('\nRead: alpha and beta start at 0.5/0.5. Values left near 0.5 mean the'
          '\noptimiser found the fused path no more trustworthy than the branch'
          '\npredictions -- the fusion is not earning its place (see PROGRESS.md 4.5).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
