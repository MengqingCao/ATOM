#!/usr/bin/env python3
"""Decode-step MegaMoE share from steps/dp.* (one line a step: per-rank scheduled tokens, any-prefill, padded bs,
running_tokens). Counts only all-decode steps with work (sum>0); 'mega' = padded group rows (ranks x running_tokens) >= 256.
Steady part = steps between 25% and 95% of the working steps (no timestamps in the log). usage: stepshare.py RUN_DIR..."""
import glob, os, sys
for run in sys.argv[1:]:
    f = sorted(glob.glob(run + "/slurm_job-*/steps/dp.*"), key=os.path.getsize)[-1]
    dec = []; pre = 0
    for l in open(f):
        p = l.split()
        if len(p) != 4: continue
        t = [int(x) for x in p[0].split(",")]
        if sum(t) == 0: continue
        if p[1] != "0": pre += 1; continue
        dec.append((t, int(p[3])))
    st = dec[len(dec)//4: len(dec)*95//100]
    m = lambda L: sum(1 for t, rt in L if len(t)*rt >= 256) / len(L)
    rows = sorted(sum(t) for t, _ in st); pad = sorted(len(t)*rt for t, rt in st)
    q = lambda L, p: L[int(p*(len(L)-1))]
    print(f"{run.split('/')[-1][:12]:12} work_decode_steps={len(dec)} prefill_steps={pre} mega_all={m(dec):.3f} mega_steady={m(st):.3f} "
          f"true_rows p10/50/90={q(rows,.1)}/{q(rows,.5)}/{q(rows,.9)} padded p10/50/90={q(pad,.1)}/{q(pad,.5)}/{q(pad,.9)}")
