# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DP load balance of the DEP decodes from a run's ATOM_DP_BALANCE_LOG files (#113).

Usage: python3 dp_balance.py RUN_DIR [--from-min 20 --to-min 60] [--out DIR]
Reads RUN_DIR/logs/slurm_job-*/dpb/*.jsonl (atom/utils/dp_balance_log.py), the
window in RUN_DIR/point.json and the benchmark's per-request records; prints the
numbers and writes summary.json plus PNGs into --out (default RUN_DIR/dpb_analysis).
Needs numpy and matplotlib only.
"""

import argparse
import datetime
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
]  # categorical slots 1-4, one per DP rank


def load(path):
    with open(path) as f:
        head = json.loads(f.readline())
        rows = []
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:  # a line cut by the container's stop
                pass
    return head, rows


def q(a, p):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else float("nan")


def dist(a):
    return {f"p{p}": round(q(a, p), 4) for p in (10, 50, 90, 99)} | {
        "mean": round(float(np.nanmean(a)), 4) if len(a) else None,
        "n": int(len(a)),
    }


def window(run):
    """(start, end) wall seconds of the benchmark window."""
    p = json.load(open(os.path.join(run, "point.json")))
    w = p.get("window") or {}
    if "start_ts" in w:
        return float(w["start_ts"]), float(w["end_ts"])
    day = json.load(open(os.path.join(run, "run.json")))["stamp"][:8]
    ts = [
        datetime.datetime.strptime(day + w[k], "%Y%m%d%H:%M:%S")
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
        for k in ("start", "end")
    ]
    if ts[1] < ts[0]:
        ts[1] += 86400
    return ts[0], ts[1]


def read_instance_files(dpb):
    """{host: {"engine": {rank: (head, rows)}, "runner": {...}, "mgr": [(head, rows)]}}"""
    out = defaultdict(lambda: {"engine": {}, "runner": {}, "mgr": []})
    for path in sorted(glob.glob(os.path.join(dpb, "*.jsonl"))):
        head, rows = load(path)
        inst = out[head["host"]]
        if head["role"] == "mgr":
            inst["mgr"].append((head, rows))
        else:
            # A restarted process (none expected) would leave two files: keep the longer one.
            old = inst[head["role"]].get(head["dp_rank"])
            if old is None or len(rows) > len(old[1]):
                inst[head["role"]][head["dp_rank"]] = (head, rows)
    return out


def table(head, rows, lo, hi):
    """Column dict of the rows with wall time in [lo, hi)."""
    f = head["fields"]
    off = head["wall_minus_mono"]
    rows = [r for r in rows if r and r[0] != "ep" and lo <= r[0] + off < hi]
    return {k: [r[i] for r in rows] for i, k in enumerate(f)} | {
        "wall": [r[0] + off for r in rows]
    }


def align_runner(runner, lo, hi):
    """Per rank runner columns, aligned on the forward index `n` (every rank runs each
    step's forward, real or dummy). The offset between ranks' counters is found from the
    step's DP-gathered token vector, the same on every rank."""
    ranks = sorted(runner)
    cols = {r: table(*runner[r], lo, hi) for r in ranks}
    ref = cols[ranks[0]]
    key = {n: tuple(t) for n, t in zip(ref["n"], ref["tok_per_rank"]) if t is not None}
    offsets = {ranks[0]: 0}
    for r in ranks[1:]:
        best = (-1, 0)
        sample = [
            (n, tuple(t)) for n, t in zip(cols[r]["n"], cols[r]["tok_per_rank"]) if t
        ][:2000]
        for d in range(-50, 51):
            hit = sum(key.get(n + d) == t for n, t in sample)
            best = max(best, (hit, d))
        offsets[r] = best[1]
        print(
            f"  rank {r}: forward-index offset {best[1]} vs rank {ranks[0]} ({best[0]}/{len(sample)} token vectors match)"
        )
    by_n = {r: {n + offsets[r]: i for i, n in enumerate(cols[r]["n"])} for r in ranks}
    common = sorted(set.intersection(*(set(m) for m in by_n.values())))
    idx = {r: [by_n[r][n] for n in common] for r in ranks}
    return ranks, cols, idx


def rung(ladder, bs):
    i = np.searchsorted(ladder, bs)
    return np.where(i < len(ladder), ladder[np.minimum(i, len(ladder) - 1)], bs)


def analyze(run, lo_min, hi_min, out):
    os.makedirs(out, exist_ok=True)
    w0, w1 = window(run)
    lo, hi = w0 + 60 * lo_min, w0 + 60 * hi_min
    dpb = glob.glob(os.path.join(run, "logs", "slurm_job-*", "dpb"))[0]
    cell = json.load(open(os.path.join(run, "cell.json")))
    ladder = np.array(sorted(json.loads(cell["service"]["decode"]["cudagraph"])))
    insts = read_instance_files(dpb)
    summary = {"run": run, "window_min": [lo_min, hi_min], "instances": {}}
    series_for_plot = {}
    for host, inst in sorted(insts.items()):
        if not inst["engine"]:
            continue
        print(f"== {host}")
        S = summary["instances"][host] = {}
        ranks, rc, idx = align_runner(inst["runner"], lo, hi)
        R = len(ranks)
        g = lambda k, dtype=float: np.array(  # noqa: E731
            [
                [rc[r][k][i] if rc[r][k][i] is not None else np.nan for i in idx[r]]
                for r in ranks
            ],
            dtype=dtype,
        )
        tok = np.array(
            [rc[ranks[0]]["tok_per_rank"][i] for i in idx[ranks[0]]], dtype=float
        ).T  # [R, steps]
        bs, ctx, dummy, meta, gpu, dt = (
            g("bs"),
            g("ctx_tok"),
            g("dummy"),
            g("meta_dt"),
            g("gpu_ms"),
            g("dt"),
        )
        run_bs, run_tok, prefill = g("run_bs")[0], g("run_tok")[0], g("prefill").max(0)
        t = np.array([rc[ranks[0]]["wall"][i] for i in idx[ranks[0]]])
        period = (
            np.diff(t, append=np.nan) * 1e3
        )  # ms from this forward's start to the next
        dec = (prefill == 0) & (tok.sum(0) > 0)
        steps = int(dec.sum())
        S["steps_decode"] = steps
        S["steps_with_prefill_or_idle"] = int((~dec).sum())
        mean_tok = tok.mean(0)
        S["sched_tokens_max_minus_min"] = dist((tok.max(0) - tok.min(0))[dec])
        S["sched_tokens_max_over_mean"] = dist((tok.max(0) / mean_tok)[dec])
        S["seqs_max_minus_min"] = dist((bs.max(0) - bs.min(0))[dec])
        S["seqs_max_over_mean"] = dist((bs.max(0) / np.maximum(bs.mean(0), 1e-9))[dec])
        S["ctx_tokens_max_over_mean"] = dist(
            (ctx.max(0) / np.maximum(ctx.mean(0), 1))[dec]
        )
        some_idle = (tok == 0).any(0) & (tok > 0).any(0)
        S["share_steps_a_rank_dummy_while_peer_works"] = round(
            float(some_idle[tok.sum(0) > 0].mean()), 4
        )
        S["dummy_steps_per_rank"] = {
            int(r): round(float(dummy[i][tok.sum(0) > 0].mean()), 4)
            for i, r in enumerate(ranks)
        }
        # Time: the CPU wait in the step's DP metadata all_gather, as a share of the step period.
        S["meta_wait_ms_per_rank"] = {
            int(r): dist(1e3 * meta[i][dec]) for i, r in enumerate(ranks)
        }
        S["meta_wait_share_of_period_per_rank"] = {
            int(r): round(
                float(np.nansum(1e3 * meta[i][dec]) / np.nansum(period[dec])), 4
            )
            for i, r in enumerate(ranks)
        }
        S["step_period_ms"] = dist(period[dec])
        # GPU: from this rank's launch to the forward's end. Ends line up (the last MoE combine
        # is collective), so max_r - own is the time this rank's GPU waited on a later launch.
        skew = np.nanmax(gpu, 0) - gpu
        S["gpu_ms_per_rank"] = {int(r): dist(gpu[i][dec]) for i, r in enumerate(ranks)}
        S["gpu_launch_wait_ms_per_rank"] = {
            int(r): dist(skew[i][dec]) for i, r in enumerate(ranks)
        }
        S["gpu_launch_wait_share_of_gpu"] = round(
            float(np.nansum(skew[:, dec]) / np.nansum(gpu[:, dec])), 4
        )
        # Padding: every rank runs the group's graph batch (the busiest rank's, on the ladder).
        q_rows = np.where(run_bs > 0, run_tok / np.maximum(run_bs, 1), 1)
        pad = 1 - bs.sum(0) / (R * run_bs)
        bal_bs = rung(ladder, np.ceil(bs.sum(0) / R))
        pad_bal = 1 - bs.sum(0) / (R * bal_bs)
        S["graph_rows_padding_share"] = round(float(np.nanmean(pad[dec])), 4)
        S["graph_rows_padding_share_if_balanced"] = round(
            float(np.nanmean(pad_bal[dec])), 4
        )
        S["graph_bs_actual_vs_balanced"] = {
            "actual": dist(run_bs[dec]),
            "balanced": dist(bal_bs[dec]),
        }
        # Cost model from this run's own steps: median step GPU time per graph batch.
        gmax = np.nanmax(gpu, 0)
        med = {}
        for b in np.unique(run_bs[dec]):
            sel = dec & (run_bs == b)
            if sel.sum() >= 50:
                med[int(b)] = float(np.nanmedian(gmax[sel]))
        ok = dec & np.isin(run_bs, list(med)) & np.isin(bal_bs, list(med))
        t_act = np.array([med.get(int(b), np.nan) for b in run_bs])
        t_bal = np.array([med.get(int(b), np.nan) for b in bal_bs])
        S["gpu_ms_median_by_graph_bs"] = med
        S["cost_graph_batch_share"] = round(
            float(np.nansum((t_act - t_bal)[ok]) / np.nansum(t_act[ok])), 4
        )
        S["cost_steps_covered"] = round(float(ok.sum() / max(steps, 1)), 4)
        # Attention: fit the step GPU time on (graph batch, the busiest rank's context tokens);
        # then its cost is the context coefficient times (max - mean) context.
        sel = ok & np.isfinite(gmax)
        X = np.column_stack([np.ones(sel.sum()), run_bs[sel], ctx.max(0)[sel] / 1e6])
        coef, *_ = np.linalg.lstsq(X, gmax[sel], rcond=None)
        res = gmax[sel] - X @ coef
        r2 = 1 - res.var() / gmax[sel].var()
        Xm = X.copy()
        Xm[:, 2] = ctx.mean(0)[sel] / 1e6
        coef_m, *_ = np.linalg.lstsq(Xm, gmax[sel], rcond=None)
        r2_m = 1 - (gmax[sel] - Xm @ coef_m).var() / gmax[sel].var()
        att = coef[2] * (ctx.max(0)[sel] - ctx.mean(0)[sel]) / 1e6
        S["fit_gpu_ms"] = {
            "coef_const_bs_ctxmax_per_Mtok": [round(float(c), 4) for c in coef],
            "r2_ctx_max": round(float(r2), 4),
            "r2_ctx_mean": round(float(r2_m), 4),
        }
        S["cost_context_imbalance_share"] = round(float(att.sum() / gmax[sel].sum()), 4)
        S["tokens_per_row"] = dist(q_rows[dec])
        # Engine: scheduler state per rank on a 100 ms grid.
        grid = np.arange(lo, hi, 0.1)
        ec = {r: table(*inst["engine"][r], lo, hi) for r in sorted(inst["engine"])}
        on = lambda c, k: np.interp(
            grid, c["wall"], np.array(c[k], dtype=float)
        )  # noqa: E731
        run_r = np.array([on(ec[r], "running") for r in ec])
        park = np.array([on(ec[r], "parked_kv") for r in ec])
        wait = np.array([on(ec[r], "waiting") for r in ec])
        kvu = np.array([on(ec[r], "kv_used") for r in ec])
        S["running_max_minus_min"] = dist(np.round(run_r.max(0) - run_r.min(0)))
        S["running_max_over_mean"] = dist(
            run_r.max(0) / np.maximum(run_r.mean(0), 1e-9)
        )
        S["running_mean_per_rank"] = {
            int(r): round(float(run_r[i].mean()), 3) for i, r in enumerate(ec)
        }
        S["parked_kv_mean_per_rank"] = {
            int(r): round(float(park[i].mean()), 3) for i, r in enumerate(ec)
        }
        S["waiting_mean_per_rank"] = {
            int(r): round(float(wait[i].mean()), 3) for i, r in enumerate(ec)
        }
        S["kv_used_max_over_mean"] = dist(kvu.max(0) / np.maximum(kvu.mean(0), 1))
        S["engine_sync_ms_per_rank"] = {
            int(r): dist(1e3 * np.array([x for x in ec[r]["sync_dt"] if x is not None]))
            for r in ec
        }
        S["gen_tokens_per_rank"] = {int(r): int(sum(ec[r]["gen_tok"])) for r in ec}
        # Requests the engine-core manager dispatched to each rank, with their prompt tokens.
        routed = defaultdict(lambda: [0, 0])
        for head, rows in inst["mgr"]:
            c = table(head, rows, lo, hi)
            for rk, pt in zip(c["rank"], c["prompt_tok"]):
                routed[rk][0] += 1
                routed[rk][1] += pt
        S["routed_reqs_prompt_tok_per_rank"] = {
            int(k): v for k, v in sorted(routed.items())
        }
        # MoE rows each EP rank receives (sum over the source ranks), per sampled layer.
        S["ep"] = ep_balance(inst["runner"], lo, hi)
        # TPOT driver: step period vs scheduled-token spread, in 1 s bins.
        sec = np.floor(t - lo).astype(int)
        spread_s, period_s = [], []
        for s_ in np.unique(sec[dec]):
            m = dec & (sec == s_)
            spread_s.append(float(np.mean(tok.max(0)[m] / mean_tok[m])))
            period_s.append(float(np.nanmedian(period[m])))
        S["corr_1s_spread_vs_period"] = round(
            float(np.corrcoef(spread_s, period_s)[0, 1]), 3
        )
        sp = tok.max(0) / mean_tok
        S["corr_step_spread_vs_period"] = round(
            float(
                np.corrcoef(
                    sp[dec & np.isfinite(period)], period[dec & np.isfinite(period)]
                )[0, 1]
            ),
            3,
        )
        b = np.digitize(sp, [1.1, 1.2, 1.35, 1.5, 1.75, 2.0])
        S["period_ms_by_spread_bin"] = {
            lab: dist(period[dec & (b == i)])["p50"]
            for i, lab in enumerate(
                ["<1.1", "1.1-1.2", "1.2-1.35", "1.35-1.5", "1.5-1.75", "1.75-2", ">=2"]
            )
        }
        series_for_plot[host] = (grid, run_r, t, tok, dec, period, sp, run_bs, bal_bs)
        for k, v in S.items():
            print(f"  {k}: {v}")
    summary["tpot"] = tpot_bins(run, lo, hi)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    plots(series_for_plot, lo, out)
    return summary


def ep_balance(runner, lo, hi):
    per_rank = {}
    for r, (head, rows) in runner.items():
        off = head["wall_minus_mono"]
        ep = [x for x in rows if x and x[0] == "ep"]
        if not ep:
            return None
        per_rank[r] = (
            ep[0][2],
            [(x[1] + off, np.array(x[3], dtype=float)) for x in ep],
        )
    layers = next(iter(per_rank.values()))[0]
    # Steady-window rows received per EP rank: (last - first) cumulative counts, per source rank.
    recv = None
    for r, (_, snaps) in per_rank.items():
        inside = [s for s in snaps if lo <= s[0] < hi]
        if len(inside) < 2:
            return None
        d = inside[-1][1] - inside[0][1]  # [layers, ep]
        recv = d if recv is None else recv + d
    ratio = recv.max(1) / recv.mean(1)
    # Per-interval imbalance, from the rank-0 snapshots' times (all ranks snapshot on the same forwards).
    inter = []
    snaps = {r: [s for s in v[1] if lo <= s[0] < hi] for r, v in per_rank.items()}
    n = min(len(s) for s in snaps.values())
    for i in range(1, n):
        d = sum(snaps[r][i][1] - snaps[r][i - 1][1] for r in snaps)
        m = d.mean(1)
        inter.extend((d.max(1) / np.where(m > 0, m, np.nan)).tolist())
    return {
        "layers": layers,
        "window_max_over_mean_per_layer": [round(float(x), 4) for x in ratio],
        "interval_max_over_mean": dist(inter),
        "share_received_per_ep_rank": [
            round(float(x), 4) for x in recv.sum(0) / recv.sum()
        ],
    }


def tpot_bins(run, lo, hi):
    """Per-request TPOT (decode duration / (output tokens - 1)) of requests ending in the window."""
    f = glob.glob(
        os.path.join(
            run,
            "logs",
            "slurm_job-*",
            "benchmark_results",
            "aiperf*",
            "profile_export.jsonl",
        )
    )
    if not f:
        return None
    tp = []
    for line in open(f[0]):
        d = json.loads(line)
        md, m = d["metadata"], d["metrics"]
        if md.get("benchmark_phase") != "profiling" or "decode_duration" not in m:
            continue
        end = md["request_end_ns"] / 1e9
        n = m.get("output_sequence_length", {}).get("value", 0)
        if lo <= end < hi and n > 1:
            tp.append(m["decode_duration"]["value"] / (n - 1))
    return {"tpot_ms": dist(tp)}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, length=0)


def plots(series, lo, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hosts = sorted(series)
    # 1. Running requests per DP rank, 1 s means, one panel per decode instance.
    fig, axs = plt.subplots(
        len(hosts), 1, figsize=(13, 3.6 * len(hosts)), facecolor=SURFACE, squeeze=False
    )
    for ax, h in zip(axs[:, 0], hosts):
        grid, run_r = series[h][0], series[h][1]
        k = 10
        n = len(grid) // k
        x = (grid[: n * k].reshape(n, k).mean(1) - lo) / 60 + 20
        for i in range(run_r.shape[0]):
            ax.plot(
                x,
                run_r[i, : n * k].reshape(n, k).mean(1),
                color=SERIES[i],
                lw=1.2,
                label=f"DP rank {i}",
            )
        _style(ax)
        ax.set_title(
            f"{h}: running requests per DP rank (1 s means)",
            loc="left",
            color=INK,
            fontsize=11,
        )
        ax.set_ylabel("running requests", color=INK2)
    axs[-1, 0].set_xlabel("minutes into the measured hour", color=INK2)
    axs[0, 0].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=4, frameon=False
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "running_per_rank.png"), dpi=110)
    plt.close(fig)
    # 2. Scheduled tokens per rank over a 5 s stretch: the step-level picture.
    fig, axs = plt.subplots(
        len(hosts), 1, figsize=(13, 3.6 * len(hosts)), facecolor=SURFACE, squeeze=False
    )
    for ax, h in zip(axs[:, 0], hosts):
        t, tok = series[h][2], series[h][3]
        sel = (t >= lo + 600) & (t < lo + 605)
        for i in range(tok.shape[0]):
            ax.plot(
                t[sel] - lo - 600,
                tok[i, sel],
                color=SERIES[i],
                lw=1,
                label=f"DP rank {i}",
            )
        _style(ax)
        ax.set_title(
            f"{h}: scheduled tokens per step and DP rank, 5 s, 10 min into the window",
            loc="left",
            color=INK,
            fontsize=11,
        )
        ax.set_ylabel("tokens in the step", color=INK2)
    axs[-1, 0].set_xlabel("seconds", color=INK2)
    axs[0, 0].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=4, frameon=False
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tokens_per_step_5s.png"), dpi=110)
    plt.close(fig)
    # 3. Distribution of the per-step max/mean of scheduled tokens (CDF, one line per instance).
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    for i, h in enumerate(hosts):
        sp, dec = series[h][6], series[h][4]
        v = np.sort(sp[dec])
        ax.plot(v, np.linspace(0, 1, len(v)), color=SERIES[i], lw=2, label=h)
    _style(ax)
    ax.set_xlim(1, 3)
    ax.set_title(
        "Per-step max / mean of scheduled tokens across the 4 DP ranks (CDF)",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.set_xlabel("max / mean", color=INK2)
    ax.set_ylabel("share of decode steps", color=INK2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "spread_cdf.png"), dpi=110)
    plt.close(fig)
    # 4. Step period by spread bin (median, IQR), one line per instance.
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    edges = [1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 3.0]
    for i, h in enumerate(hosts):
        sp, dec, period = series[h][6], series[h][4], series[h][5]
        xs, ys, lo_, hi_ = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = dec & (sp >= a) & (sp < b) & np.isfinite(period)
            if m.sum() >= 50:
                xs.append((a + b) / 2)
                ys.append(np.median(period[m]))
                lo_.append(np.percentile(period[m], 25))
                hi_.append(np.percentile(period[m], 75))
        ax.fill_between(xs, lo_, hi_, color=SERIES[i], alpha=0.15, lw=0)
        ax.plot(
            xs,
            ys,
            color=SERIES[i],
            lw=2,
            marker="o",
            ms=8,
            label=f"{h} (median, band = IQR)",
        )
    _style(ax)
    ax.set_title(
        "Step period vs per-step token spread across DP ranks",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.set_xlabel("max / mean of scheduled tokens in the step", color=INK2)
    ax.set_ylabel("step period, ms", color=INK2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "period_vs_spread.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--from-min", type=float, default=20)
    ap.add_argument("--to-min", type=float, default=60)
    ap.add_argument("--out")
    a = ap.parse_args()
    analyze(a.run, a.from_min, a.to_min, a.out or os.path.join(a.run, "dpb_analysis"))
