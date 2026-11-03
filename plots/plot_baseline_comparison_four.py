#!/usr/bin/env python3
"""Compare fixed-only, naive reclamation and idealized recovery in Frontier.

Produces grouped P95 bars by arrival window, overall P50/P95/P99 bars,
and a CSV summary. No SLO charts.

python plot_baseline_comparison_bars.py \
 --fixed outputs/malleserve/runs/baseline_fixed4_r10 \
 --fixed8 outputs/malleserve/runs/baseline_fixed8_r10 \
 --naive outputs/malleserve/runs/baseline_naive_r10 \
 --ideal outputs/malleserve/runs/baseline_ideal_r10 \
 --output outputs/malleserve/figures/baseline_comparison
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'Fixed 4 GPUs': '#607086', 'Naive': '#223C61', 'Ideal recovery': '#19A398', 'Fixed 8 GPUs': '#B37D36'}
INK, MUTED, GRID = '#18253A', '#607086', '#E8EDF3'


def load(path: Path):
    summary = json.loads((path / 'malleserve_summary.json').read_text())
    arrivals, completions = {}, {}
    with (path / 'metrics_ground_truth.jsonl').open() as fh:
        for line in fh:
            if not line.strip():
                continue
            event = json.loads(line)
            rid = str(event.get('request_id'))
            if event.get('event_type') == 'request_arrival':
                if rid in arrivals:
                    raise ValueError(f'Duplicate arrival {rid} in {path}')
                arrivals[rid] = float(event['arrived_at'])
            elif event.get('event_type') == 'request_completion':
                if rid in completions:
                    raise ValueError(f'Duplicate completion {rid} in {path}')
                completions[rid] = float(event['completed_at'])
    lifecycle = path / 'transient_lifecycle.json'
    life = json.loads(lifecycle.read_text()) if lifecycle.exists() else {}
    return summary, arrivals, completions, life


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })


def axes_style(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DCE3EA')
    ax.tick_params(axis='both', length=0, colors=MUTED)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.png', '.pdf'):
        fig.savefig(path.with_suffix(extension), dpi=300 if extension == '.png' else None,
                    bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    print('Saved', path.with_suffix('.png'), 'and', path.with_suffix('.pdf'))


def percentile(values, p):
    return float(np.percentile(values, p)) if values else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixed', 'naive', 'ideal', 'fixed8'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-seconds', type=float, default=20)
    args = parser.parse_args()
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        parser.error('--window-seconds must be positive and finite')
    style()

    paths = {'Fixed 4 GPUs': args.fixed, 'Naive': args.naive, 'Ideal recovery': args.ideal, 'Fixed 8 GPUs': args.fixed8}
    runs = {name: load(path) for name, path in paths.items()}
    periods = [tuple(map(float, r[0]['measurement_window_s'])) for r in runs.values()]
    if len(set(periods)) != 1:
        raise ValueError(f'Mismatched measurement windows: {periods}')
    start, end = periods[0]
    reference = sorted(runs['Fixed 4 GPUs'][1].values())
    for name, (summary, arrivals, completions, _) in runs.items():
        if len(arrivals) != len(reference) or not np.allclose(sorted(arrivals.values()), reference, atol=1e-8, rtol=0):
            raise ValueError(f'Arrival traces differ: {name}')
        missing = sum(start <= t < end and rid not in completions for rid, t in arrivals.items())
        if missing:
            raise ValueError(f'{name}: {missing} measured arrivals lack completion events; cannot compare unbiased P95')
        if summary.get('measured_no_completion_event'):
            raise ValueError(f'{name}: summary reports unfinished requests')

    edges = [start]
    while edges[-1] < end - 1e-9:
        edges.append(min(end, edges[-1] + args.window_seconds))
    intervals = list(zip(edges[:-1], edges[1:]))
    names = list(paths)
    window_metrics = {}
    overall_metrics = {}
    summaries = []
    for name, (summary, arrivals, completions, life) in runs.items():
        measured = [completions[rid] - t for rid, t in arrivals.items() if start <= t < end]
        if any(not math.isfinite(lat) or lat < 0 for lat in measured):
            raise ValueError(f'Bad request latency for {name}')
        overall_metrics[name] = [percentile(measured, q) for q in (50, 95, 99)]
        window_metrics[name] = []
        for lo, hi in intervals:
            latencies = [completions[rid] - t for rid, t in arrivals.items() if lo <= t < hi]
            window_metrics[name].append(percentile(latencies, 95))
        summaries.append({
            'policy': name, 'measured_arrivals': len(measured),
            'completed': len(measured),
            'p50_s': overall_metrics[name][0],
            'p95_s': overall_metrics[name][1],
            'p99_s': overall_metrics[name][2],
            'retries': life.get('retries', 0),
            'lost_processed_tokens': life.get('lost_processed_tokens', 0),
            'oracle_restored_requests': life.get('oracle_restored_requests', 0),
            'oracle_restored_tokens': life.get('oracle_restored_tokens', 0),
        })

    # Grouped bars: windows, comparing policies side-by-side.
    n = len(intervals)
    fig_width = max(10.0, .72 * n + 2.6)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    fig.subplots_adjust(left=.09, right=.985, top=.81, bottom=.22)
    x = np.arange(n)
    width = .20
    for index, name in enumerate(names):
        ax.bar(x + (index - (len(names)-1)/2) * width, window_metrics[name], width,
               color=COLORS[name], label=name, zorder=3)
    # ax.set_title('P95 latency by request-arrival window | fixed vs transient',
    #              fontsize=11.5, fontweight='bold', color=INK, pad=13)
    ax.set_xticks(x, [f'{lo:g}–{hi:g}' for lo, hi in intervals], rotation=35, ha='right')
    ax.set_ylabel('P95 latency (s)', color=INK)
    ax.set_xlabel('Request arrival time (s)', color=INK, labelpad=8)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=4, fontsize=8.8, loc='lower left',
              bbox_to_anchor=(0, 1.005))
    axes_style(ax)
    save(fig, args.output.with_name(args.output.name + '_p95_windows'))

    # Compact overall latency percentile chart (not an SLO figure).
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    fig.subplots_adjust(left=.12, right=.98, top=.80, bottom=.20)
    x = np.arange(3)
    width = .20
    all_heights = []
    for index, name in enumerate(names):
        values = overall_metrics[name]
        all_heights.extend(values)
        bars = ax.bar(x + (index - (len(names)-1)/2) * width, values, width,
                      color=COLORS[name], label=name, zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.1f}', (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 4), textcoords='offset points', ha='center',
                        va='bottom', fontsize=8.5, color=INK)
    ax.set_ylim(0, max(all_heights) * 1.24 if all_heights else 1)
    ax.set_xticks(x, ('P50', 'P95', 'P99'))
    ax.set_ylabel('Request latency (s)', color=INK)
    # ax.set_title('Request latency | fixed vs transient',
    #              fontsize=11.5, fontweight='bold', color=INK, pad=13)
    ax.legend(frameon=False, ncol=4, fontsize=8.8, loc='lower left',
              bbox_to_anchor=(0, 1.005))
    axes_style(ax)
    save(fig, args.output.with_name(args.output.name + '_latency_summary'))

    out = args.output.with_name(args.output.name + '_summary.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print('Saved', out)

if __name__ == '__main__':
    main()
