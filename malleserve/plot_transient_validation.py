#!/usr/bin/env python3
"""Polaris transient capacity setup + Frontier validation charts, without slide-sized titles.

Usage:
  python plot_transient_validation.py \
    --csv outputs/malleserve/validation/one_reclaim_aligned.csv \
    --output outputs/malleserve/figures/transient_validation

Produces *_timeline, *_p95, *_metrics (PNG at 300 dpi and vector PDF).
CSV must be the aligned output of malleserve.tools.compare_one_reclaim.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAVY = '#223C61'
TEAL = '#19A398'
ORANGE = '#C26D20'
INK = '#18253A'
MUTED = '#607086'
GRID = '#E8EDF3'
WINDOWS = ((30, 180), (180, 240), (240, 299), (299, 330))


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.12)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    print('Saved', path.with_suffix('.png'), 'and', path.with_suffix('.pdf'))


def read_pairs(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    by_key = {}
    for row in rows:
        key = (row.get('system', '').lower(), int(row['arrival_start_s']), int(row['arrival_end_s']))
        if key in by_key:
            raise ValueError(f'Duplicate comparison row: {key}')
        by_key[key] = row
    pairs = []
    for start, end in WINDOWS:
        try:
            frontier = by_key['frontier', start, end]
            polaris = by_key['polaris', start, end]
        except KeyError as exc:
            raise ValueError(f'Missing aligned window {start}–{end}s. Re-run comparator with --aligned-windows.') from exc
        for system, row in (('Frontier', frontier), ('Polaris', polaris)):
            arrived = int(row['arrived'])
            completed = int(row['completed_eventually'])
            missing = int(row['no_completion_event'])
            if arrived == 0 or completed != arrived or missing:
                raise ValueError(f'{system} {start}–{end}s has incomplete requests: {row}')
        pairs.append((frontier, polaris))
    return pairs


def plot_timeline(output):
    fig, ax = plt.subplots(figsize=(7.1, 3.3))
    fig.subplots_adjust(left=.12, right=.98, top=.87, bottom=.22)
    ax.fill_between([0, 330], [0, 0], [4, 4], color=NAVY, alpha=.10, step='post')
    ax.fill_between([0, 180, 299, 330], [4, 4, 4, 4], [8, 8, 4, 8],
                    step='post', color=TEAL, alpha=.19)
    ax.step([0, 180, 299, 330], [8, 4, 8, 8], where='post', color=TEAL, lw=2.8)
    ax.hlines(4, 0, 330, color=NAVY, lw=1.8, linestyle='--', alpha=.8)
    for x, label, color in ((180, 'Reclaimed', ORANGE), (299, 'Ready again', TEAL)):
        ax.axvline(x, color=color, ls=(0, (4, 3)), lw=1.15)
        ax.annotate(label, (x, 8.35), ha='center', va='bottom', color=color,
                    fontsize=9.5, fontweight='semibold')
    ax.annotate('4 persistent GPUs', (75, 2.1), ha='center', color=NAVY, fontsize=10,
                fontweight='semibold')
    ax.annotate('4 transient GPUs', (83, 5.95), ha='center', color=TEAL,
                fontsize=10, fontweight='semibold')
    ax.set_xlim(0, 330)
    ax.set_ylim(0, 9.5)
    ax.set_yticks([0, 4, 8])
    ax.set_xticks([0, 30, 60, 120, 180, 240, 299, 330])
    ax.set_ylabel('Available GPUs', color=INK)
    ax.set_xlabel('Experiment time (s)', color=INK)
    ax.set_title('Transient capacity | 8 → 4 → 8 GPUs | 10 RPS', fontsize=11.5,
                 fontweight='bold', color=INK, pad=12)
    ax.yaxis.grid(True, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=0, colors=MUTED)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DCE3EA')
    save(fig, output.with_name(output.name + '_timeline'))


def plot_p95(pairs, output):
    real = np.array([float(p['p95_s']) for _, p in pairs])
    sim = np.array([float(f['p95_s']) for f, _ in pairs])
    labels = [f'{a}–{b}s' for a, b in WINDOWS]
    x = np.arange(len(WINDOWS))
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    fig.subplots_adjust(left=.12, right=.98, top=.79, bottom=.20)
    width = .33
    first = ax.bar(x - width/2, real, width, color=NAVY, label='Polaris (measured)', zorder=3)
    second = ax.bar(x + width/2, sim, width, color=TEAL, label='Frontier (simulated)', zorder=3)
    for bars in (first, second):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', (bar.get_x()+bar.get_width()/2, h),
                        xytext=(0, 4), textcoords='offset points', ha='center',
                        va='bottom', fontsize=9.2, fontweight='semibold', color=INK)
    ax.set_ylim(0, max(real.max(), sim.max())*1.25)
    ax.set_xticks(x, labels)
    ax.set_ylabel('P95 request latency (s)', color=INK)
    ax.set_title('Polaris vs Frontier | Transient 8 → 4 → 8 GPUs | 10 RPS',
                 fontsize=11.1, fontweight='bold', color=INK, pad=12)
    ax.legend(
              frameon=False, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=0, colors=MUTED)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DCE3EA')
    save(fig, output.with_name(output.name + '_p95'))


def plot_metrics(pairs, output):
    labels = [f'{a}–{b}s' for a, b in WINDOWS]
    metrics = [('Mean', 'mean_s'), ('P50', 'p50_s'), ('P95', 'p95_s'), ('P99', 'p99_s')]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.9), sharex=True)
    fig.subplots_adjust(left=.09, right=.98, top=.83, bottom=.12, hspace=.38, wspace=.30)
    for ax, (title, key) in zip(axes.flat, metrics):
        p = np.array([float(pol[key]) for _, pol in pairs])
        f = np.array([float(fr[key]) for fr, _ in pairs])
        x = np.arange(len(labels))
        ax.plot(x, p, 'o-', color=NAVY, lw=2, ms=5, label='Polaris')
        ax.plot(x, f, 's-', color=TEAL, lw=2, ms=5, label='Simulated')
        ax.set_title(title, loc='left', color=INK, fontweight='semibold', fontsize=11)
        ax.set_ylabel('Latency (s)', color=MUTED)
        ax.set_ylim(bottom=0)
        ax.set_xticks(x, labels)
        ax.yaxis.grid(True, color=GRID, lw=.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=0, colors=MUTED, labelbottom=True)
        ax.tick_params(axis='x', labelsize=8)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DCE3EA')
    fig.suptitle('Polaris vs Frontier | Transient 8 → 4 → 8 GPUs | 10 RPS',
                 x=.09, y=.97, ha='left', fontsize=12, fontweight='bold', color=INK)


    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.6,
    )



    save(fig, output.with_name(output.name + '_metrics'))
    


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('outputs/malleserve/validation/one_reclaim_aligned.csv'),
                        help='Aligned comparison CSV from compare_one_reclaim --aligned-windows')
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/malleserve/figures/transient_validation'))
    parser.add_argument('--timeline-only', action='store_true')
    args = parser.parse_args()
    style()
    plot_timeline(args.output)
    if not args.timeline_only:
        pairs = read_pairs(args.csv)
        plot_p95(pairs, args.output)
        plot_metrics(pairs, args.output)


if __name__ == '__main__':
    main()
