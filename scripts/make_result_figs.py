#!/usr/bin/env python3
"""Regenerate the two quantitative result figures from the sealed public CSVs."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results" / "paper"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.linewidth": 0.9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fig_detectability():
    decisions = [row for row in read_rows(RESULTS / "detectability" / "E4_4_frame_decisions.csv")
                 if 6 <= int(row["frame"]) <= 8]
    clean = np.asarray([float(row["b_t"]) for row in decisions if row["cond"] == "clean"])
    attack = np.asarray([float(row["b_t"]) for row in decisions if row["cond"] == "attack"])
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    clean_color, attack_color, dark = "#91AA9D", "#3E606F", "#193441"
    boxplot = ax.boxplot([clean, attack], positions=[1, 2], widths=0.5, showfliers=False,
                         patch_artist=True, medianprops=dict(color=dark, lw=1.7))
    for patch, color in zip(boxplot["boxes"], [clean_color, attack_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.42)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.3)
    rng = np.random.default_rng(0)
    for position, values, color in [(1, clean, clean_color), (2, attack, attack_color)]:
        jitter = position + rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(jitter, np.maximum(values, 0.3), s=13, color=color, alpha=0.45, linewidths=0)
    ax.axhline(8, color=dark, ls="--", lw=1.3)
    ax.text(0.62, 9.2, "threshold $b_t=8$", fontsize=12, va="bottom", ha="left", color=dark)
    ax.set_yscale("log")
    ax.set_ylim(0.25, 300)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["clean frames 6–8", "F&F burst 6–8"])
    ax.set_ylabel("candidate births per frame  $b_t$", fontsize=13)
    ax.set_xlim(0.5, 2.6)
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    save(fig, "fig_R1_detectability")


def fig_cg_full_results():
    series = [("c0", "clean", "clean"), ("c0", "A", "attack"),
              ("g3", "A", "attack+audit+BG"), ("e19_fi40", "A", "attack+audit+BG+CG")]
    colors = ["#D1DBBD", "#91AA9D", "#3E606F", "#193441"]
    specs = [("mot17_development", ["02", "04", "05"], "(a) train videos"),
             ("mot17_additional", ["09", "10", "11", "13"], "(b) test videos")]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.55), sharey=True)
    width = 0.19
    for ax, (experiment, videos, title) in zip(axes, specs):
        data = read_rows(RESULTS / experiment / "video_summary.csv")
        values = []
        for method, condition, _ in series:
            per_video = [100 * float(next(row["IDF1"] for row in data
                                         if row["method"] == method and row["condition"] == condition
                                         and row["video"].zfill(2) == video))
                         for video in videos]
            values.append(per_video + [float(np.mean(per_video))])
        assert np.isfinite(np.asarray(values)).all()
        x = np.arange(len(videos) + 1)
        for index, ((_, _, label), color) in enumerate(zip(series, colors)):
            offset = (index - (len(series) - 1) / 2) * width
            bars = ax.bar(x + offset, values[index], width, label=label,
                          color=color, edgecolor="#193441", linewidth=0.45)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                        f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=6.3)
        ax.set_xticks(x)
        ax.set_xticklabels(videos + ["Macro"])
        ax.set_xlabel(title, fontsize=11, labelpad=8)
        ax.set_ylim(0, 108)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("IDF1 (%) $\\uparrow$")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#193441", linewidth=0.45)
               for color in colors]
    fig.legend(handles, [item[2] for item in series], loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.04), handlelength=1.3,
               columnspacing=1.5, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig_CG_full_results")


if __name__ == "__main__":
    fig_detectability()
    fig_cg_full_results()
    print("ALL DONE ->", OUT)
