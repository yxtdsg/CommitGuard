# -*- coding: utf-8 -*-
"""
Generate 2 schematic (principle) figures: five-dim commit-gate and FN x FP synergy.
Vector schematics, no numeric data. Output PNG (300dpi) + PDF into figures/.
Run: python scripts/make_principle_figs.py
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = str(Path(__file__).resolve().parents[1] / "results" / "paper" / "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "figure.dpi": 300, "savefig.bbox": "tight"})

C_GATE = "#E0A33E"
C_DIM = "#4C9BD6"
C_FN = "#91AA9D"
C_FP = "#3E606F"
C_OK = "#3FA37C"


def box(ax, x, y, w, h, text, fc, ec="black", fs=10, tc="black", rounding=0.08):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={rounding}",
                       linewidth=1.1, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc)
    return b


def arrow(ax, p1, p2, color="black", lw=1.3, style="-|>", connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                        linewidth=lw, color=color, connectionstyle=connectionstyle)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Principle Fig 1: five-dimension decisions + commit gate on the recursion loop
# ---------------------------------------------------------------------------
def fig_commit_gate():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 5.6); ax.axis("off")

    # input
    box(ax, 0.2, 3.9, 1.7, 0.8, "Detections\n$D_t$", "#EAEAEA", fs=10)
    # candidate decisions
    box(ax, 2.5, 3.9, 2.0, 0.8, "Candidate\ndecisions", C_DIM, tc="white", fs=10)
    # commit gate
    box(ax, 5.1, 3.8, 1.5, 1.0, "Commit\nGate", C_GATE, tc="black", fs=11)
    # state
    box(ax, 7.3, 3.9, 1.9, 0.8, "Track state\n$S_t$", C_OK, tc="white", fs=10)

    arrow(ax, (1.9, 4.3), (2.5, 4.3))
    arrow(ax, (4.5, 4.3), (5.1, 4.3))
    arrow(ax, (6.6, 4.3), (7.3, 4.3))

    # five decision dims below candidate box
    dims = ["Birth", "Identity", "Kinematic", "Memory", "Survival"]
    x0 = 0.6
    for i, d in enumerate(dims):
        box(ax, x0 + i*1.95, 2.2, 1.75, 0.62, d, "#F2F7FC", ec=C_DIM, fs=9.5)
    ax.text(5.25, 3.05, "five decision dimensions audited before commit",
            ha="center", fontsize=9.5, color=C_DIM, style="italic")
    arrow(ax, (3.5, 3.9), (3.5, 2.85), color=C_DIM, lw=1.0, style="-")

    # recursion arrow from state back to candidate
    arrow(ax, (8.25, 3.9), (8.25, 1.2), color="gray", lw=1.2)
    arrow(ax, (8.25, 1.2), (3.5, 1.2), color="gray", lw=1.2)
    arrow(ax, (3.5, 1.2), (3.5, 3.9), color="gray", lw=1.2)
    ax.text(5.9, 0.85, "recursion: committed state feeds the next frame's decisions",
            ha="center", fontsize=9.5, color="gray", style="italic")

    ax.text(5.85, 4.95, "audit happens between deciding and writing",
            ha="center", fontsize=10, color=C_GATE)
    ax.set_title("Immune layer: a commit gate over five trajectory-decision dimensions",
                 fontsize=12, pad=12)
    fig.savefig(os.path.join(OUT, "fig_P1_commit_gate.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "fig_P1_commit_gate.pdf"))
    plt.close(fig)
    print("saved fig_P1_commit_gate")


# ---------------------------------------------------------------------------
# Principle Fig 2: FN x FP temporal synergy
# ---------------------------------------------------------------------------
def fig_fnfp_synergy():
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.9], "hspace": 0.18})
    frames = list(range(1, 13))
    attack = [6, 7, 8]

    # top: false negatives reduce detection recall
    recall = [0.97, 0.97, 0.96, 0.97, 0.96, 0.30, 0.28, 0.29, 0.55, 0.78, 0.93, 0.96]
    ax = axes[0]
    ax.axvspan(5.5, 8.5, color=C_FN, alpha=0.10)
    ax.plot(frames, recall, color=C_FN, lw=2.2, marker="o", ms=4)
    ax.set_ylabel("Detection Recall", color=C_FN)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelcolor=C_FN)
    ax.set_title("(a) False negatives: detection recall decreases during the attack", fontsize=10.5)
    ax.annotate("detection erasure", xy=(7, 0.3), xytext=(3.0, 0.45),
                arrowprops=dict(arrowstyle="->", color=C_FN), color=C_FN, fontsize=9.5)

    # middle: false positives create new tracks
    false_tracks = [1, 1, 2, 1, 1, 38, 41, 39, 6, 2, 1, 1]
    ax = axes[1]
    ax.axvspan(5.5, 8.5, color=C_FP, alpha=0.10)
    ax.bar(frames, false_tracks, color=C_FP, alpha=0.75, width=0.62)
    ax.set_ylabel("New false tracks", color=C_FP)
    ax.tick_params(axis="y", labelcolor=C_FP)
    ax.set_title("(b) False positives: false alarms create new tracks", fontsize=10.5)
    ax.annotate("false tracks created", xy=(7, 40), xytext=(2.4, 33),
                arrowprops=dict(arrowstyle="->", color=C_FP), color=C_FP, fontsize=9.5)

    # bottom: identity errors persist after the attack
    ax = axes[2]
    ax.axvspan(5.5, 8.5, color="#D1DBBD", alpha=0.16)
    ax.axvline(9, color="#193441", ls="--", lw=1.3)
    idsw = [0, 0, 0, 0, 0, 0, 0, 0, 9, 5, 2, 0]
    ax.bar(frames, idsw, color="#193441", width=0.55)
    ax.set_ylabel("ID switches")
    ax.set_xlabel("Frame")
    ax.set_title("(c) Identity switches increase after the attack", fontsize=10.5)
    ax.annotate("post-attack ID switches", xy=(9, 9), xytext=(9.4, 7.5),
                arrowprops=dict(arrowstyle="->", color="#193441"), color="#193441", fontsize=9.5)
    ax.set_xticks(frames)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle("F&F induces false negatives and false positives during frames 6–8",
                 fontsize=11, y=0.985)
    fig.savefig(os.path.join(OUT, "fig_P2_fnfp_synergy.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "fig_P2_fnfp_synergy.pdf"))
    plt.close(fig)
    print("saved fig_P2_fnfp_synergy")


def fig_cg_full_pipeline():
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(-0.15, 5.1); ax.axis("off")

    # Top: previous alarms causally activate full review before association.
    box(ax, 0.15, 3.55, 1.75, 0.82, "Alarm at $t-1$\nor recovery", C_GATE, fs=9)
    box(ax, 2.25, 3.55, 1.75, 0.82, "YOLOv7 at $t$", C_DIM, tc="white", fs=9.5)
    box(ax, 4.35, 3.55, 2.05, 0.82, "Auxiliary filtering\n+ FN recovery", "#F2F7FC", ec=C_DIM, fs=9)
    box(ax, 6.85, 3.55, 2.15, 0.82, "ByteTrack\nassociations", "#EAEAEA", fs=9.5)
    box(ax, 9.55, 3.55, 2.10, 0.82, "State update\nand output", C_OK, tc="white", fs=9.5)
    for x1, x2 in [(1.90, 2.25), (4.00, 4.35), (6.40, 6.85), (9.00, 9.55)]:
        arrow(ax, (x1, 3.96), (x2, 3.96), lw=1.0)
    ax.text(7.92, 3.30, "existing-track updates commit before $b_t$",
            ha="center", fontsize=8.3, color="black")

    # Bottom: the current alarm audits births after association but before creation.
    box(ax, 2.25, 1.60, 2.05, 0.82, "Unmatched birth\ncandidates", "#EAEAEA", fs=9)
    box(ax, 4.80, 1.60, 1.55, 0.82, "$b_t\\geq8$?", C_GATE, fs=10)
    box(ax, 6.85, 1.60, 2.15, 0.82, "YOLOv7 birth\nsupport check", C_DIM, tc="white", fs=9)
    box(ax, 9.55, 1.60, 2.10, 0.82, "Birth queue\n(no new ID yet)", "#F2F7FC", ec=C_DIM, fs=9)
    for x1, x2 in [(4.30, 4.80), (6.35, 6.85), (9.00, 9.55)]:
        arrow(ax, (x1, 2.01), (x2, 2.01), lw=1.0)
    arrow(ax, (7.92, 3.55), (3.55, 2.42), color="gray", lw=1.0,
          connectionstyle="arc3,rad=0.12")

    box(ax, 8.00, 0.02, 3.80, 0.70, "At $t+1$: confirm held candidates\nIoU-match unmatched reviewed detections", "#F7F7F7", ec="gray", fs=8.5)
    arrow(ax, (10.60, 1.60), (10.60, 0.72), color="gray", lw=1.0)
    ax.text(0.20, 4.72, "Full-list review (previous alarm)", fontsize=9.2, color="black", weight="bold")
    ax.text(0.20, 2.68, "Current-frame birth audit", fontsize=9.2, color="black", weight="bold")

    fig.savefig(os.path.join(OUT, "fig_CG_full_pipeline.png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(OUT, "fig_CG_full_pipeline.pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved fig_CG_full_pipeline")


if __name__ == "__main__":
    fig_cg_full_pipeline()
    fig_fnfp_synergy()
    print("PRINCIPLE FIGS DONE ->", OUT)
