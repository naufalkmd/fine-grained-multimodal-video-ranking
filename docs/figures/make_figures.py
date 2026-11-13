"""Draw the README figures, each in a light and a dark version (PNG, 200 dpi).

    python docs/figures/make_figures.py

task_*.png: what one episode looks like (placeholder frames: the organizers' data can't be published, and the
candidates are a shortened illustration in the style of the organizers' slide examples).
scores_*.png: the public leaderboard score at each step. The numbers are the public scores in the README and
docs/experiments/validation.md.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
# No hinting: text widths then scale exactly with resolution, so words placed by measuring them line up at any dpi
plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 10,
                     "text.hinting": "none", "text.hinting_factor": 1})

THEMES = {  # reference palette (dataviz skill), validated: slot 1 blue, slot 2 orange
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984", "grid": "#e4e3df",
              "text_only": "#2a78d6", "video": "#eb6834", "tile": "#e9e8e4", "tile_edge": "#d3d2cd",
              "mark": "#fde7b0", "card": "#f4f3f0"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#8f8e86", "grid": "#33332f",
             "text_only": "#3987e5", "video": "#d95926", "tile": "#2c2c29", "tile_edge": "#44443f",
             "mark": "#5a4613", "card": "#232321"},
}

# (label, public score, uses the video); top to bottom in the order we got there
STEPS = [
    ("Baseline and trained video models", (0.3469, 0.3968), True),
    ("Event-order check (SigLIP)", 0.4295, True),
    ("Pretrained SigLIP + centrality", 0.4682, True),
    ("Reversed familiarity", 0.5327, False),
    ("Centrality", 0.5515, False),
    ("Centrality, three measures combined", 0.5521, False),
    ("Take turns: centrality and familiarity", 0.5958, False),
    ("+ gate: which list goes first", 0.6368, False),
    ("+ own-words familiarity (final)", 0.6385, False),
]
LABELLED = {"Pretrained SigLIP + centrality", "Centrality", "Take turns: centrality and familiarity",
            "+ own-words familiarity (final)"}

# The organizers' slides show a reversed action and an added action; the other two are our negative types.
CANDIDATES = [
    ([("A person is ", 0), ("buttoning", 0), (" a shirt and then picks up a cup.", 0)], "correct"),
    ([("A person is ", 0), ("unbuttoning", 1), (" a shirt and then picks up a cup.", 0)], "reversed action"),
    ([("A person is buttoning a shirt, picks up a cup", 0), (" and then sits down", 1), (".", 0)], "added action"),
    ([("A person ", 0), ("picks up a cup and then is buttoning a shirt", 1), (".", 0)], "events reordered"),
    ([("A person is buttoning a ", 0), ("jacket", 1), (" and then picks up a cup.", 0)], "object swapped"),
]


def save(fig, name, theme):
    fig.savefig(OUT / f"{name}_{theme}.png", dpi=200, facecolor=THEMES[theme]["surface"], bbox_inches="tight",
                pad_inches=0.15)
    plt.close(fig)


def scores(theme):
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    fig.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    n = len(STEPS)
    for i, (label, value, video) in enumerate(STEPS):
        y = n - 1 - i
        color = c["video"] if video else c["text_only"]
        if isinstance(value, tuple):
            ax.plot(value, [y, y], color=color, lw=2, solid_capstyle="round", zorder=3)
            ax.scatter(value, [y, y], s=60, color=color, edgecolor=c["surface"], linewidth=2, zorder=4)
            ax.text(value[1] + 0.008, y, "0.35–0.40", va="center", color=c["text2"], fontsize=9)
        else:
            ax.scatter([value], [y], s=80, color=color, edgecolor=c["surface"], linewidth=2, zorder=4)
            if label in LABELLED:
                final = "final" in label
                ax.text(value + 0.008, y, f"{value:.4f}", va="center", color=c["text"] if final else c["text2"],
                        fontsize=9.5 if final else 9, fontweight="bold" if final else "normal")
    ax.set_yticks(range(n))
    ax.set_yticklabels([s[0] for s in reversed(STEPS)], color=c["text"])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(0.30, 0.70)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks([0.30, 0.40, 0.50, 0.60, 0.70])
    ax.tick_params(axis="x", colors=c["text2"], labelsize=9, length=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color=c["grid"], linewidth=0.8, zorder=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.axvline(0.375, color=c["muted"], linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.text(0.377, n - 0.45, "chance 0.375", color=c["text2"], fontsize=9, va="bottom")
    ax.set_xlabel("Public leaderboard score (573 test episodes)", color=c["text2"], fontsize=9, labelpad=8)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c["video"], label="uses the video"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c["text_only"], label="text only")]
    legend = ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, handletextpad=0.3)
    for t in legend.get_texts():
        t.set_color(c["text"])
    save(fig, "scores", theme)


def words(ax, x, y, parts, c, fontsize):
    """Draw a sentence piece by piece; marked pieces get a tinted box, so the one difference stands out."""
    renderer = ax.figure.canvas.get_renderer()

    def width(s):  # text extents ignore leading and trailing spaces, so spaces are measured and added by hand
        t = ax.text(0, 0, s, fontsize=fontsize)
        box = t.get_window_extent(renderer=renderer)
        t.remove()
        return ax.transData.inverted().transform((box.x1, 0))[0] - ax.transData.inverted().transform((box.x0, 0))[0]

    space = width("x x") - width("xx")
    for text, marked in parts:
        if text.startswith(" "):
            x += space
        t = ax.text(x, y, text.strip(), va="center", ha="left", fontsize=fontsize, color=c["text"],
                    fontweight="bold" if marked else "normal",
                    bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", facecolor=c["mark"], edgecolor="none")
                    if marked else None)
        box = t.get_window_extent(renderer=renderer)
        x = ax.transData.inverted().transform((box.x1, box.y0))[0]
        if text.endswith(" "):
            x += space


def task(theme):
    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    fig.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.0, 0.97, "Input: one short video as 8–20 frames", fontsize=10.5, fontweight="bold", color=c["text"],
            va="top")
    for k in range(8):
        x = 0.0 + k * 0.083
        ax.add_patch(FancyBboxPatch((x, 0.72), 0.072, 0.15, boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor=c["tile"], edgecolor=c["tile_edge"], linewidth=1))
        ax.text(x + 0.036, 0.795, str(k + 1), ha="center", va="center", fontsize=9, color=c["muted"])
    ax.annotate("", xy=(0.66, 0.69), xytext=(0.0, 0.69),
                arrowprops=dict(arrowstyle="-|>", color=c["muted"], lw=1.1))
    ax.text(0.67, 0.69, "time", va="center", fontsize=9, color=c["text2"])

    ax.text(0.0, 0.6, "Rank these five descriptions; exactly one is correct", fontsize=10.5, fontweight="bold",
            color=c["text"], va="top")
    for i, (parts, kind) in enumerate(CANDIDATES):
        y = 0.49 - i * 0.1
        ax.add_patch(FancyBboxPatch((0.0, y - 0.04), 1.0, 0.08, boxstyle="round,pad=0,rounding_size=0.015",
                                    facecolor=c["card"], edgecolor="none"))
        words(ax, 0.015, y, parts, c, 9.5)
        ax.text(0.985, y, kind, ha="right", va="center", fontsize=9,
                color=c["text"] if kind == "correct" else c["text2"],
                fontweight="bold" if kind == "correct" else "normal")
    ax.text(0.0, -0.02, "Illustration in the style of the organizers' examples; real candidates are about 60 words, "
            "and their order is shuffled.", fontsize=8.5, color=c["muted"], va="top")
    save(fig, "task", theme)


if __name__ == "__main__":
    for theme in THEMES:
        scores(theme)
        task(theme)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
