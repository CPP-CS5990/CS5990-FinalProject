import csv
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ARCHIVE = Path(__file__).parent
PLAYERS = ["Ameer", "David", "Devaansh", "Jack", "Sabrina"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load(path: Path) -> dict[str, list[int]]:
    ticks: list[int] = []
    kills: list[int] = []
    food: list[int] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            ticks.append(int(row["num_ticks_lived"]))
            kills.append(int(row["kills"]))
            food.append(int(row["food_eaten"]))
    return {"ticks": ticks, "kills": kills, "food": food}


def stats(data: dict[str, list[int]]) -> tuple[float, float, float, float, int, float]:
    ticks, kills, food = data["ticks"], data["kills"], data["food"]
    ticks_per_food = [n / fe for n, fe in zip(ticks, food) if fe > 0]
    return (
        sum(ticks) / len(ticks),
        sum(ticks_per_food) / len(ticks_per_food),
        sum(kills) / len(kills),
        sum(food) / len(food),
        max(food),
        median(food),
    )


def plot_distributions(raw: dict[str, dict[str, list[int]]]) -> None:
    panels = [
        ("Food eaten", "food"),
        ("Kills", "kills"),
        ("Ticks lived", "ticks"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, key) in zip(axes, panels):
        data = [raw[p][key] for p in PLAYERS]
        bp = ax.boxplot(
            data,
            tick_labels=PLAYERS,
            patch_artist=True,
            showfliers=True,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        )
        for patch, c in zip(bp["boxes"], COLORS):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Per-game distributions")
    handles = [Patch(facecolor=c, label=p) for p, c in zip(PLAYERS, COLORS)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.97))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = ARCHIVE / "distributions.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main() -> None:
    raw = {p: load(ARCHIVE / f"{p}.csv") for p in PLAYERS}
    results = {p: stats(raw[p]) for p in PLAYERS}

    metrics = [
        ("Avg ticks lived", 0),
        ("Avg ticks per food eaten (lower is better)", 1),
        ("Avg kills", 2),
        ("Avg food eaten per game", 3),
        ("Max food eaten", 4),
        ("Median food eaten", 5),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax in axes[len(metrics) :]:
        ax.axis("off")
    for ax, (title, idx) in zip(axes, metrics):
        values = [results[p][idx] for p in PLAYERS]
        bars = ax.bar(PLAYERS, values, color=COLORS)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Agent comparison across scoreboards")
    handles = [Patch(facecolor=c, label=p) for p, c in zip(PLAYERS, COLORS)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.97))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = ARCHIVE / "comparison.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    plot_distributions(raw)


if __name__ == "__main__":
    main()
