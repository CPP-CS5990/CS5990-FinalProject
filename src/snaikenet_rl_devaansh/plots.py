from pathlib import Path

import matplotlib.pyplot as plt


def _rolling_avg(values: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


def save_session_plot(
    metrics: list[tuple[int, float, float]],
    start_update: int,
    end_update: int,
    agent_name: str,
    plots_dir: str = "plots",
    smooth_window: int = 50,
) -> str | None:
    if not metrics:
        return None

    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    updates   = [m[0] for m in metrics]
    avg_steps = [m[1] for m in metrics]
    avg_food  = [m[2] for m in metrics]

    smooth_steps = _rolling_avg(avg_steps, smooth_window)
    smooth_food  = _rolling_avg(avg_food,  smooth_window)

    color_steps = "#4C9BE8"
    color_food  = "#F4A261"

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Raw data — faint background
    ax1.plot(updates, avg_steps, color=color_steps, linewidth=0.5, alpha=0.25)
    # Smoothed trend — bold foreground
    ax1.plot(updates, smooth_steps, color=color_steps, linewidth=2.0, label=f"Steps survived (avg {smooth_window})")
    ax1.set_xlabel("PPO Update")
    ax1.set_ylabel("Avg Steps Survived", color=color_steps)
    ax1.tick_params(axis="y", labelcolor=color_steps)

    ax2 = ax1.twinx()
    ax2.plot(updates, avg_food, color=color_food, linewidth=0.5, alpha=0.25)
    ax2.plot(updates, smooth_food, color=color_food, linewidth=2.0, label=f"Food eaten (avg {smooth_window})")
    ax2.set_ylabel("Avg Food Eaten", color=color_food)
    ax2.tick_params(axis="y", labelcolor=color_food)

    fig.suptitle(f"{agent_name}  —  updates {start_update}–{end_update}", fontweight="bold")
    fig.tight_layout()

    out = f"{plots_dir}/{agent_name}_session_{start_update}_{end_update}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out