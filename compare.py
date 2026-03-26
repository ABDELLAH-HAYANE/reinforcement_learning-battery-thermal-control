"""
compare.py — Comparaison multi-agents sur BatteryThermalEnv.

Compare le SAC entraîné contre les baselines déterministes et aléatoire.

Usage :
    python compare.py --checkpoint runs/day3_run/checkpoints/sac_final.pt
    python compare.py --checkpoint runs/day3_run/checkpoints/sac_final.pt --n_episodes 20
    python compare.py --checkpoint runs/day3_run/checkpoints/sac_final.pt --plot
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EnvConfig, ThermalConfig, RewardConfig
from envs.battery_thermal_env import BatteryThermalEnv
from agent.sac import SACAgent, SACConfig
from evaluate import load_agent, run_episode
from baselines.rule_based import (
    RandomController,
    BangBangController,
    ProportionalController,
    HysteresisController,
)


# ---------------------------------------------------------------------------
# Évaluation d'un agent sur N épisodes
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env: BatteryThermalEnv, n_episodes: int, seed_offset: int = 0) -> dict:
    """
    Évalue n_episodes épisodes et retourne les statistiques agrégées.
    Réinitialise l'état interne des contrôleurs à état (HysteresisController).
    """
    results = []
    for i in range(n_episodes):
        # Réinitialise les contrôleurs à état
        if hasattr(agent, "reset"):
            agent.reset()
        ep = run_episode(agent, env, seed=seed_offset + i)
        results.append(ep)

    returns   = [r["return"]   for r in results]
    pct_safes = [r["pct_safe"] for r in results]
    t_maxes   = [r["T_max"]    for r in results]
    t_means   = [r["T_mean"]   for r in results]
    lengths   = [r["length"]   for r in results]

    return {
        "return_mean":   float(np.mean(returns)),
        "return_std":    float(np.std(returns)),
        "pct_safe_mean": float(np.mean(pct_safes)),
        "T_max_mean":    float(np.mean(t_maxes)),
        "T_mean_mean":   float(np.mean(t_means)),
        "length_mean":   float(np.mean(lengths)),
        "results":       results,
    }


# ---------------------------------------------------------------------------
# Affichage tableau
# ---------------------------------------------------------------------------

def print_table(rows: list[tuple[str, dict]]) -> None:
    header = f"{'Politique':<22}  {'Return':>10}  {'±':>7}  {'pct_safe':>9}  {'T_max':>7}  {'T_mean':>7}  {'len':>6}"
    sep    = "-" * len(header)
    print(f"\n{header}")
    print(sep)
    for name, m in rows:
        print(
            f"{name:<22}  "
            f"{m['return_mean']:10.2f}  "
            f"{m['return_std']:7.2f}  "
            f"{m['pct_safe_mean']:9.1f}%  "
            f"{m['T_max_mean']:7.1f}°C  "
            f"{m['T_mean_mean']:7.1f}°C  "
            f"{m['length_mean']:6.0f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Plots comparatifs
# ---------------------------------------------------------------------------

def plot_comparison(rows: list[tuple[str, dict]], out_path: str) -> None:
    """Barplots return et pct_safe, + trajectoire temperature pour chaque politique."""
    names   = [r[0] for r in rows]
    metrics = [r[1] for r in rows]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Return moyen
    ax = axes[0]
    returns = [m["return_mean"] for m in metrics]
    errs    = [m["return_std"]  for m in metrics]
    bars    = ax.bar(names, returns, yerr=errs, color=colors[:len(names)],
                     capsize=5, alpha=0.85)
    ax.set_title("Return moyen (± std)")
    ax.set_ylabel("Return")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, returns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errs)*0.05,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # % temps en zone sûre
    ax = axes[1]
    pct_safes = [m["pct_safe_mean"] for m in metrics]
    bars      = ax.bar(names, pct_safes, color=colors[:len(names)], alpha=0.85)
    ax.set_title("% Steps en zone thermique sûre")
    ax.set_ylabel("% steps")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="green", linewidth=0.8, linestyle="--", label="100%")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, pct_safes):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Plot comparaison → {out_path}")


def plot_trajectories(rows: list[tuple[str, dict]], env: BatteryThermalEnv, out_path: str) -> None:
    """Trace la trajectoire de température du meilleur épisode de chaque politique."""
    tc = env.tc
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=(13, 5))

    for (name, m), color in zip(rows, colors):
        # Meilleur épisode de la politique
        best = max(m["results"], key=lambda r: r["return"])
        steps = range(len(best["T_hist"]))
        ax.plot(steps, best["T_hist"], label=name, color=color, linewidth=1.2, alpha=0.85)

    ax.axhline(tc.T_safe_min, color="blue",   linestyle="--", alpha=0.5,
               label=f"T_safe_min ({tc.T_safe_min}°C)")
    ax.axhline(tc.T_safe_max, color="orange", linestyle="--", alpha=0.5,
               label=f"T_safe_max ({tc.T_safe_max}°C)")
    ax.axhline(tc.T_cutoff,   color="red",    linestyle=":",  alpha=0.5,
               label=f"T_cutoff ({tc.T_cutoff}°C)")
    ax.fill_between(range(max(len(m["results"][0]["T_hist"]) for _, m in rows)),
                    tc.T_safe_min, tc.T_safe_max, alpha=0.06, color="green")

    ax.set_title("Trajectoire température — meilleur épisode par politique")
    ax.set_xlabel("Step")
    ax.set_ylabel("Température (°C)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Plot trajectoires → {out_path}")


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def compare(args):
    env_cfg = EnvConfig(thermal=ThermalConfig(), reward=RewardConfig())
    env     = BatteryThermalEnv(config=env_cfg)
    tc      = env_cfg.thermal

    # --- Agents ---
    agents = []

    # SAC entraîné
    if args.checkpoint:
        sac_agent = load_agent(
            args.checkpoint,
            obs_dim    = env.observation_space.shape[0],
            action_dim = env.action_space.shape[0],
            hidden_dim = args.hidden_dim,
            n_layers   = args.n_layers,
        )
        agents.append(("SAC (entraîné)", sac_agent))

    agents += [
        ("Proportionnel",  ProportionalController(tc)),
        ("Bang-bang",      BangBangController(tc)),
        ("Hystérésis",     HysteresisController(tc)),
        ("Aléatoire",      RandomController(seed=42)),
    ]

    print(f"\n=== Comparaison — {args.n_episodes} épisodes par politique ===")
    if args.checkpoint:
        print(f"Checkpoint SAC : {args.checkpoint}")

    rows = []
    for name, agent in agents:
        print(f"  {name} ...", end=" ", flush=True)
        m = evaluate_agent(agent, env, n_episodes=args.n_episodes)
        print(f"return={m['return_mean']:.1f}  pct_safe={m['pct_safe_mean']:.1f}%")
        rows.append((name, m))

    print_table(rows)

    # --- Plots ---
    if args.plot:
        out_dir = os.path.dirname(os.path.abspath(args.checkpoint)) if args.checkpoint else "runs"
        plot_comparison(rows, os.path.join(out_dir, "comparison_metrics.png"))
        plot_trajectories(rows, env, os.path.join(out_dir, "comparison_trajectories.png"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Comparaison multi-politiques")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint SAC (.pt). Optionnel.")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers",   type=int, default=2)
    p.add_argument("--plot",       action="store_true",
                   help="Génère les graphiques comparatifs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare(args)
