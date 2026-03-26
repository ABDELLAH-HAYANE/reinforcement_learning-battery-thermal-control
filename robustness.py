"""
robustness.py — Test de robustesse d'un agent SAC entraîné.

Évalue le checkpoint sur des configurations hors distribution (OOD) :
    baseline    : config d'entraînement standard
    hot_ambient : T_amb élevée (35–45°C)
    cold_ambient: T_amb basse  (0–10°C)
    high_load   : courant max doublé (I_max=80A)
    weak_cooling: puissance de refroidissement réduite de moitié

Usage :
    python robustness.py --checkpoint runs/day3_run/checkpoints/sac_final.pt
    python robustness.py --checkpoint runs/day6_run/checkpoints/sac_final.pt --n_episodes 20
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EnvConfig, ThermalConfig, RewardConfig
from envs.battery_thermal_env import BatteryThermalEnv
from evaluate import load_agent, run_episode


# ---------------------------------------------------------------------------
# Scénarios OOD
# ---------------------------------------------------------------------------

def _make_scenarios(base_tc: ThermalConfig) -> dict[str, ThermalConfig]:
    """
    Retourne un dict nom → ThermalConfig pour chaque scénario.
    Tous sont construits à partir des paramètres de base.
    """
    import copy

    def _clone(**overrides) -> ThermalConfig:
        tc = copy.copy(base_tc)
        for k, v in overrides.items():
            setattr(tc, k, v)
        return tc

    return {
        "Baseline":     _clone(),
        "Hot ambient":  _clone(T_amb_min=35.0, T_amb_max=45.0,
                               T_init_min=35.0, T_init_max=42.0),
        "Cold ambient": _clone(T_amb_min=0.0,  T_amb_max=10.0,
                               T_init_min=5.0,  T_init_max=15.0),
        "High load":    _clone(I_max=80.0, I_min=-20.0),
        "Weak cooling": _clone(P_cool_max=base_tc.P_cool_max * 0.5),
    }


# ---------------------------------------------------------------------------
# Évaluation d'un scénario
# ---------------------------------------------------------------------------

def _eval_scenario(
    agent,
    scenario_name: str,
    tc: ThermalConfig,
    n_episodes: int,
) -> dict:
    env_cfg = EnvConfig(thermal=tc, reward=RewardConfig())
    env     = BatteryThermalEnv(config=env_cfg)

    results = []
    for i in range(n_episodes):
        ep = run_episode(agent, env, seed=i)
        results.append(ep)

    returns   = [r["return"]   for r in results]
    pct_safes = [r["pct_safe"] for r in results]
    t_maxes   = [r["T_max"]    for r in results]

    return {
        "scenario":      scenario_name,
        "return_mean":   float(np.mean(returns)),
        "return_std":    float(np.std(returns)),
        "pct_safe_mean": float(np.mean(pct_safes)),
        "T_max_mean":    float(np.mean(t_maxes)),
        "results":       results,
    }


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict], baseline_return: float) -> None:
    header = (
        f"{'Scénario':<16}  {'Return':>10}  {'±':>7}  "
        f"{'pct_safe':>9}  {'T_max':>7}  {'Δ return':>9}"
    )
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)
    for r in rows:
        delta = r["return_mean"] - baseline_return
        delta_str = f"{delta:+.1f}" if r["scenario"] != "Baseline" else "   —"
        print(
            f"{r['scenario']:<16}  "
            f"{r['return_mean']:10.2f}  "
            f"{r['return_std']:7.2f}  "
            f"{r['pct_safe_mean']:9.1f}%  "
            f"{r['T_max_mean']:7.1f}°C  "
            f"{delta_str:>9}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_robustness(rows: list[dict], out_path: str) -> None:
    names     = [r["scenario"]      for r in rows]
    returns   = [r["return_mean"]   for r in rows]
    errs      = [r["return_std"]    for r in rows]
    pct_safes = [r["pct_safe_mean"] for r in rows]

    colors = ["tab:blue", "tab:red", "tab:cyan", "tab:orange", "tab:purple"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Return
    ax = axes[0]
    bars = ax.bar(names, returns, yerr=errs, color=colors[:len(names)],
                  capsize=5, alpha=0.85)
    ax.axhline(returns[0], color="black", linestyle="--", linewidth=1,
               label="Baseline", alpha=0.6)
    ax.set_title("Return moyen par scénario")
    ax.set_ylabel("Return")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, returns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(errs) * 0.05 + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    # % Safe
    ax = axes[1]
    bars = ax.bar(names, pct_safes, color=colors[:len(names)], alpha=0.85)
    ax.axhline(100, color="green", linestyle="--", linewidth=0.8)
    ax.set_title("% Steps en zone sûre")
    ax.set_ylabel("% steps")
    ax.set_ylim(0, 115)
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, pct_safes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Plot → {out_path}")


def _plot_trajectories(rows: list[dict], out_path: str, base_tc: ThermalConfig) -> None:
    """Trajectoire du meilleur épisode pour chaque scénario."""
    colors = ["tab:blue", "tab:red", "tab:cyan", "tab:orange", "tab:purple"]
    fig, ax = plt.subplots(figsize=(13, 5))

    for row, color in zip(rows, colors):
        best = max(row["results"], key=lambda r: r["return"])
        ax.plot(best["T_hist"], label=row["scenario"], color=color,
                linewidth=1.2, alpha=0.85)

    ax.axhline(base_tc.T_safe_min, color="blue",   linestyle="--", alpha=0.5,
               label=f"T_safe_min ({base_tc.T_safe_min}°C)")
    ax.axhline(base_tc.T_safe_max, color="orange", linestyle="--", alpha=0.5,
               label=f"T_safe_max ({base_tc.T_safe_max}°C)")
    ax.axhline(base_tc.T_cutoff,   color="red",    linestyle=":",  alpha=0.5,
               label=f"T_cutoff ({base_tc.T_cutoff}°C)")
    ax.fill_between(range(max(len(r["results"][0]["T_hist"]) for r in rows)),
                    base_tc.T_safe_min, base_tc.T_safe_max,
                    alpha=0.07, color="green")

    ax.set_title("Trajectoire température — meilleur épisode par scénario OOD")
    ax.set_xlabel("Step")
    ax.set_ylabel("Température (°C)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Plot → {out_path}")


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def robustness_test(args):
    base_tc   = ThermalConfig()
    scenarios = _make_scenarios(base_tc)

    env      = BatteryThermalEnv(config=EnvConfig(thermal=base_tc))
    obs_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.shape[0]
    agent    = load_agent(args.checkpoint, obs_dim, act_dim,
                          hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    print(f"\n=== Test de robustesse — {args.n_episodes} épisodes / scénario ===")
    print(f"Checkpoint : {args.checkpoint}\n")

    rows = []
    for name, tc in scenarios.items():
        print(f"  {name:<16} ...", end=" ", flush=True)
        m = _eval_scenario(agent, name, tc, args.n_episodes)
        rows.append(m)
        print(f"return={m['return_mean']:.1f}  pct_safe={m['pct_safe_mean']:.1f}%")

    baseline_return = rows[0]["return_mean"]
    _print_table(rows, baseline_return)

    # Sauvegarde JSON
    out_dir   = os.path.dirname(os.path.abspath(args.checkpoint))
    json_path = os.path.join(out_dir, "robustness_results.json")
    summary   = [{k: v for k, v in r.items() if k != "results"} for r in rows]
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats → {json_path}")

    # Plots
    if not args.no_plot:
        _plot_robustness(rows, os.path.join(out_dir, "robustness_metrics.png"))
        _plot_trajectories(rows, os.path.join(out_dir, "robustness_trajectories.png"), base_tc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Test de robustesse SAC")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers",   type=int, default=2)
    p.add_argument("--no_plot",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    robustness_test(args)
