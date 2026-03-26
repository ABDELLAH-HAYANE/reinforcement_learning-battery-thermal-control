"""
evaluate.py — Évaluation déterministe d'un checkpoint SAC.

Usage :
    python evaluate.py --checkpoint runs/day3_run/checkpoints/sac_final.pt
    python evaluate.py --checkpoint runs/exp1/checkpoints/sac_ep100.pt --n_episodes 20
    python evaluate.py --checkpoint runs/day3_run/checkpoints/sac_final.pt --no_plot

Résultats sauvegardés dans le même dossier que le checkpoint :
    eval_results.json      — métriques agrégées
    eval_best_episode.png  — trajectoire du meilleur épisode
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
from agent.sac import SACAgent, SACConfig


# ---------------------------------------------------------------------------
# Chargement agent
# ---------------------------------------------------------------------------

def load_agent(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    n_layers: int = 2,
) -> SACAgent:
    """Charge un checkpoint SAC et retourne l'agent prêt à évaluer."""
    cfg = SACConfig(hidden_dim=hidden_dim, n_layers=n_layers)
    agent = SACAgent(obs_dim, action_dim, config=cfg)
    agent.load(checkpoint_path)
    return agent


# ---------------------------------------------------------------------------
# Rouler un épisode
# ---------------------------------------------------------------------------

def run_episode(
    agent: SACAgent,
    env: BatteryThermalEnv,
    seed: int = None,
) -> dict:
    """
    Exécute un épisode complet en mode déterministe.

    Returns :
        dict avec métriques scalaires + historiques complets (T, SoC, action)
    """
    obs, _ = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    T_hist, SoC_hist, action_hist = [], [], []

    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        T_hist.append(info["T"])
        SoC_hist.append(info["SoC"])
        action_hist.append(float(action[0]))
        done = terminated or truncated

    T_arr = np.array(T_hist)
    tc = env.tc
    in_safe = np.sum((T_arr >= tc.T_safe_min) & (T_arr <= tc.T_safe_max))

    return {
        "return":      ep_return,
        "length":      env._step_count,
        "T_max":       float(T_arr.max()),
        "T_min":       float(T_arr.min()),
        "T_mean":      float(T_arr.mean()),
        "pct_safe":    100.0 * float(in_safe) / len(T_arr),
        "SoC_final":   float(SoC_hist[-1]),
        "T_hist":      T_hist,
        "SoC_hist":    SoC_hist,
        "action_hist": action_hist,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_episode(ep_data: dict, out_path: str, env: BatteryThermalEnv) -> None:
    """Trace température, SoC et action sur un épisode et sauvegarde en PNG."""
    tc = env.tc
    steps = range(len(ep_data["T_hist"]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Température
    ax = axes[0]
    ax.plot(steps, ep_data["T_hist"], color="tab:red", linewidth=1.2, label="T_cell")
    ax.axhline(tc.T_safe_min, color="blue",   linestyle="--", alpha=0.7,
               label=f"T_safe_min ({tc.T_safe_min}°C)")
    ax.axhline(tc.T_safe_max, color="orange", linestyle="--", alpha=0.7,
               label=f"T_safe_max ({tc.T_safe_max}°C)")
    ax.axhline(tc.T_cutoff,   color="red",    linestyle=":",  alpha=0.7,
               label=f"T_cutoff ({tc.T_cutoff}°C)")
    ax.set_ylabel("Température (°C)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # SoC
    ax = axes[1]
    ax.plot(steps, ep_data["SoC_hist"], color="tab:green", linewidth=1.2)
    ax.set_ylabel("SoC")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    # Action (puissance de refroidissement)
    ax = axes[2]
    ax.plot(steps, ep_data["action_hist"], color="tab:blue", linewidth=1.0)
    ax.set_ylabel("Refroidissement (0–1)")
    ax.set_xlabel("Step")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Évaluation — return={ep_data['return']:.1f}  "
        f"pct_safe={ep_data['pct_safe']:.1f}%  "
        f"T_max={ep_data['T_max']:.1f}°C",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def evaluate(args):
    env_cfg = EnvConfig(thermal=ThermalConfig(), reward=RewardConfig())
    env = BatteryThermalEnv(config=env_cfg)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = load_agent(
        args.checkpoint, obs_dim, action_dim,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
    )

    print(f"\n=== Évaluation — {args.n_episodes} épisodes (déterministe) ===")
    print(f"Checkpoint : {args.checkpoint}\n")
    print(f"{'ep':>4}  {'return':>10}  {'T_max':>7}  {'T_mean':>7}  {'pct_safe':>9}  {'SoC_final':>9}")
    print("-" * 58)

    results = []
    for i in range(args.n_episodes):
        ep = run_episode(agent, env, seed=i)
        results.append(ep)
        print(
            f"{i+1:4d}  {ep['return']:10.2f}  {ep['T_max']:7.1f}  "
            f"{ep['T_mean']:7.1f}  {ep['pct_safe']:9.1f}%  {ep['SoC_final']:9.3f}"
        )

    # --- Résumé agrégé ---
    returns   = [r["return"]   for r in results]
    pct_safes = [r["pct_safe"] for r in results]
    t_maxes   = [r["T_max"]    for r in results]

    summary = {
        "return_mean":   float(np.mean(returns)),
        "return_std":    float(np.std(returns)),
        "pct_safe_mean": float(np.mean(pct_safes)),
        "T_max_mean":    float(np.mean(t_maxes)),
        "n_episodes":    args.n_episodes,
        "checkpoint":    args.checkpoint,
    }

    print("-" * 58)
    print(f"return   : {summary['return_mean']:.2f} ± {summary['return_std']:.2f}")
    print(f"pct_safe : {summary['pct_safe_mean']:.1f}%")
    print(f"T_max    : {summary['T_max_mean']:.1f}°C")

    # --- Sauvegarde JSON ---
    out_dir   = os.path.dirname(os.path.abspath(args.checkpoint))
    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésultats → {json_path}")

    # --- Plot meilleur épisode ---
    if not args.no_plot:
        best_idx   = int(np.argmax(returns))
        plot_path  = os.path.join(out_dir, "eval_best_episode.png")
        plot_episode(results[best_idx], plot_path, env)
        print(f"Plot      → {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Évaluation checkpoint SAC")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Chemin vers le fichier .pt")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers",   type=int, default=2)
    p.add_argument("--no_plot",    action="store_true",
                   help="Désactive la génération du graphique")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
