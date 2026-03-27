"""
compare.py — Comparaison SAC vs baselines sur N épisodes.

Usage :
    python compare.py
    python compare.py --n_episodes 20 --seed 0 --out_dir results/

Génère dans results/ :
    comparison_trajectories.png  — trajectoires T, SoC, action (1 épisode)
    comparison_boxplots.png      — distributions return / pct_safe (N épisodes)
    benchmark.json               — métriques agrégées
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EnvConfig, ThermalConfig, RewardConfig
from envs.battery_thermal_env import BatteryThermalEnv
from agent.sac import SACAgent, SACConfig
from baselines.rule_based import (
    PIDController, BangBangController,
    ProportionalController, HysteresisController,
)

# ---------------------------------------------------------------------------
# Palette couleurs cohérente
# ---------------------------------------------------------------------------

COLORS = {
    "SAC":          "#2196F3",
    "PID":          "#4CAF50",
    "BangBang":     "#F44336",
    "Proportional": "#FF9800",
    "Hysteresis":   "#9C27B0",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoint() -> str:
    ckpts = sorted(glob.glob("runs/*/checkpoints/sac_final.pt"))
    if not ckpts:
        raise FileNotFoundError("Aucun checkpoint sac_final.pt trouvé dans runs/")
    return ckpts[-1]


def run_episode(controller, env, tc, seed: int = 42) -> dict:
    controller.reset()
    obs, _ = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    T_hist, SoC_hist, action_hist = [], [], []

    while not done:
        action = controller.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        T_hist.append(info["T"])
        SoC_hist.append(info["SoC"])
        action_hist.append(float(action[0]))
        done = terminated or truncated

    T_arr   = np.array(T_hist)
    in_safe = np.sum((T_arr >= tc.T_safe_min) & (T_arr <= tc.T_safe_max))
    return {
        "return":      ep_return,
        "length":      len(T_hist),
        "pct_safe":    100.0 * float(in_safe) / len(T_arr),
        "T_max":       float(T_arr.max()),
        "T_min":       float(T_arr.min()),
        "T_mean":      float(T_arr.mean()),
        "T_hist":      T_hist,
        "SoC_hist":    SoC_hist,
        "action_hist": action_hist,
    }


def run_n_episodes(controller, env, tc, n: int, base_seed: int = 0) -> list:
    return [run_episode(controller, env, tc, seed=base_seed + i) for i in range(n)]


# ---------------------------------------------------------------------------
# Figure 1 — Trajectoires (1 épisode, seed=42)
# ---------------------------------------------------------------------------

def plot_trajectories(results_1ep: dict, tc: ThermalConfig, out_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    max_len = max(len(ep["T_hist"]) for ep in results_1ep.values())

    for name, ep in results_1ep.items():
        steps = range(len(ep["T_hist"]))
        c = COLORS.get(name, "gray")
        axes[0].plot(steps, ep["T_hist"],      color=c, lw=1.4, label=name, alpha=0.85)
        axes[1].plot(steps, ep["SoC_hist"],    color=c, lw=1.4, alpha=0.85)
        axes[2].plot(steps, ep["action_hist"], color=c, lw=1.0, alpha=0.85)

    # Repères thermiques
    ax = axes[0]
    ax.axhline(tc.T_safe_min, color="steelblue",  ls="--", alpha=0.6, lw=1.2,
               label=f"T_safe [{tc.T_safe_min}–{tc.T_safe_max}°C]")
    ax.axhline(tc.T_safe_max, color="darkorange",  ls="--", alpha=0.6, lw=1.2)
    ax.axhline(tc.T_cutoff,   color="crimson",     ls=":",  alpha=0.6, lw=1.2,
               label=f"T_cutoff ({tc.T_cutoff}°C)")
    ax.fill_between(range(max_len), tc.T_safe_min, tc.T_safe_max,
                    alpha=0.07, color="green", label="Zone sûre")
    ax.set_ylabel("Température (°C)", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)

    axes[1].set_ylabel("SoC", fontsize=11)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Action refroidissement", fontsize=11)
    axes[2].set_xlabel("Step", fontsize=11)
    axes[2].set_ylim(-0.02, 1.05)
    axes[2].grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], color=COLORS.get(n, "gray"), lw=2, label=n)
               for n in results_1ep]
    axes[2].legend(handles=handles, fontsize=9, loc="upper right")

    fig.suptitle("Comparaison des contrôleurs — Trajectoires (seed=42)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Trajectoires -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Boxplots sur N épisodes
# ---------------------------------------------------------------------------

def plot_boxplots(results_n: dict, out_path: str) -> None:
    names    = list(results_n.keys())
    returns  = [[ep["return"]   for ep in results_n[n]] for n in names]
    pct_safe = [[ep["pct_safe"] for ep in results_n[n]] for n in names]
    colors   = [COLORS.get(n, "gray") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, title, ylabel in [
        (axes[0], returns,  "Return cumulé",            "Return"),
        (axes[1], pct_safe, "% steps en zone sûre",     "pct_safe (%)"),
    ]:
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", lw=2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, fontsize=10, rotation=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    n = len(next(iter(results_n.values())))
    fig.suptitle(f"Comparaison des contrôleurs — {n} épisodes",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Boxplots      -> {out_path}")


# ---------------------------------------------------------------------------
# Benchmark JSON
# ---------------------------------------------------------------------------

def save_benchmark(results_n: dict, out_path: str) -> None:
    benchmark = {}
    for name, episodes in results_n.items():
        returns   = [e["return"]   for e in episodes]
        pct_safes = [e["pct_safe"] for e in episodes]
        t_maxes   = [e["T_max"]    for e in episodes]
        benchmark[name] = {
            "return_mean":   round(float(np.mean(returns)),   2),
            "return_std":    round(float(np.std(returns)),    2),
            "pct_safe_mean": round(float(np.mean(pct_safes)), 2),
            "pct_safe_std":  round(float(np.std(pct_safes)),  2),
            "T_max_mean":    round(float(np.mean(t_maxes)),   2),
            "n_episodes":    len(episodes),
        }
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"  Benchmark     -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    env_cfg = EnvConfig(thermal=ThermalConfig(), reward=RewardConfig())
    tc      = env_cfg.thermal
    env     = BatteryThermalEnv(config=env_cfg)

    ckpt = _find_checkpoint()
    print(f"Checkpoint SAC : {ckpt}")
    sac = SACAgent(obs_dim=5, action_dim=1, config=SACConfig(hidden_dim=256, n_layers=2))
    sac.load(ckpt)
    sac.reset = lambda: None

    controllers = {
        "SAC":          sac,
        "PID":          PIDController(tc),
        "BangBang":     BangBangController(tc),
        "Proportional": ProportionalController(tc),
        "Hysteresis":   HysteresisController(tc),
    }

    # Trajectoires (1 épisode)
    print("\n[1/3] Trajectoires (1 épisode)...")
    results_1ep = {n: run_episode(c, env, tc, seed=42) for n, c in controllers.items()}
    plot_trajectories(results_1ep, tc,
                      os.path.join(args.out_dir, "comparison_trajectories.png"))

    # N épisodes
    print(f"\n[2/3] Benchmark ({args.n_episodes} épisodes)...")
    results_n = {}
    for name, ctrl in controllers.items():
        print(f"  {name:<14}", end=" ", flush=True)
        episodes = run_n_episodes(ctrl, env, tc, n=args.n_episodes, base_seed=args.seed)
        results_n[name] = episodes
        print(f"return={np.mean([e['return'] for e in episodes]):8.0f}  "
              f"pct_safe={np.mean([e['pct_safe'] for e in episodes]):.1f}%")

    print("\n[3/3] Figures et benchmark JSON...")
    plot_boxplots(results_n, os.path.join(args.out_dir, "comparison_boxplots.png"))
    save_benchmark(results_n, os.path.join(args.out_dir, "benchmark.json"))

    # Résumé console
    print("\n=== Résumé ===")
    print(f"{'Contrôleur':<14} {'return':>10}  {'±':>7}  {'pct_safe':>9}  {'T_max':>7}")
    print("-" * 55)
    for name, episodes in results_n.items():
        r = [e["return"]   for e in episodes]
        s = [e["pct_safe"] for e in episodes]
        t = [e["T_max"]    for e in episodes]
        print(f"{name:<14} {np.mean(r):10.1f}  {np.std(r):7.1f}  "
              f"{np.mean(s):8.1f}%  {np.mean(t):7.1f}°C")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--out_dir",    type=str, default="results")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
