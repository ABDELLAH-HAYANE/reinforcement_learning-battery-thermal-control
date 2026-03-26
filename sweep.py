"""
sweep.py — Grid search sur les hyperparamètres SAC.

Lance plusieurs runs courts et compare les performances finales.
Résultats sauvegardés dans runs/sweep_TIMESTAMP/sweep_results.csv.

Usage :
    python sweep.py                     # grille complète (12 runs × 20k steps)
    python sweep.py --total_steps 5000  # grille rapide pour tester
"""

import argparse
import csv
import itertools
import os
import time
import numpy as np

from config import EnvConfig, ThermalConfig, RewardConfig
from envs.battery_thermal_env import BatteryThermalEnv
from agent.sac import SACAgent, SACConfig
from train import evaluate as eval_fn, shaped_reward


# ---------------------------------------------------------------------------
# Grille de recherche — adapte les valeurs à ton contexte
# ---------------------------------------------------------------------------

GRID = {
    "lr":    [1e-4, 3e-4, 1e-3],
    "gamma": [0.95, 0.99],
    "tau":   [0.005, 0.01],
}


# ---------------------------------------------------------------------------
# Un run complet
# ---------------------------------------------------------------------------

def run_single(
    lr: float,
    gamma: float,
    tau: float,
    total_steps: int = 20_000,
    seed: int = 42,
) -> dict:
    """
    Entraîne un agent SAC avec les hyperparamètres donnés et retourne
    les métriques d'évaluation finale (5 épisodes déterministes).
    """
    env_cfg = EnvConfig(thermal=ThermalConfig(), reward=RewardConfig())
    sac_cfg = SACConfig(
        lr_actor=lr, lr_critic=lr, lr_alpha=lr,
        gamma=gamma, tau=tau,
        hidden_dim=256, n_layers=2,
        batch_size=256, buffer_capacity=50_000,
        learning_starts=1_000,
    )

    env      = BatteryThermalEnv(config=env_cfg)
    eval_env = BatteryThermalEnv(config=env_cfg)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, action_dim, config=sac_cfg)

    obs, _ = env.reset(seed=seed)
    total = 0

    while total < total_steps:
        if total < sac_cfg.learning_starts:
            action = env.action_space.sample().flatten()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        reward_s = shaped_reward(reward, info, env_cfg.thermal)
        agent.push(obs, action, reward_s, next_obs, float(terminated))

        if agent.ready:
            agent.update()

        obs = next_obs
        total += 1

        if terminated or truncated:
            obs, _ = env.reset()

    return eval_fn(agent, eval_env, n_episodes=5)


# ---------------------------------------------------------------------------
# Script sweep
# ---------------------------------------------------------------------------

def sweep(args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join("runs", f"sweep_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))

    print(f"\n=== Sweep hyperparamètres ===")
    print(f"Configurations : {len(combos)}")
    print(f"Steps par run  : {args.total_steps}")
    print(f"Résultats      → {out_dir}/sweep_results.csv\n")

    metric_keys = ["eval_return_mean", "eval_return_std", "eval_pct_safe", "eval_T_max_mean"]
    fieldnames  = keys + metric_keys
    csv_path    = os.path.join(out_dir, "sweep_results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            print(f"[{i+1:2d}/{len(combos)}] lr={params['lr']:.0e}  "
                  f"gamma={params['gamma']}  tau={params['tau']} ...",
                  end=" ", flush=True)

            t0 = time.time()
            metrics = run_single(**params, total_steps=args.total_steps)
            elapsed = time.time() - t0

            row = {**params, **{k: metrics[k] for k in metric_keys}}
            writer.writerow(row)
            f.flush()

            print(
                f"return={metrics['eval_return_mean']:8.1f}  "
                f"pct_safe={metrics['eval_pct_safe']:.1f}%  "
                f"({elapsed:.0f}s)"
            )

    print(f"\nSweep terminé → {csv_path}")
    _print_best(csv_path, keys)


def _print_best(csv_path: str, param_keys: list) -> None:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    best = max(rows, key=lambda r: float(r["eval_return_mean"]))
    print("\n--- Meilleure configuration ---")
    for k in param_keys:
        print(f"  {k:6s} = {best[k]}")
    print(f"  return   = {float(best['eval_return_mean']):.2f} ± {float(best['eval_return_std']):.2f}")
    print(f"  pct_safe = {float(best['eval_pct_safe']):.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Grid search SAC")
    p.add_argument("--total_steps", type=int, default=20_000,
                   help="Steps d'entraînement par configuration")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sweep(args)
