"""
train.py — Boucle d'entraînement SAC sur BatteryThermalEnv.

Usage :
    python train.py                        # paramètres par défaut
    python train.py --total_steps 200000   # entraînement plus long
    python train.py --run_name exp1        # nom du dossier de logs
    python train.py --eval_every 20        # évaluation tous les 20 épisodes

Les logs sont sauvegardés dans runs/<run_name>/.
Les checkpoints dans runs/<run_name>/checkpoints/.
"""

import argparse
import os
import numpy as np

from config import EnvConfig, ThermalConfig, RewardConfig
from envs.battery_thermal_env import BatteryThermalEnv
from agent.sac import SACAgent, SACConfig
from utils.logger import TrainingLogger


# ---------------------------------------------------------------------------
# Reward shaping — fonctions supplémentaires appliquées au-dessus de l'env
# ---------------------------------------------------------------------------

def shaped_reward(reward: float, info: dict, cfg: ThermalConfig) -> float:
    """
    Reward shaping optionnel appliqué après le reward de l'env.

    Ajoute un bonus de proximité à la température optimale :
        T_opt = (T_safe_min + T_safe_max) / 2
    Plus l'agent maintient T proche de T_opt, plus le bonus est élevé.
    Cela guide l'apprentissage initial (dense signal).

    Le coefficient est intentionnellement petit pour ne pas dominer le reward
    original (qui encode les vraies contraintes).
    """
    T = info["T"]
    T_opt = (cfg.T_safe_min + cfg.T_safe_max) / 2.0
    proximity_bonus = 0.01 * max(0.0, 1.0 - abs(T - T_opt) / (cfg.T_safe_max - T_opt))
    return reward + proximity_bonus


# ---------------------------------------------------------------------------
# Évaluation déterministe
# ---------------------------------------------------------------------------

def evaluate(agent: SACAgent, env: BatteryThermalEnv, n_episodes: int = 5) -> dict:
    """
    Évalue l'agent en mode déterministe (sans exploration).

    Returns :
        dict avec mean/std de return, longueur d'épisode, T_max, pct_in_safe
    """
    returns, lengths, t_maxes, pct_safes = [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        T_arr = np.array(env.history["T"])
        tc = env.tc
        in_safe = np.sum((T_arr >= tc.T_safe_min) & (T_arr <= tc.T_safe_max))

        returns.append(ep_return)
        lengths.append(env._step_count)
        t_maxes.append(T_arr.max())
        pct_safes.append(100.0 * in_safe / len(T_arr))

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std":  float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_T_max_mean":  float(np.mean(t_maxes)),
        "eval_pct_safe":    float(np.mean(pct_safes)),
    }


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def train(args):
    # --- Config ---
    env_cfg = EnvConfig(
        thermal=ThermalConfig(),
        reward=RewardConfig(),
    )
    sac_cfg = SACConfig(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        lr_actor=args.lr,
        lr_critic=args.lr,
        lr_alpha=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        learning_starts=args.learning_starts,
    )

    # --- Environnements ---
    env      = BatteryThermalEnv(config=env_cfg)
    eval_env = BatteryThermalEnv(config=env_cfg)

    # --- Agent ---
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, action_dim, config=sac_cfg)

    # --- Logger ---
    run_dir  = os.path.join("runs", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = TrainingLogger(log_dir=run_dir, print_every=args.print_every)

    print(f"\n=== SAC — BatteryThermalEnv ===")
    print(f"obs_dim={obs_dim}  action_dim={action_dim}")
    print(f"total_steps={args.total_steps}  learning_starts={args.learning_starts}")
    print(f"logs → {run_dir}\n")

    # --- Boucle ---
    total_steps  = 0
    ep           = 0
    obs, _       = env.reset(seed=args.seed)

    ep_return    = 0.0
    ep_T_list    = []
    ep_steps     = 0

    while total_steps < args.total_steps:
        # Exploration aléatoire avant learning_starts
        if total_steps < args.learning_starts:
            action = env.action_space.sample().flatten()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Reward shaping
        reward_shaped = shaped_reward(reward, info, env_cfg.thermal)

        # Push dans le buffer (avec reward shapé)
        agent.push(obs, action, reward_shaped, next_obs, float(terminated))

        ep_return += reward          # log du reward original (non shapé)
        ep_T_list.append(info["T"])
        ep_steps  += 1
        total_steps += 1

        obs = next_obs

        # Update SAC
        if agent.ready:
            metrics = agent.update()
            logger.log_update(step=total_steps, **metrics)

        # Fin d'épisode
        if done:
            ep += 1
            T_arr    = np.array(ep_T_list)
            tc       = env_cfg.thermal
            in_safe  = np.sum((T_arr >= tc.T_safe_min) & (T_arr <= tc.T_safe_max))

            ep_metrics = {
                "ep_return":    ep_return,
                "ep_length":    ep_steps,
                "T_max":        float(T_arr.max()),
                "T_min":        float(T_arr.min()),
                "T_mean":       float(T_arr.mean()),
                "SoC_final":    info["SoC"],
                "pct_in_safe":  100.0 * in_safe / len(T_arr),
            }
            logger.log_episode(ep=ep, total_steps=total_steps, **ep_metrics)

            # Évaluation périodique
            if ep % args.eval_every == 0:
                eval_metrics = evaluate(agent, eval_env)
                print(
                    f"  [EVAL] return={eval_metrics['eval_return_mean']:8.2f} ± "
                    f"{eval_metrics['eval_return_std']:.2f}  |  "
                    f"pct_safe={eval_metrics['eval_pct_safe']:.1f}%  |  "
                    f"T_max={eval_metrics['eval_T_max_mean']:.1f}°C"
                )
                logger.log_episode(ep=ep, total_steps=total_steps, **ep_metrics, **eval_metrics)

            # Checkpoint
            if ep % args.save_every == 0:
                path = os.path.join(ckpt_dir, f"sac_ep{ep}.pt")
                agent.save(path)

            # Reset
            obs, _    = env.reset()
            ep_return = 0.0
            ep_T_list = []
            ep_steps  = 0

    # Sauvegarde finale
    agent.save(os.path.join(ckpt_dir, "sac_final.pt"))
    logger.close()
    print(f"\nEntraînement terminé. Modèle sauvegardé dans {ckpt_dir}/sac_final.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SAC — Contrôle thermique batterie")

    # Durée
    p.add_argument("--total_steps",     type=int,   default=100_000)
    p.add_argument("--learning_starts", type=int,   default=1_000)

    # Architecture
    p.add_argument("--hidden_dim", type=int,   default=256)
    p.add_argument("--n_layers",   type=int,   default=2)

    # Hyperparamètres SAC
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--tau",          type=float, default=0.005)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--buffer_capacity", type=int, default=100_000)

    # Logging
    p.add_argument("--run_name",    type=str, default="default")
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--eval_every",  type=int, default=50)
    p.add_argument("--save_every",  type=int, default=100)

    # Reproductibilité
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
