"""
logger.py — Logger d'entraînement : console + CSV.

Enregistre les métriques par épisode et par update SAC dans deux fichiers CSV
séparés. Affiche aussi un résumé console tous les N épisodes.
"""

import csv
import os
import time
from collections import deque


class TrainingLogger:
    """
    Logger léger pour l'entraînement RL.

    Crée deux fichiers dans `log_dir` :
        episodes.csv   — métriques par épisode (return, longueur, T_max, ...)
        updates.csv    — métriques par update SAC (losses, alpha, entropy)

    Usage :
        logger = TrainingLogger(log_dir="runs/exp1")
        logger.log_episode(ep=1, total_steps=500, **ep_metrics)
        logger.log_update(step=100, **sac_metrics)
        logger.print_summary(ep=10)
    """

    def __init__(self, log_dir: str = "runs/default", print_every: int = 10):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir     = log_dir
        self.print_every = print_every
        self.start_time  = time.time()

        # Fenêtres glissantes pour le résumé console
        self._ep_returns  = deque(maxlen=print_every)
        self._ep_lengths  = deque(maxlen=print_every)

        # Fichiers CSV
        self._ep_file  = open(os.path.join(log_dir, "episodes.csv"),  "w", newline="")
        self._upd_file = open(os.path.join(log_dir, "updates.csv"),   "w", newline="")
        self._ep_writer  = None
        self._upd_writer = None

    # ------------------------------------------------------------------

    def log_episode(self, ep: int, total_steps: int, **kwargs) -> None:
        """
        Enregistre les métriques d'un épisode complet.

        Kwargs typiques :
            ep_return   : récompense cumulée
            ep_length   : nombre de steps
            T_max       : température max atteinte [°C]
            T_min       : température min atteinte [°C]
            T_mean      : température moyenne [°C]
            SoC_final   : SoC en fin d'épisode
            pct_in_safe : % de steps dans la plage sûre
        """
        row = {"ep": ep, "total_steps": total_steps, **kwargs}

        if self._ep_writer is None:
            self._ep_writer = csv.DictWriter(self._ep_file, fieldnames=list(row.keys()))
            self._ep_writer.writeheader()

        self._ep_writer.writerow(row)
        self._ep_file.flush()

        # Mise à jour fenêtres glissantes
        if "ep_return" in kwargs:
            self._ep_returns.append(kwargs["ep_return"])
        if "ep_length" in kwargs:
            self._ep_lengths.append(kwargs["ep_length"])

        if ep % self.print_every == 0:
            self.print_summary(ep, total_steps)

    def log_update(self, step: int, **kwargs) -> None:
        """Enregistre les métriques d'une update SAC."""
        row = {"step": step, **kwargs}

        if self._upd_writer is None:
            self._upd_writer = csv.DictWriter(self._upd_file, fieldnames=list(row.keys()))
            self._upd_writer.writeheader()

        self._upd_writer.writerow(row)
        self._upd_file.flush()

    def print_summary(self, ep: int, total_steps: int = 0) -> None:
        elapsed = time.time() - self.start_time
        mean_ret = sum(self._ep_returns) / len(self._ep_returns) if self._ep_returns else float("nan")
        mean_len = sum(self._ep_lengths) / len(self._ep_lengths) if self._ep_lengths else float("nan")

        print(
            f"[ep {ep:4d} | steps {total_steps:7d} | {elapsed:6.0f}s] "
            f"return={mean_ret:8.2f}  len={mean_len:6.0f}"
        )

    def close(self) -> None:
        self._ep_file.close()
        self._upd_file.close()
