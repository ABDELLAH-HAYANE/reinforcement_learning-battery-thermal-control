"""
config.py — Paramètres génériques de simulation thermique batterie.

Toutes les valeurs sont commentées avec leur signification physique.
Adapte ces valeurs à ton type de batterie / contexte FSE.
"""

from dataclasses import dataclass


@dataclass
class ThermalConfig:
    # --- Paramètres thermiques (modèle RC 1 nœud) ---
    C_th: float = 100.0       # Capacité thermique [J/K]   — augmenter pour un pack plus lourd
    R_th: float = 1.0         # Résistance thermique vers l'ambiant [K/W] — diminuer si bon refroidissement passif
    R_int: float = 0.05       # Résistance interne [Ω]     — chaleur générée = I² × R_int
    P_cool_max: float = 50.0  # Puissance max de refroidissement [W]

    # --- Paramètres batterie ---
    C_batt: float = 10.0      # Capacité de la batterie [Ah]
    I_max: float = 50.0       # Courant max de charge [A]
    I_min: float = -10.0      # Courant min (décharge) [A]  — négatif = décharge

    # --- Plages de température ---
    T_init_min: float = 20.0  # Température initiale min [°C]
    T_init_max: float = 30.0  # Température initiale max [°C]
    T_amb_min: float = 15.0   # Température ambiante min [°C]
    T_amb_max: float = 35.0   # Température ambiante max [°C]
    T_safe_min: float = 15.0   # Seuil bas de température sûre [°C]
    T_safe_max: float = 45.0   # Seuil haut de température sûre [°C]
    T_warning: float = 40.0    # Seuil d'alerte dT/dt — pénalise les montées rapides au-delà [°C]
    T_cutoff: float = 60.0     # Coupure de sécurité (fin d'épisode) [°C]

    # --- SoC initial ---
    SoC_init_min: float = 0.2  # SoC initial min [fraction 0-1]
    SoC_init_max: float = 0.8  # SoC initial max [fraction 0-1]

    # --- Simulation ---
    dt: float = 1.0            # Pas de temps [s]
    max_steps: int = 3600      # Durée max d'un épisode [steps] — 1h à dt=1s


@dataclass
class RewardConfig:
    # Poids de la fonction de récompense — à tuner selon les priorités
    w_temp_high: float = 5.0   # Pénalité dépassement haut (quadratique) — ↑ pour punir plus fort les pics
    w_temp_low: float = 1.0    # Pénalité dépassement bas (quadratique)
    w_cooling: float = 0.01    # Coût énergétique du refroidissement — ↓ pour autoriser plus de cooling
    w_soc: float = 0.1         # Bonus progression de charge
    w_delta_T: float = 0.5     # Pénalité vitesse de montée thermique (anticipe les pics)


@dataclass
class EnvConfig:
    thermal: ThermalConfig = None
    reward: RewardConfig = None

    def __post_init__(self):
        if self.thermal is None:
            self.thermal = ThermalConfig()
        if self.reward is None:
            self.reward = RewardConfig()


# Instance par défaut utilisable directement
DEFAULT_CONFIG = EnvConfig()
