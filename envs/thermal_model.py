"""
thermal_model.py — Modèle thermique RC 1 nœud (découplé de Gymnasium).

Modèle physique :
    dT/dt = (P_gen - P_cool) / C_th - (T - T_amb) / R_th

    P_gen  = I² × R_int          [W]  chaleur générée par le courant
    P_cool = action × P_cool_max  [W]  puissance de refroidissement active

Intégration Euler explicite avec pas dt.

Ce module est volontairement sans dépendance gym/numpy pour rester testable isolément.
"""


class ThermalModel:
    """Modèle RC thermique 1 nœud pour une cellule / pack batterie."""

    def __init__(self, config):
        """
        Args:
            config: ThermalConfig (voir config.py)
        """
        self.cfg = config

    def step(
        self,
        T: float,
        I: float,
        action: float,
        T_amb: float,
    ) -> tuple[float, float]:
        """
        Calcule la prochaine température et les puissances impliquées.

        Args:
            T      : température courante [°C]
            I      : courant de charge [A]  (positif = charge, négatif = décharge)
            action : fraction de refroidissement ∈ [0, 1]
            T_amb  : température ambiante [°C]

        Returns:
            T_next   : température après dt [°C]
            P_gen    : puissance générée [W]
            P_cool   : puissance de refroidissement appliquée [W]
        """
        action = float(max(0.0, min(1.0, action)))  # clip sécurité

        P_gen = (I ** 2) * self.cfg.R_int
        P_cool = action * self.cfg.P_cool_max

        dT = ((P_gen - P_cool) - (T - T_amb) / self.cfg.R_th) / self.cfg.C_th
        T_next = T + dT * self.cfg.dt

        return T_next, P_gen, P_cool

    def steady_state_temp(self, I: float, action: float, T_amb: float) -> float:
        """
        Température d'équilibre théorique (dT/dt = 0).
        Utile pour vérifier la cohérence des paramètres.

        T_ss = T_amb + R_th × (I²·R_int - action·P_cool_max)
        """
        P_net = (I ** 2) * self.cfg.R_int - action * self.cfg.P_cool_max
        return T_amb + self.cfg.R_th * P_net
