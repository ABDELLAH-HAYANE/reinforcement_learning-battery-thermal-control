"""
test_day5.py — Tests pour baselines/rule_based.py et compare.py.
"""

import numpy as np
import pytest

from config import ThermalConfig, EnvConfig, RewardConfig
from envs.battery_thermal_env import BatteryThermalEnv
from baselines.rule_based import (
    RandomController,
    BangBangController,
    ProportionalController,
    HysteresisController,
    _denorm_T,
)
from compare import evaluate_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tc():
    return ThermalConfig()


@pytest.fixture
def env():
    cfg = EnvConfig(thermal=ThermalConfig(max_steps=200), reward=RewardConfig())
    return BatteryThermalEnv(config=cfg)


def _obs_for_T(T: float, tc: ThermalConfig) -> np.ndarray:
    """Construit une obs minimale avec T_norm calculé depuis T réel."""
    T_lo = tc.T_amb_min - 5.0
    T_hi = tc.T_cutoff
    T_norm = 2.0 * (T - T_lo) / (T_hi - T_lo) - 1.0
    return np.array([T_norm, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# _denorm_T
# ---------------------------------------------------------------------------

class TestDenormT:

    def test_round_trip(self, tc):
        for T_real in [15.0, 25.0, 45.0, 60.0]:
            obs = _obs_for_T(T_real, tc)
            T_back = _denorm_T(float(obs[0]), tc)
            assert abs(T_back - T_real) < 1e-4, f"Round-trip failed for T={T_real}"

    def test_cutoff_maps_to_one(self, tc):
        obs = _obs_for_T(tc.T_cutoff, tc)
        assert abs(obs[0] - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# RandomController
# ---------------------------------------------------------------------------

class TestRandomController:

    def test_action_in_bounds(self):
        ctrl = RandomController(seed=0)
        obs = np.zeros(5, dtype=np.float32)
        for _ in range(50):
            action = ctrl.select_action(obs)
            assert 0.0 <= float(action[0]) <= 1.0

    def test_output_shape(self):
        ctrl = RandomController()
        action = ctrl.select_action(np.zeros(5, dtype=np.float32))
        assert action.shape == (1,)


# ---------------------------------------------------------------------------
# BangBangController
# ---------------------------------------------------------------------------

class TestBangBangController:

    def test_full_cooling_when_overtemp(self, tc):
        ctrl = BangBangController(tc)
        obs  = _obs_for_T(tc.T_safe_max + 5.0, tc)
        action = ctrl.select_action(obs)
        assert float(action[0]) == 1.0

    def test_no_cooling_when_safe(self, tc):
        ctrl = BangBangController(tc)
        obs  = _obs_for_T((tc.T_safe_min + tc.T_safe_max) / 2.0, tc)
        action = ctrl.select_action(obs)
        assert float(action[0]) == 0.0

    def test_action_is_binary(self, tc):
        ctrl = BangBangController(tc)
        for T in [20.0, 35.0, 46.0, 55.0]:
            obs = _obs_for_T(T, tc)
            a = float(ctrl.select_action(obs)[0])
            assert a in (0.0, 1.0), f"Action non binaire pour T={T}: {a}"


# ---------------------------------------------------------------------------
# ProportionalController
# ---------------------------------------------------------------------------

class TestProportionalController:

    def test_action_in_bounds(self, tc):
        ctrl = ProportionalController(tc)
        for T in [10.0, 20.0, 30.0, 45.0, 60.0]:
            obs = _obs_for_T(T, tc)
            a = float(ctrl.select_action(obs)[0])
            assert 0.0 <= a <= 1.0, f"Action hors bornes pour T={T}: {a}"

    def test_higher_T_higher_action(self, tc):
        ctrl = ProportionalController(tc)
        obs_low  = _obs_for_T(25.0, tc)
        obs_high = _obs_for_T(50.0, tc)
        a_low  = float(ctrl.select_action(obs_low)[0])
        a_high = float(ctrl.select_action(obs_high)[0])
        assert a_high >= a_low, "Action doit croître avec la température"

    def test_no_cooling_below_opt(self, tc):
        ctrl = ProportionalController(tc)
        T_below_opt = tc.T_safe_min  # bien en dessous de T_opt
        obs = _obs_for_T(T_below_opt, tc)
        a = float(ctrl.select_action(obs)[0])
        assert a == 0.0, f"Pas de refroidissement attendu sous T_opt, got {a}"


# ---------------------------------------------------------------------------
# HysteresisController
# ---------------------------------------------------------------------------

class TestHysteresisController:

    def test_activates_above_threshold(self, tc):
        ctrl = HysteresisController(tc, hysteresis=3.0)
        obs  = _obs_for_T(tc.T_safe_max + 1.0, tc)
        action = ctrl.select_action(obs)
        assert float(action[0]) == 1.0

    def test_stays_off_in_safe_range(self, tc):
        ctrl = HysteresisController(tc, hysteresis=3.0)
        obs  = _obs_for_T(tc.T_safe_max - 5.0, tc)
        action = ctrl.select_action(obs)
        assert float(action[0]) == 0.0

    def test_hysteresis_maintains_state(self, tc):
        ctrl = HysteresisController(tc, hysteresis=5.0)
        # Active le refroidissement
        obs_hot  = _obs_for_T(tc.T_safe_max + 1.0, tc)
        ctrl.select_action(obs_hot)
        assert ctrl._cooling_on is True

        # Dans la zone d'hystérésis : maintient l'état
        obs_mid  = _obs_for_T(tc.T_safe_max - 2.0, tc)
        action   = ctrl.select_action(obs_mid)
        assert float(action[0]) == 1.0  # toujours actif

    def test_reset_clears_state(self, tc):
        ctrl = HysteresisController(tc)
        ctrl._cooling_on = True
        ctrl.reset()
        assert ctrl._cooling_on is False


# ---------------------------------------------------------------------------
# evaluate_agent (intégration)
# ---------------------------------------------------------------------------

class TestEvaluateAgent:

    def test_returns_expected_keys(self, env):
        ctrl   = BangBangController(env.tc)
        result = evaluate_agent(ctrl, env, n_episodes=3)
        for k in ["return_mean", "return_std", "pct_safe_mean", "T_max_mean", "results"]:
            assert k in result

    def test_metrics_finite(self, env):
        ctrl   = ProportionalController(env.tc)
        result = evaluate_agent(ctrl, env, n_episodes=3)
        for k in ["return_mean", "pct_safe_mean", "T_max_mean"]:
            assert np.isfinite(result[k]), f"{k} non fini"

    def test_proportional_beats_random(self, env):
        """Le contrôleur proportionnel doit avoir un meilleur pct_safe que le hasard."""
        prop_m   = evaluate_agent(ProportionalController(env.tc), env, n_episodes=10, seed_offset=0)
        random_m = evaluate_agent(RandomController(seed=0),       env, n_episodes=10, seed_offset=0)
        assert prop_m["pct_safe_mean"] >= random_m["pct_safe_mean"], (
            f"Proportionnel ({prop_m['pct_safe_mean']:.1f}%) "
            f"devrait battre Random ({random_m['pct_safe_mean']:.1f}%)"
        )
