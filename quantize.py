"""
quantize.py — Quantification INT8 de l'actor SAC + export ONNX.

Deux étapes :
    1. Quantification dynamique PyTorch (INT8) — taille et latence
    2. Export ONNX de l'actor — prêt pour déploiement C++/edge

Usage :
    python quantize.py
    python quantize.py --out_dir results/

Fichiers générés dans out_dir/ :
    actor_fp32.pt      — actor poids float32
    actor_int8.pt      — actor quantifié INT8
    actor.onnx         — actor export ONNX (float32)
    quantize_report.json
"""

import argparse
import glob
import json
import os
import time
import warnings
import numpy as np
import torch
import torch.quantization

warnings.filterwarnings("ignore", category=DeprecationWarning)

from agent.sac import SACAgent, SACConfig
from agent.networks import Actor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoint() -> str:
    ckpts = sorted(glob.glob("runs/*/checkpoints/sac_final.pt"))
    if not ckpts:
        raise FileNotFoundError("Aucun checkpoint sac_final.pt trouve dans runs/")
    return ckpts[-1]


def _file_size_kb(path: str) -> float:
    return os.path.getsize(path) / 1024.0


def _measure_latency(model_fn, obs_batch: torch.Tensor, n_runs: int = 500) -> float:
    """Retourne la latence moyenne en millisecondes sur n_runs passes."""
    # Warmup
    for _ in range(20):
        model_fn(obs_batch)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        model_fn(obs_batch)
    elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / n_runs  # ms par inférence


def _max_action_error(actor_fp32: Actor, actor_int8, obs_batch: torch.Tensor) -> float:
    """
    Erreur max sur mu (pre-tanh) entre FP32 et INT8.
    On compare mu directement : plus stable numeriquement que tanh(mu).
    """
    with torch.no_grad():
        mu_fp, _ = actor_fp32(obs_batch)
        mu_q,  _ = actor_int8(obs_batch)
    return float((mu_fp - mu_q).abs().max().item())


# ---------------------------------------------------------------------------
# Etape 1 — Quantification dynamique INT8
# ---------------------------------------------------------------------------

def quantize_dynamic(actor: Actor) -> torch.nn.Module:
    """
    Quantification dynamique PyTorch INT8.

    Les poids des couches Linear sont convertis en INT8.
    Les activations restent en float32 (pas besoin de données de calibration).
    Adapte pour l'inference CPU embarquee.
    """
    actor_int8 = torch.quantization.quantize_dynamic(
        actor,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8,
    )
    return actor_int8


# ---------------------------------------------------------------------------
# Etape 2 — Export ONNX
# ---------------------------------------------------------------------------

def export_onnx(actor: Actor, obs_dim: int, out_path: str) -> None:
    """
    Exporte l'actor en ONNX (opset 17).

    Le modele exporte correspond au mode deterministe :
        obs -> mu -> tanh(mu)  (action dans (-1,1))
    """

    class ActorDeterministic(torch.nn.Module):
        """Wrapper deterministe : retourne uniquement tanh(mu)."""
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, obs):
            mu, _ = self.actor(obs)
            return torch.tanh(mu)

    wrapper = ActorDeterministic(actor)
    wrapper.eval()

    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}},
        opset_version=17,
        verbose=False,
        dynamo=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Charger le checkpoint
    ckpt = _find_checkpoint()
    print(f"Checkpoint : {ckpt}")

    agent = SACAgent(obs_dim=5, action_dim=1,
                     config=SACConfig(hidden_dim=256, n_layers=2))
    agent.load(ckpt)
    actor_fp32 = agent.actor.eval()

    obs_dim = 5

    # Batch de test (1000 observations aleatoires dans [-1,1])
    torch.manual_seed(42)
    obs_batch = torch.rand(1000, obs_dim) * 2 - 1

    # --- Etape 1 : Quantification INT8 ---
    print("\n[1/3] Quantification dynamique INT8...")

    path_fp32 = os.path.join(args.out_dir, "actor_fp32.pt")
    torch.save(actor_fp32.state_dict(), path_fp32)

    actor_int8 = quantize_dynamic(actor_fp32)

    path_int8 = os.path.join(args.out_dir, "actor_int8.pt")
    torch.save(actor_int8.state_dict(), path_int8)

    size_fp32 = _file_size_kb(path_fp32)
    size_int8 = _file_size_kb(path_int8)
    compression = size_fp32 / size_int8 if size_int8 > 0 else 0.0

    print(f"  FP32 : {size_fp32:.1f} KB")
    print(f"  INT8 : {size_int8:.1f} KB  (x{compression:.2f} compression)")

    # Latence
    lat_fp32 = _measure_latency(
        lambda x: actor_fp32(x), obs_batch[:1], n_runs=500
    )
    lat_int8 = _measure_latency(
        lambda x: actor_int8(x), obs_batch[:1], n_runs=500
    )
    speedup = lat_fp32 / lat_int8 if lat_int8 > 0 else 0.0

    print(f"  Latence FP32 : {lat_fp32:.3f} ms")
    print(f"  Latence INT8 : {lat_int8:.3f} ms  (x{speedup:.2f} speedup)")

    # Precision
    max_err = _max_action_error(actor_fp32, actor_int8, obs_batch)
    level = "OK" if max_err < 0.1 else "elevee - QAT recommande pour production"
    print(f"  Erreur max mu     : {max_err:.4f}  [{level}]")

    # --- Etape 2 : Export ONNX ---
    print("\n[2/3] Export ONNX...")
    path_onnx = os.path.join(args.out_dir, "actor.onnx")
    export_onnx(actor_fp32, obs_dim, path_onnx)
    size_onnx = _file_size_kb(path_onnx)
    print(f"  actor.onnx : {size_onnx:.1f} KB")

    # Validation ONNX (si onnxruntime disponible)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path_onnx,
                                    providers=["CPUExecutionProvider"])
        obs_np = obs_batch[:1].numpy()
        out_onnx = sess.run(None, {"obs": obs_np})[0]
        with torch.no_grad():
            mu, _ = actor_fp32(obs_batch[:1])
            out_pt = torch.tanh(mu).numpy()
        onnx_err = float(np.abs(out_onnx - out_pt).max())
        print(f"  Validation ONNX vs PyTorch : erreur max = {onnx_err:.8f}")
    except ImportError:
        print("  (onnxruntime non installe - validation ignoree)")
        onnx_err = None

    # --- Rapport JSON ---
    print("\n[3/3] Rapport JSON...")
    report = {
        "checkpoint":        ckpt,
        "fp32_size_kb":      round(size_fp32, 2),
        "int8_size_kb":      round(size_int8, 2),
        "compression_ratio": round(compression, 3),
        "latency_fp32_ms":   round(lat_fp32, 4),
        "latency_int8_ms":   round(lat_int8, 4),
        "speedup":           round(speedup, 3),
        "max_action_error":  round(max_err, 8),
        "onnx_size_kb":      round(size_onnx, 2),
        "onnx_vs_pt_error":  round(onnx_err, 8) if onnx_err is not None else None,
    }

    report_path = os.path.join(args.out_dir, "quantize_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Rapport -> {report_path}")
    print("\n=== Resume quantification ===")
    print(f"  Compression  : x{compression:.2f}  ({size_fp32:.1f} KB -> {size_int8:.1f} KB)")
    print(f"  Speedup      : x{speedup:.2f}  ({lat_fp32:.3f} ms -> {lat_int8:.3f} ms)")
    level = "OK" if max_err < 0.1 else "elevee - QAT recommande"
    print(f"  Precision    : erreur max mu = {max_err:.4f}  [{level}]")
    print(f"  ONNX         : {size_onnx:.1f} KB  (deploiement edge/C++)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="results")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
