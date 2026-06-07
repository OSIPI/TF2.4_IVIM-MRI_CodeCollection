"""
run_e_cp2_check.py
==================
Smell test to verify trained NPE posteriors on held-out theta/SNR test cases.
Converts outputs to display units and prints posterior medians and 90% credible intervals.
"""
from __future__ import annotations

import argparse
import os
import sys
import torch
import numpy as np

from npe_prior import to_display, PARAM_NAMES, DISPLAY_UNITS
from npe_simulator import IVIMNPESimulator
from train_npe import pack_x, SNRWrapperEmbedding

# Resolve pickling namespace issue: the models were trained with train_npe.py run as __main__
sys.modules['__main__'].SNRWrapperEmbedding = SNRWrapperEmbedding


def run_smell_test(model_path: str, mode: str, seed: int = 1234) -> None:
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Train the model first.")
        return

    print("=" * 80)
    print(f"Running Smell Test for {mode.upper()} posterior loaded from {model_path}")
    print("=" * 80)

    # Load posterior
    posterior = torch.load(model_path, map_location="cpu", weights_only=False)

    # Define test cases: [D, Dstar, f] in absolute units
    # D: 0.8e-3 (display: 0.8), Dstar: 50e-3 (display: 50), f: 0.25 (display: 0.25) or 0.01 (display: 0.01)
    cases = [
        {
            "name": "Case 1: Identifiable, High SNR",
            "theta": torch.tensor([0.8e-3, 50.0e-3, 0.25]),
            "snr": 100.0
        },
        {
            "name": "Case 2: Identifiable, Low SNR",
            "theta": torch.tensor([0.8e-3, 50.0e-3, 0.25]),
            "snr": 10.0
        },
        {
            "name": "Case 3: Degenerate (f->0), High SNR",
            "theta": torch.tensor([0.8e-3, 50.0e-3, 0.01]),
            "snr": 100.0
        },
        {
            "name": "Case 4: Very Degenerate, Low SNR",
            "theta": torch.tensor([0.8e-3, 50.0e-3, 0.01]),
            "snr": 10.0
        }
    ]

    sim = IVIMNPESimulator(representation=mode, clean=False, seed=seed)

    for case in cases:
        print(f"\n--- {case['name']} (SNR={case['snr']}) ---")
        theta_true = case["theta"]
        theta_true_disp = to_display(theta_true.unsqueeze(0)).squeeze(0)

        # Generate observation
        obs, snr_ctx = sim(theta_true, snr=case["snr"])
        x = pack_x(obs, snr_ctx, mode)

        # Sample from posterior
        with torch.no_grad():
            samples = posterior.sample((10000,), x=x, show_progress_bars=False)

        # Convert to display units
        samples_disp = to_display(samples)

        # Compute stats
        medians = torch.median(samples_disp, dim=0).values
        p5 = torch.quantile(samples_disp, 0.05, dim=0)
        p95 = torch.quantile(samples_disp, 0.95, dim=0)
        widths = p95 - p5

        # Print results in a table
        print(f"  {'Param':<10} | {'True':>10} | {'Posterior Median':>18} | {'90% CI':>20} | {'90% Width':>10}")
        print("-" * 75)
        for i, name in enumerate(PARAM_NAMES):
            ci_str = f"[{p5[i]:.3f}, {p95[i]:.3f}]"
            print(f"  {DISPLAY_UNITS[i]:<10} | {theta_true_disp[i]:10.3f} | {medians[i]:18.3f} | {ci_str:>20} | {widths[i]:10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smell test on trained posterior.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pt)")
    parser.add_argument("--mode", type=str, required=True, choices=["masked_grid", "set"], help="Representation mode")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    args = parser.parse_args()

    run_smell_test(args.model, args.mode, seed=args.seed)


if __name__ == "__main__":
    main()
