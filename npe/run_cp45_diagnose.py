"""
run_cp45_diagnose.py
=====================
Diagnostic script for CP4.5 Checkpoint D0.
Probes the trained posterior setB model on SNR response and context isolation.
Saves a diagnostic plot and prints metrics.
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe_prior import to_display, PARAM_NAMES, DISPLAY_UNITS, invert_theta
from npe_simulator import IVIMNPESimulator
from train_npe import pack_x, SNRWrapperEmbedding
from run_cp3_validation import compute_crlb

# Resolve pickling namespace dynamically
sys.modules['__main__'].SNRWrapperEmbedding = SNRWrapperEmbedding


def run_diagnostics():
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_path = "npe/npe_posterior_setB.pt"
    if not os.path.exists(model_path):
        model_path = "npe_posterior_setB.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Could not find npe_posterior_setB.pt")

    print(f"Loading posterior from {model_path}...")
    posterior = torch.load(model_path, map_location="cpu", weights_only=False)

    # Clean simulator
    sim = IVIMNPESimulator(representation="set", clean=True, seed=seed)
    active_bvals = sim.active_bvals

    # 1. Choose a fixed regime recoverable at high SNR
    # D: 1.1 display, Dstar: 40 display, f: 0.25 (display: 0.25)
    # Convert to absolute units: D = 1.1e-3, Dstar = 40e-3, f = 0.25
    theta_true = torch.tensor([1.1e-3, 40e-3, 0.25], dtype=torch.float32)
    theta_true_disp = to_display(theta_true)
    print(f"True theta (display units): D={theta_true_disp[0]:.3f}, Dstar={theta_true_disp[1]:.3f}, f={theta_true_disp[2]:.3f}")

    # Generate one clean observation at SNR = 50 (to have a middle ground (b, S))
    obs, _ = sim(theta_true, snr=50.0) # shape (1, K_active, 2)
    print(f"Generated clean observation (shape: {obs.shape})")

    # SNR sweep range: 10 to 100, 8 points
    snrs = np.linspace(10.0, 100.0, 8)
    print(f"Sweeping SNRs: {snrs}")

    widths = {name: [] for name in PARAM_NAMES}
    medians = {name: [] for name in PARAM_NAMES}
    crlb_widths = {name: [] for name in PARAM_NAMES} # theoretical 90% width = 2 * 1.645 * CRLB_SD

    # We will also check context-isolation by comparing samples from first and last SNR
    first_samples = None
    last_samples = None

    # Auto-detect log_dstar
    log_dstar = bool(posterior.prior.support.base_constraint.lower_bound[1] < 0)
    print(f"Detected log_dstar={log_dstar} from posterior prior bounds")

    for idx, snr in enumerate(snrs):
        # Vary only the log10_snr context
        snr_ctx = torch.tensor([[np.log10(snr)]], dtype=torch.float32)
        x = pack_x(obs, snr_ctx, "set")

        # Sample the posterior
        with torch.no_grad():
            samples = posterior.sample((10000,), x=x, show_progress_bars=False) # (10000, 3)

        samples_abs = invert_theta(samples, log_dstar=log_dstar)
        samples_disp = to_display(samples_abs)

        if idx == 0:
            first_samples = samples_disp
        if idx == len(snrs) - 1:
            last_samples = samples_disp

        # Compute 90% marginal widths (p95 - p5) and medians
        p5 = torch.quantile(samples_disp, 0.05, dim=0)
        p95 = torch.quantile(samples_disp, 0.95, dim=0)
        p50 = torch.quantile(samples_disp, 0.50, dim=0)

        # Compute analytical CRLB for this SNR
        crlb_abs = compute_crlb(theta_true.numpy(), active_bvals, snr)
        crlb_disp = crlb_abs * np.array([1000.0, 1000.0, 1.0])

        for i, name in enumerate(PARAM_NAMES):
            widths[name].append((p95[i] - p5[i]).item())
            medians[name].append(p50[i].item())
            # For a Gaussian distribution, the 90% width is ~3.29 * SD
            crlb_widths[name].append(3.29 * crlb_disp[i])

    # Print results in a table
    print("\n" + "="*80)
    print("SNR SWEEP PROBE RESULTS")
    print("="*80)
    print(f"{'SNR':>6} | {'D 90% W':>10} (CRLB) | {'D* 90% W':>10} (CRLB) | {'f 90% W':>10} (CRLB)")
    print("-" * 80)
    for idx, snr in enumerate(snrs):
        print(f"{snr:6.1f} | "
              f"{widths['D'][idx]:10.3f} ({crlb_widths['D'][idx]:.3f}) | "
              f"{widths['Dstar'][idx]:10.3f} ({crlb_widths['Dstar'][idx]:.3f}) | "
              f"{widths['f'][idx]:10.3f} ({crlb_widths['f'][idx]:.3f})")
    print("="*80)

    # Check context-isolation quantitatively: compute difference in mean/median between SNR=10 and SNR=100
    mean_diff = torch.mean(last_samples - first_samples, dim=0)
    std_diff = torch.std(last_samples, dim=0) - torch.std(first_samples, dim=0)
    print("\nQuantitative change in marginals between SNR=10 and SNR=100:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name}: mean diff = {mean_diff[i]:.6f}, std diff = {std_diff[i]:.6f}")

    # Plot width vs SNR per parameter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, name in enumerate(PARAM_NAMES):
        ax = axes[i]
        ax.plot(snrs, widths[name], 'bo-', label='NPE 90% Width')
        ax.plot(snrs, crlb_widths[name], 'r--', label='CRLB 90% Width (Gaussian approximation)')
        ax.set_title(DISPLAY_UNITS[i])
        ax.set_xlabel('SNR')
        ax.set_ylabel('90% Width')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("90% Marginal Posterior Width vs SNR (Held-out Observation fixed at theta_true)", fontsize=14)
    fig.tight_layout()
    plot_path = "npe/diagnose_snr_width.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved diagnostic plot to {plot_path}")


if __name__ == "__main__":
    run_diagnostics()
