"""
train_npe.py
============
Script to train the amortized neural posterior estimator (NPE) for biexponential IVIM.
Supports both "masked_grid" and "set" modes, and handles the folding of SNR context.
"""
from __future__ import annotations

import argparse
import json
import time
import os
import torch
import torch.nn as nn
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding

from npe_prior import get_processed_prior
from npe_prior_alt import get_alt_processed_prior
from npe_simulator import IVIMNPESimulator, build_set_embedding
from ivim_simulator import B_SCHEMES


class SNRWrapperEmbedding(nn.Module):
    """
    SNRWrapperEmbedding
    --------------------
    A wrapper embedding that takes input containing both observation signal and SNR.
    It passes the observation signal to the base embedding network (MLP or PermutationInvariantEmbedding),
    then appends the log10(SNR) scalar context directly to the output of the base embedding.

    Input shapes:
    - masked_grid: (B, 35) = [signal (17) | mask (17) | log10_snr (1)]
      Base embedding sees (B, 34).
    - set: (B, K, 3) = [b_i, S_i, log10_snr]
      Base embedding sees (B, K, 2).

    Conditioning dimension:
    - If base embedding output dimension is output_dim (default 20),
      the final output dimension is output_dim + 1 (default 21).
    """
    def __init__(self, base_embedding: nn.Module, mode: str, obs_dim: int = 34):
        super().__init__()
        self.base_embedding = base_embedding
        self.mode = mode
        self.obs_dim = obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x is batched to prevent dimension issues
        is_batched = (x.ndim == 2 and self.mode == "masked_grid") or (x.ndim == 3 and self.mode == "set")
        if not is_batched:
            x = x.unsqueeze(0)

        if self.mode == "masked_grid":
            obs = x[:, :self.obs_dim]
            log10_snr = x[:, self.obs_dim:]
        elif self.mode == "set":
            obs = x[:, :, :2]
            # Since SNR context is same for all points in the set, take the first one
            log10_snr = x[:, 0, 2:3]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        emb_out = self.base_embedding(obs)
        out = torch.cat([emb_out, log10_snr], dim=-1)

        if not is_batched:
            out = out.squeeze(0)
        return out


def pack_x(obs: torch.Tensor, snr_ctx: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Packs observation signal and log10_snr context into a single tensor.
    - masked_grid: (B, 34) and (B, 1) -> (B, 35)
    - set: (B, K, 2) and (B, 1) -> (B, K, 3)
    """
    if mode == "masked_grid":
        return torch.cat([obs, snr_ctx], dim=-1)
    elif mode == "set":
        B, K, _ = obs.shape
        # Expand snr_ctx (B, 1) to (B, K, 1) and concatenate
        snr_ctx_expanded = snr_ctx[:, None, :].expand(-1, K, -1)
        return torch.cat([obs, snr_ctx_expanded], dim=-1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IVIM NPE neural posterior estimator.")
    parser.add_argument("--mode", type=str, required=True, choices=["masked_grid", "set"],
                        help="Observation representation mode.")
    parser.add_argument("--budget", type=int, default=50000,
                        help="Number of simulations to generate.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum training epochs.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for the trained posterior (.pt).")
    parser.add_argument("--loss-output", type=str, required=True,
                        help="Output path for the training/validation loss JSON.")
    parser.add_argument("--embed-out-dim", type=int, default=20,
                        help="Output dimension of the base embedding net.")
    parser.add_argument("--hidden-features", type=int, default=50,
                        help="Hidden features for the flow transforms.")
    parser.add_argument("--num-transforms", type=int, default=5,
                        help="Number of flow transforms (NSF coupling blocks / MAF MADE layers).")
    parser.add_argument("--density-estimator", type=str, default="nsf",
                        choices=["nsf", "maf"],
                        help="Normalizing-flow family for the posterior density estimator: "
                             "'nsf' (neural spline flow, default — the main-result model) or "
                             "'maf' (masked autoregressive flow, for the architecture ablation). "
                             "Only this swaps; embedding, prior, conditioning and training loop are identical.")
    parser.add_argument("--log-dstar", action="store_true",
                        help="Use log10(Dstar) reparameterization.")
    parser.add_argument("--prior", type=str, default="boxuniform",
                        choices=["boxuniform", "tissue_dstar"],
                        help="Prior over [D, Dstar, f]: 'boxuniform' (default — the "
                             "log-uniform-style setB prior) or 'tissue_dstar' (the "
                             "prior-sensitivity ablation: D and f identical uniforms, but "
                             "log10(Dstar) ~ truncated Normal centered on a physiological "
                             "Dstar; requires --log-dstar). Only the prior changes; "
                             "architecture, budget, seed and b-scheme are untouched.")
    parser.add_argument("--b-scheme", type=str, default="clinical_sparse",
                        choices=sorted(B_SCHEMES.keys()),
                        help="Acquisition b-value scheme to train on (default: clinical_sparse, "
                             "the 8-point clinical protocol). 'dense' is the 16-point protocol used "
                             "for the acquisition-density supplementary experiment.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to train on (e.g. cpu, mps, cuda).")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Training IVIM NPE: mode={args.mode}, density_estimator={args.density_estimator}, "
          f"budget={args.budget}, seed={args.seed}")
    print("=" * 70)

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    
    # 1. Prior
    if args.prior == "tissue_dstar":
        if not args.log_dstar:
            parser.error("--prior tissue_dstar requires --log-dstar (its D* marginal is on log10(D*)).")
        prior, n_params, _ = get_alt_processed_prior(device="cpu", log_dstar=True)
        print(f"Loaded {n_params}-D TISSUE-INFORMED prior "
              f"[D~U, log10(Dstar)~TruncNormal, f~U]")
    else:
        prior, n_params, _ = get_processed_prior(device="cpu", log_dstar=args.log_dstar)
        print(f"Loaded {n_params}-D prior [D, {'log10(Dstar)' if args.log_dstar else 'Dstar'}, f]")

    # 2. Simulator
    active_bvals = B_SCHEMES[args.b_scheme]
    print(f"Initializing simulator with representation={args.mode}, b_scheme={args.b_scheme} "
          f"({active_bvals.size} b-values)...")
    sim = IVIMNPESimulator(representation=args.mode, clean=False,
                           active_bvals=active_bvals, seed=args.seed)
    obs_shape = sim.observation_shape
    print(f"Individual observation shape: {obs_shape}")

    # 3. Simulate training data
    print(f"Simulating {args.budget} samples from prior...")
    t_sim_start = time.perf_counter()
    theta = prior.sample((args.budget,))
    from npe_prior import invert_theta
    theta_abs = invert_theta(theta, log_dstar=args.log_dstar)
    obs, snr_ctx = sim(theta_abs)
    x = pack_x(obs, snr_ctx, args.mode)
    t_sim_end = time.perf_counter()
    print(f"Simulation took {t_sim_end - t_sim_start:.4f} seconds ({(t_sim_end - t_sim_start)/args.budget*1e6:.2f} us/sim)")
    print(f"Generated data: theta shape {theta.shape}, x shape {x.shape}")

    # 4. Define base embedding and wrapper embedding
    if args.mode == "masked_grid":
        base_embedding = FCEmbedding(
            input_dim=obs_shape[0],
            output_dim=args.embed_out_dim,
            num_layers=2,
            num_hiddens=40
        )
        wrapper_embedding = SNRWrapperEmbedding(
            base_embedding,
            mode=args.mode,
            obs_dim=obs_shape[0]
        )
    else:  # set
        base_embedding = build_set_embedding(
            trial_feature_dim=2,
            latent_dim=16,
            output_dim=args.embed_out_dim,
            num_layers_trial=2,
            num_hiddens_trial=40,
            num_layers_out=2,
            num_hiddens_out=40,
            aggregation_fn="mean",
            seed=args.seed
        )
        wrapper_embedding = SNRWrapperEmbedding(
            base_embedding,
            mode=args.mode
        )

    conditioning_dim = args.embed_out_dim + 1
    print(f"Wrapper embedding output dim (conditioning dim): {conditioning_dim}")

    # 5. Density estimator
    density_estimator_builder = posterior_nn(
        model=args.density_estimator,
        embedding_net=wrapper_embedding,
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms
    )

    # 6. Initialize NPE and train
    from torch.utils.tensorboard import SummaryWriter
    class DummySummaryWriter(SummaryWriter):
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def flush(self, *args, **kwargs):
            pass
        def close(self, *args, **kwargs):
            pass

    inference = NPE(
        prior=prior,
        density_estimator=density_estimator_builder,
        device=args.device,
        summary_writer=DummySummaryWriter()
    )
    inference.append_simulations(theta, x)

    print("Training density estimator...")
    t_train_start = time.perf_counter()
    density_estimator = inference.train(
        show_train_summary=True,
        max_num_epochs=args.epochs
    )
    t_train_end = time.perf_counter()
    wall_clock = t_train_end - t_train_start
    print(f"Training completed in {wall_clock:.2f} seconds.")

    # 7. Build and save posterior
    print(f"Building posterior and saving to {args.output}...")
    posterior = inference.build_posterior(density_estimator)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(posterior, args.output)

    # 8. Save loss curves and training stats
    train_loss = inference._summary["training_loss"]
    val_loss = inference._summary["validation_loss"]
    epochs_trained = sum(inference._summary["epochs_trained"])
    
    logs = {
        "mode": args.mode,
        "density_estimator": args.density_estimator,
        "prior": args.prior,
        "b_scheme": args.b_scheme,
        "hidden_features": args.hidden_features,
        "num_transforms": args.num_transforms,
        "log_dstar": args.log_dstar,
        "seed": args.seed,
        "budget": args.budget,
        "epochs_trained": epochs_trained,
        "wall_clock_sec": wall_clock,
        "per_epoch_sec": wall_clock / epochs_trained if epochs_trained > 0 else 0.0,
        "training_loss": train_loss,
        "validation_loss": val_loss
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(args.loss_output)), exist_ok=True)
    with open(args.loss_output, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"Loss curves and stats saved to {args.loss_output}")
    print("Done!")


if __name__ == "__main__":
    main()
