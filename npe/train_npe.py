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
from npe_simulator import IVIMNPESimulator, build_set_embedding


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
                        help="Hidden features for Neural Spline Flow (NSF).")
    parser.add_argument("--num-transforms", type=int, default=5,
                        help="Number of transforms for NSF.")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Training IVIM NPE: mode={args.mode}, budget={args.budget}, seed={args.seed}")
    print("=" * 70)

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    
    # 1. Prior
    prior, n_params, _ = get_processed_prior(device="cpu")
    print(f"Loaded {n_params}-D prior [D, Dstar, f]")

    # 2. Simulator
    print(f"Initializing simulator with representation={args.mode}...")
    sim = IVIMNPESimulator(representation=args.mode, clean=False, seed=args.seed)
    obs_shape = sim.observation_shape
    print(f"Individual observation shape: {obs_shape}")

    # 3. Simulate training data
    print(f"Simulating {args.budget} samples from prior...")
    t_sim_start = time.perf_counter()
    theta = prior.sample((args.budget,))
    obs, snr_ctx = sim(theta)
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
        model="nsf",
        embedding_net=wrapper_embedding,
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms
    )

    # 6. Initialize NPE and train
    inference = NPE(
        prior=prior,
        density_estimator=density_estimator_builder,
        device="cpu"
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
