import torch
import pytest
from sbi.neural_nets.embedding_nets import FCEmbedding
from npe_simulator import build_set_embedding
# We will import from train_npe. Since train_npe is in PYTHONPATH, we can import it.
# Wait, let's write train_npe first, or write the test first since we are in execution.
# Importing from train_npe:
from train_npe import SNRWrapperEmbedding, pack_x

def test_pack_x():
    # masked_grid mode
    obs_grid = torch.randn(10, 34)
    snr_grid = torch.randn(10, 1)
    x_grid = pack_x(obs_grid, snr_grid, "masked_grid")
    assert x_grid.shape == (10, 35)

    # set mode
    obs_set = torch.randn(10, 8, 2)
    snr_set = torch.randn(10, 1)
    x_set = pack_x(obs_set, snr_set, "set")
    assert x_set.shape == (10, 8, 3)
    # Check that SNR is propagated correctly to all elements in the set
    assert torch.allclose(x_set[:, :, 2], snr_set.expand(-1, 8))

def test_snr_wrapper_embedding_masked_grid():
    base = FCEmbedding(input_dim=34, output_dim=20, num_layers=2, num_hiddens=40)
    wrapper = SNRWrapperEmbedding(base, mode="masked_grid", obs_dim=34)
    
    # Batched input
    x = torch.randn(5, 35)
    out = wrapper(x)
    assert out.shape == (5, 21)
    
    # Unbatched input
    x_single = torch.randn(35)
    out_single = wrapper(x_single)
    assert out_single.shape == (21,)

def test_snr_wrapper_embedding_set():
    base = build_set_embedding(
        trial_feature_dim=2,
        latent_dim=16,
        output_dim=20,
        num_layers_trial=2,
        num_hiddens_trial=40,
        num_layers_out=2,
        num_hiddens_out=40,
        aggregation_fn="mean"
    )
    wrapper = SNRWrapperEmbedding(base, mode="set")
    
    # Batched input
    x = torch.randn(5, 8, 3)
    out = wrapper(x)
    assert out.shape == (5, 21)
    
    # Unbatched input
    x_single = torch.randn(8, 3)
    out_single = wrapper(x_single)
    assert out_single.shape == (21,)
