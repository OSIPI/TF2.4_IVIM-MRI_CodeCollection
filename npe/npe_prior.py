"""
npe_prior.py
============
BoxUniform prior over the IVIM biexponential parameters

    theta = [D, Dstar, f]

for amortized neural posterior estimation (NPE). This is Step 2 of the Phase-E
NPE scaffold: the prior is the *sole gatekeeper* of parameter ranges (the forward
model in ``ivim_simulator`` does no internal clamping), so its bounds define the
entire region the amortized posterior will ever be asked about.

UNITS -- absolute, matching ``ivim_simulator.ivim_signal``:
    D, Dstar : mm^2/s     (NOT scaled; e.g. tissue D ~ 0.7e-3, D* ~ 30e-3)
    f        : unitless    (perfusion fraction)
    b        : s/mm^2      (so the product b*D is dimensionless)

The box deliberately spans two *degenerate* regimes so the trained posterior has
to represent the resulting uncertainty rather than being shielded from it:
    * f -> 0      : the perfusion compartment vanishes and the signal collapses to
                    a mono-exponential, leaving Dstar unidentifiable.
    * Dstar -> D  : the two compartments coalesce; the biexponential becomes
                    indistinguishable from a mono-exponential.

DISPLAY SCALING (the *reporting boundary*)
------------------------------------------
Diffusivities are conventionally *reported* in units of 1e-3 mm^2/s (== um^2/ms).
Everything in this project stays in absolute mm^2/s internally; we multiply D and
Dstar by ``DISPLAY_SCALE`` (= 1000) ONLY at the moment we print or plot. ``f`` is
unitless and is never scaled. Never feed display-scaled values back into the
simulator or the prior -- ``to_display`` is a one-way reporting convenience.
"""
from __future__ import annotations

import torch
from sbi.utils import BoxUniform, process_prior

# theta = [D, Dstar, f], absolute units (mm^2/s, mm^2/s, unitless).
PARAM_NAMES = ("D", "Dstar", "f")
N_PARAMS = 3

# Prior box (absolute units). See the module docstring for the regime rationale.
PRIOR_LOW = (0.2e-3, 3.0e-3, 0.0)
PRIOR_HIGH = (3.0e-3, 0.15, 0.5)

# Conventional display units for the diffusivities: 1e-3 mm^2/s (== um^2/ms).
# Applied ONLY at the reporting boundary; f (index 2) is never scaled.
DISPLAY_SCALE = (1000.0, 1000.0, 1.0)
DISPLAY_UNITS = ("D [1e-3 mm^2/s]", "Dstar [1e-3 mm^2/s]", "f [-]")


def get_prior(device: str = "cpu") -> BoxUniform:
    """The ``BoxUniform`` prior over ``theta = [D, Dstar, f]`` in absolute units."""
    low = torch.as_tensor(PRIOR_LOW, dtype=torch.float32, device=device)
    high = torch.as_tensor(PRIOR_HIGH, dtype=torch.float32, device=device)
    return BoxUniform(low=low, high=high)


def get_processed_prior(device: str = "cpu"):
    """Prior passed through sbi's ``process_prior``, ready for the inference object.

    Returns the sbi triple ``(prior, num_parameters, prior_returns_numpy)``;
    ``num_parameters`` is 3 here. Hand this straight to ``NPE(prior=...)`` etc.
    """
    return process_prior(get_prior(device=device))


def to_display(theta) -> torch.Tensor:
    """Scale absolute-unit ``theta`` to conventional *display* units (reporting only).

    D and Dstar are multiplied by 1000 (mm^2/s -> 1e-3 mm^2/s == um^2/ms); f is
    left unchanged. Accepts anything tensor-like of shape ``(..., 3)``. The result
    is for printing/plotting ONLY -- do not feed it back into the simulator/prior.
    """
    theta = torch.as_tensor(theta, dtype=torch.float32)
    scale = torch.as_tensor(DISPLAY_SCALE, dtype=torch.float32, device=theta.device)
    return theta * scale
