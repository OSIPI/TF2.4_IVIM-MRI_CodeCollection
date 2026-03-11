"""
Body-part specific IVIM parameter defaults.

Literature-based initial guesses for different anatomical regions,
using expected healthy-tissue means from the upcoming Sigmund et al.
IVIM consensus recommendations paper ("Towards Clinical Translation
of Intravoxel Incoherent Motion MRI: Acquisition and Analysis
Consensus Recommendations", JMRI).

Bounds are intentionally set to broad physical limits for ALL organs
to preserve sensitivity to lesions (which deviate from healthy tissue).
The organ-specific bounds structure is retained so that organ-specific
bounds can be introduced at a later stage by changing these numbers.

References:
    [1] Vieni 2020 – Brain  (DOI: 10.1016/j.neuroimage.2019.116228)
    [2] Ljimani 2020 – Kidney  (DOI: 10.1007/s10334-019-00790-y)
    [3] Li 2017 – Liver  (DOI: 10.21037/qims.2017.02.03)
    [4] Englund 2022 – Muscle  (DOI: 10.1002/jmri.27876)
    [5] Liang 2020 – Breast  (DOI: 10.3389/fonc.2020.585486)
    [6] Zhu 2021 – Pancreas  (DOI: 10.1007/s00330-021-07891-9)
"""

import warnings
import copy

# Broad physical bounds applied to every organ.
# These are intentionally wide to avoid restricting lesion contrast.
_BROAD_BOUNDS = {
    "S0": [0.5, 1.5],
    "f": [0, 1.0],
    "Dp": [0.005, 0.2],
    "D": [0, 0.005],
}

def _broad_bounds():
    """Return a fresh deep copy of the broad bounds dict."""
    return copy.deepcopy(_BROAD_BOUNDS)

IVIM_BODY_PART_DEFAULTS = {
    "brain": {
        "initial_guess": {"S0": 1.0, "f": 0.0764, "Dp": 0.01088, "D": 0.00083},
        "bounds": _broad_bounds(),
        "thresholds": [200],
    },
    "kidney": {
        "initial_guess": {"S0": 1.0, "f": 0.1888, "Dp": 0.04053, "D": 0.00189},
        "bounds": _broad_bounds(),
        "thresholds": [200],
    },
    "liver": {
        "initial_guess": {"S0": 1.0, "f": 0.2305, "Dp": 0.07002, "D": 0.00109},
        "bounds": _broad_bounds(),
        "thresholds": [200],
    },
    "muscle": {
        "initial_guess": {"S0": 1.0, "f": 0.1034, "Dp": 0.03088, "D": 0.00147},
        "bounds": _broad_bounds(),
        "thresholds": [200],
    },
    "breast_benign": {
        "initial_guess": {"S0": 1.0, "f": 0.0700, "Dp": 0.05233, "D": 0.00143},
        "bounds": _broad_bounds(),
        "thresholds": [200],
    },
    "breast_malignant": {
        "initial_guess": {"S0": 1.0, "f": 0.1131, "Dp": 0.03776, "D": 0.00097},
        "bounds": _broad_bounds(),
        "thresholds": [200],
    },
    "pancreas_benign": {
        "initial_guess": {"S0": 1.0, "f": 0.2003, "Dp": 0.02539, "D": 0.00141},
        "bounds": _broad_bounds(),
        "thresholds": [200],
    },
    "pancreas_malignant": {
        "initial_guess": {"S0": 1.0, "f": 0.1239, "Dp": 0.02216, "D": 0.00140},
        "bounds": _broad_bounds(),
        "thresholds": [200],
    },
}

# Keep the current universal defaults as "generic"
IVIM_BODY_PART_DEFAULTS["generic"] = {
    "initial_guess": {"S0": 1.0, "f": 0.1, "Dp": 0.01, "D": 0.001},
    "bounds": dict(_BROAD_BOUNDS),
    "thresholds": [200],
}


def get_body_part_defaults(body_part):
    """Get IVIM default parameters for a given body part.

    Args:
        body_part (str): Name of the body part (e.g., "brain", "liver", "kidney").
                         Case-insensitive. Spaces and hyphens are normalized to
                         underscores (e.g., "breast benign" -> "breast_benign").

    Returns:
        dict: Dictionary with keys "initial_guess", "bounds", and "thresholds".

    Raises:
        ValueError: If the body part is not in the lookup table.
    """
    key = body_part.lower().replace(" ", "_").replace("-", "_")
    if key not in IVIM_BODY_PART_DEFAULTS:
        available = ", ".join(sorted(IVIM_BODY_PART_DEFAULTS.keys()))
        raise ValueError(
            f"Unknown body part '{body_part}'. "
            f"Available body parts: {available}"
        )

    # Emit warning when organ-specific preset is selected (not for "generic")
    if key != "generic":
        warnings.warn(
            f"Organ-specific preset '{body_part}' selected. "
            "Initial guesses are based on healthy tissue means from the "
            "IVIM consensus recommendations (Sigmund et al.) and references "
            "therein. Fitting bounds are currently set to broad physical "
            "limits and are not organ-specific.",
            UserWarning,
            stacklevel=2,
        )

    return copy.deepcopy(IVIM_BODY_PART_DEFAULTS[key])


def get_available_body_parts():
    """Return a sorted list of all available body part names.

    Returns:
        list: Sorted list of body part name strings.
    """
    return sorted(IVIM_BODY_PART_DEFAULTS.keys())
