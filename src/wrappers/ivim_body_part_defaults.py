"""
Body-part specific IVIM parameter defaults.

Literature-based initial guesses, bounds, and thresholds for
different anatomical regions. Used by OsipiBase when the user
specifies a body_part parameter.

References:
    Brain:    Federau 2017 (DOI: 10.1002/nbm.3780)
    Liver:    Dyvorne 2013 (DOI: 10.1016/j.ejrad.2013.03.003),
              Guiu 2012 (DOI: 10.1002/jmri.23762)
    Kidney:   Li 2017 (DOI: 10.1002/jmri.25571),
              Ljimani 2020 (DOI: 10.1007/s10334-019-00802-0)
    Prostate: Kuru 2014 (DOI: 10.1007/s00330-014-3165-y)
    Pancreas: Barbieri 2020 (DOI: 10.1002/mrm.27910)
    Head/Neck: Sumi 2012 (DOI: 10.1259/dmfr/15696758)
    Breast:   Lee 2018 (DOI: 10.1097/RCT.0000000000000661)
    Placenta: Zhu 2023 (DOI: 10.1002/jmri.28858)
"""

IVIM_BODY_PART_DEFAULTS = {
    "brain": {
        "initial_guess": {"S0": 1.0, "f": 0.05, "Dp": 0.01, "D": 0.0008},
        "bounds": {
            "S0": [0.7, 1.3],
            "f": [0.0, 0.15],
            "Dp": [0.005, 0.05],
            "D": [0.0003, 0.002],
        },
        "thresholds": [200],
    },
    "liver": {
        "initial_guess": {"S0": 1.0, "f": 0.12, "Dp": 0.06, "D": 0.001},
        "bounds": {
            "S0": [0.7, 1.3],
            "f": [0.0, 0.40],
            "Dp": [0.01, 0.15],
            "D": [0.0003, 0.003],
        },
        "thresholds": [200],
    },
    "kidney": {
        "initial_guess": {"S0": 1.0, "f": 0.20, "Dp": 0.03, "D": 0.0019},
        "bounds": {
            "S0": [0.7, 1.3],
            "f": [0.0, 0.50],
            "Dp": [0.01, 0.08],
            "D": [0.0005, 0.004],
        },
        "thresholds": [200],
    },
    "prostate": {
        "initial_guess": {"S0": 1.0, "f": 0.08, "Dp": 0.025, "D": 0.0015},
        "bounds": {
            "S0": [0.7, 1.3],
            "f": [0.0, 0.25],
            "Dp": [0.005, 0.06],
            "D": [0.0003, 0.003],
        },
        "thresholds": [200],
    },
    "pancreas": {
        "initial_guess": {"S0": 1.0, "f": 0.18, "Dp": 0.02, "D": 0.0012},
        "bounds": {
            "S0": [0.7, 1.3],
            "f": [0.0, 0.50],
            "Dp": [0.005, 0.06],
            "D": [0.0003, 0.003],
        },
        "thresholds": [200],
    },
    "head_and_neck": {
        "initial_guess": {"S0": 1.0, "f": 0.15, "Dp": 0.025, "D": 0.001},
        "bounds": {
            "S0": [0.7, 1.3],
            "f": [0.0, 0.40],
            "Dp": [0.005, 0.08],
            "D": [0.0003, 0.003],
        },
        "thresholds": [200],
    },
    "breast": {
        "initial_guess": {"S0": 1.0, "f": 0.10, "Dp": 0.02, "D": 0.0014},
        "bounds": {
            "S0": [0.7, 1.3],
            "f": [0.0, 0.30],
            "Dp": [0.005, 0.06],
            "D": [0.0004, 0.003],
        },
        "thresholds": [200],
    },
    "placenta": {
        "initial_guess": {"S0": 1.0, "f": 0.28, "Dp": 0.04, "D": 0.0017},
        "bounds": {
            "S0": [0.7, 1.3],
            "f": [0.05, 0.60],
            "Dp": [0.01, 0.1],
            "D": [0.0005, 0.004],
        },
        "thresholds": [200],
    },
}

# Keep the current universal defaults as "generic"
IVIM_BODY_PART_DEFAULTS["generic"] = {
    "initial_guess": {"S0": 1.0, "f": 0.1, "Dp": 0.01, "D": 0.001},
    "bounds": {
        "S0": [0.7, 1.3],
        "f": [0, 1.0],
        "Dp": [0.005, 0.2],
        "D": [0, 0.005],
    },
    "thresholds": [200],
}


def get_body_part_defaults(body_part):
    """Get IVIM default parameters for a given body part.

    Args:
        body_part (str): Name of the body part (e.g., "brain", "liver", "kidney").
                         Case-insensitive. Spaces and hyphens are normalized to
                         underscores (e.g., "head and neck" -> "head_and_neck").

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
    return IVIM_BODY_PART_DEFAULTS[key]


def get_available_body_parts():
    """Return a sorted list of all available body part names.

    Returns:
        list: Sorted list of body part name strings.
    """
    return sorted(IVIM_BODY_PART_DEFAULTS.keys())
