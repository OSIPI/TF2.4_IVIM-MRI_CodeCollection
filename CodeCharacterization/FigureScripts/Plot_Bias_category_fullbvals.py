import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ============================================================
# SETTINGS
# ============================================================

SNR = 20

# Output of the SD/bias script (sheet "Category-wise (Eq2,3,8,9)")
INPUT_XLSX = f"Variability_Bias_SNR{SNR}_fullbvals.xlsx"
SHEET = "Category-wise (Eq2,3,8,9)"

# Column to plot: Eq. 8 ("All organs" rows) and Eq. 9 (organ rows)
VALUE_COL = "RMS_bias"

# Harmonization situations to plot (left -> right within each category)
harmonization_order = ["No harmonization", "Bounds + Initial guess"]

harm_colors = {
    "No harmonization": "#4C72B0",
    "Bounds + Initial guess": "#DD8452",
}

category_colors = {
    "Nonlinear LS": "#1f77b4",
    "Segmented Nonlinear LS": "#9467bd",
    "Segmented Linear LS": "#d62728",
    "Variable Projection": "#ff7f0e",
    "Bayesian": "#8c564b",
    "Neural network": "#e377c2",
}

# Order of the categories on the x-axis (categories with N < 2 are skipped)
category_order_all = [
    "All algorithms",
    "Nonlinear LS",
    "Segmented Nonlinear LS",
    "Variable Projection",
    "Segmented Linear LS",
    "Bayesian",
    "Neural network",
]

# Rows of the figure: (parameter column in the Excel file, display name, y-label)
parameter_items = [
    ("D_fitted", "D", "D\nRMS bias (mm²/s)"),
    ("f_fitted", "f", "f\nRMS bias"),
    ("Dp_fitted", r"D*", "D*\nRMS bias (mm²/s)"),
]

ALL_ORGANS = "All organs"

# Marker appearance
CAT_MARKER_SIZE = 110       # category-wise RMS bias (all organs)
ORGAN_MARKER_SIZE = 60      # organ-specific RMS bias
ORGAN_ALPHA = 0.35
JITTER = 0.25               # horizontal jitter for the organ points

group_spacing = 1.0
BAND_GAP = 0.1              # small gap between neighbouring category bands

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET)
df = df[df["Harmonization"].isin(harmonization_order)].copy()

# Keep only categories that exist and have more than one implementation
n_per_cat = df.groupby("Category")["N"].max()
category_order = [
    c for c in category_order_all
    if c in n_per_cat.index and n_per_cat[c] > 1
]

category_labels = [f"{c}\n(n={int(n_per_cat[c])})" for c in category_order]

n_harm = len(harmonization_order)
rng = np.random.default_rng(0)

# One marker shape per organ (category-wise "All organs" value stays a circle)
organ_markers = ["^", "s", "D", "v", "P", "X", "*", "<", ">", "h", "p", "8"]
organ_list = [o for o in df["Organ"].dropna().unique() if o != ALL_ORGANS]
organ_marker_map = {
    organ: organ_markers[i % len(organ_markers)]
    for i, organ in enumerate(organ_list)
}


def slot_position(cat_idx, h_idx):
    return cat_idx * (n_harm + group_spacing) + h_idx


# ============================================================
# HELPER: coloured category bands under the x-axis
# ============================================================

def add_category_bands(ax):
    trans = ax.get_xaxis_transform()
    band_y, band_height = -0.34, 0.34

    for cat_idx, category in enumerate(category_order):
        if category == "All algorithms":
            continue

        # span the full category column (separator line to separator line)
        start = slot_position(cat_idx, 0) - 0.5 - group_spacing / 2 + BAND_GAP / 2
        width = n_harm + group_spacing - BAND_GAP

        ax.add_patch(
            patches.Rectangle(
                (start, band_y), width, band_height,
                transform=trans,
                facecolor=category_colors.get(category, "#cccccc"),
                alpha=0.35, edgecolor="none", clip_on=False,
            )
        )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(13, 15), sharex=True)

for ax, (param, short_name, ylabel) in zip(axes, parameter_items):

    df_p = df[df["Parameter"] == param]
    ymax = 0

    for h_idx, harm in enumerate(harmonization_order):
        color = harm_colors[harm]
        df_h = df_p[df_p["Harmonization"] == harm]

        for cat_idx, category in enumerate(category_order):
            x0 = slot_position(cat_idx, h_idx)
            df_c = df_h[df_h["Category"] == category]

            # ---- organ-specific RMS bias (transparent, jittered) ----
            for organ in organ_list:
                organ_val = df_c.loc[df_c["Organ"] == organ, VALUE_COL].dropna().to_numpy()
                if len(organ_val) == 0:
                    continue
                xs = x0 + rng.uniform(-JITTER, JITTER, size=len(organ_val))
                ax.scatter(
                    xs, organ_val,
                    s=ORGAN_MARKER_SIZE, color=color,
                    marker=organ_marker_map[organ],
                    alpha=ORGAN_ALPHA, edgecolor="none", zorder=2,
                )
                ymax = max(ymax, organ_val.max())

            # ---- category-wise RMS bias over all organs (solid) ----
            all_val = df_c.loc[df_c["Organ"] == ALL_ORGANS, VALUE_COL].dropna().to_numpy()
            if len(all_val):
                ax.scatter(
                    [x0], all_val,
                    s=CAT_MARKER_SIZE, color=color,
                    edgecolor="black", linewidth=1.2, zorder=3,
                )
                ymax = max(ymax, all_val.max())

    # Y-axis (RMS bias is >= 0)
    ax.set_ylim(0, ymax * 1.1)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Light vertical separation between categories
    for cat_idx in range(1, len(category_order)):
        sep = slot_position(cat_idx, 0) - group_spacing / 2 - 0.5
        ax.axvline(sep, color="lightgray", linewidth=0.6, zorder=0)

# Bands only on the bottom axis
add_category_bands(axes[-1])

# ============================================================
# SHARED X AXIS
# ============================================================

tick_positions = [
    slot_position(c, 0) + (n_harm - 1) / 2 for c in range(len(category_order))
]
axes[-1].set_xticks(tick_positions)
axes[-1].set_xticklabels(category_labels, rotation=45, ha="center")

x_left = slot_position(0, 0) - 1.0
x_right = slot_position(len(category_order) - 1, n_harm - 1) + 1.0
axes[-1].set_xlim(x_left, x_right)

for ax in axes[:-1]:
    ax.tick_params(axis="x", which="both", length=0, labelbottom=False)

# ============================================================
# LEGEND
# ============================================================

legend_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=11,
           markerfacecolor=harm_colors[h], markeredgecolor="black", label=h)
    for h in harmonization_order
]
legend_handles += [
    Line2D([0], [0], marker="o", linestyle="", markersize=11,
           markerfacecolor="gray", markeredgecolor="black",
           label="All organs"),
]
legend_handles += [
    Line2D([0], [0], marker=organ_marker_map[o], linestyle="", markersize=9,
           markerfacecolor="gray", markeredgecolor="none", alpha=0.6,
           label=f"{o}")
    for o in organ_list
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    frameon=False,
    fontsize=13,
)

fig.subplots_adjust(hspace=0.08, bottom=0.20, top=0.98)

filename = f"RMSbias_per_category_SNR{SNR}_fullbvals.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved {filename}")