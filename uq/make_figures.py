"""
make_figures.py
===============
W3 Checkpoint 3 — figures from calib_w3.csv.

Emits four artifacts from the long-form calibration table:
  fig_reliability.png        -- reliability diagrams, one panel per paradigm
  fig_calibration_heatmap.png-- method x cell coverage-gap heatmap (per param)
  reliability_diagrams.jsx   -- same reliability view as a self-contained React/SVG
                                component (data baked in, no chart deps)
  calibration_heatmap.jsx    -- same heatmap as a self-contained React/SVG component

Both views show the headline result: for D*, the Gaussian uncertainties
(Laplace / MCMC posterior SD) under-cover because the posterior is skewed and
bound-pinned, while the MCMC 2.5/97.5 quantile interval recovers nominal coverage.

Run after run_w3_calib.py:
    .venv/bin/python make_figures.py            # reads ./calib_w3.csv
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = sys.argv[1] if len(sys.argv) > 1 else "calib_w3.csv"
# Where to write the PNG / JSX outputs. argv[2] > $FIG_OUTDIR > cwd.
OUTDIR = (sys.argv[2] if len(sys.argv) > 2 else os.environ.get("FIG_OUTDIR", "."))
os.makedirs(OUTDIR, exist_ok=True)


def _out(name):
    return os.path.join(OUTDIR, name)

# --- Project Fashion palette --------------------------------------------------
PAL = dict(bg="#0a0e14", panel="#11161f", ink="#e7e3d8", amber="#e8a33d",
           teal="#3ec5b6", red="#e5705a", violet="#9d8cff", green="#6fcf73",
           grid="#243042", mute="#7c8696")
PARAM_COLOR = {"D": PAL["teal"], "Dstar": PAL["red"], "f": PAL["violet"]}
PARAM_LABEL = {"D": "D", "Dstar": "D*", "f": "f"}
NOMINALS = [0.50, 0.80, 0.90, 0.95]

PARADIGMS = [
    ("Classical", ["bootstrap"]),
    ("Bayesian", ["laplace", "mcmc", "mcmc_quantile"]),
    ("Deep learning", ["input_perturbation", "ensemble"]),
]
GEN_STYLE = {  # (linestyle, marker, display)
    "bootstrap":          ("-",  "o", "bootstrap"),
    "laplace":            ("--", "s", "Laplace SD"),
    "mcmc":               ("-.", "^", "MCMC SD"),
    "mcmc_quantile":      ("-",  "D", "MCMC 2.5/97.5"),
    "input_perturbation": ("-",  "o", "input-perturb"),
    "ensemble":           (":",  "x", "ensemble (epist.)"),
}

plt.rcParams.update({
    "font.family": ["JetBrains Mono", "DejaVu Sans Mono", "monospace"],
    "figure.facecolor": PAL["bg"], "axes.facecolor": PAL["panel"],
    "savefig.facecolor": PAL["bg"], "text.color": PAL["ink"],
    "axes.labelcolor": PAL["ink"], "xtick.color": PAL["ink"],
    "ytick.color": PAL["ink"], "axes.edgecolor": PAL["grid"],
    "grid.color": PAL["grid"], "font.size": 9,
})


def load():
    if not os.path.exists(CSV):
        sys.exit(f"make_figures: {CSV} not found — run run_w3_calib.py first.")
    df = pd.read_csv(CSV)
    df["param"] = df["param"].astype(str)
    return df


def _curve(df, generator, param):
    """Mean empirical coverage over cells at each nominal for one generator/param."""
    sub = df[(df.generator == generator) & (df.param == param)]
    xs, ys = [], []
    for L in NOMINALS:
        v = sub[np.isclose(sub.nominal, L)]["coverage"].dropna()
        if len(v):
            xs.append(L); ys.append(float(v.mean()))
    return xs, ys


# ============================ reliability PNG =================================
def fig_reliability(df):
    paradigms = [(name, [g for g in gens if g in set(df.generator)])
                 for name, gens in PARADIGMS]
    paradigms = [(n, g) for n, g in paradigms if g]
    npan = max(len(paradigms), 1)
    fig, axes = plt.subplots(1, npan, figsize=(4.7 * npan, 4.6), squeeze=False)
    axes = axes[0]

    for ax, (name, gens) in zip(axes, paradigms):
        ax.plot([0, 1], [0, 1], color=PAL["mute"], ls=(0, (4, 4)), lw=1, zorder=1)
        for gen in gens:
            ls, mk, _ = GEN_STYLE[gen]
            for param in ("D", "Dstar", "f"):
                xs, ys = _curve(df, gen, param)
                if not xs:
                    continue
                is_star = param == "Dstar"
                ax.plot(xs, ys, ls=ls, marker=mk, ms=5 if is_star else 4,
                        color=PARAM_COLOR[param], lw=2.4 if is_star else 1.2,
                        alpha=1.0 if is_star else 0.45, zorder=3 if is_star else 2)
        ax.set_title(name, color=PAL["amber"], fontsize=11, pad=8)
        ax.set_xlim(0.4, 1.0); ax.set_ylim(0, 1.02)
        ax.set_xlabel("nominal coverage")
        ax.grid(True, lw=0.5, alpha=0.4)
    axes[0].set_ylabel("empirical coverage")

    param_handles = [Line2D([0], [0], color=PARAM_COLOR[p], lw=3,
                            label=f"{PARAM_LABEL[p]}" + (" (skew focus)" if p == "Dstar" else ""))
                     for p in ("D", "Dstar", "f")]
    gen_handles = [Line2D([0], [0], color=PAL["ink"], ls=GEN_STYLE[g][0],
                          marker=GEN_STYLE[g][1], label=GEN_STYLE[g][2])
                   for _, gens in paradigms for g in gens]
    seen, gh = set(), []
    for h in gen_handles:
        if h.get_label() not in seen:
            seen.add(h.get_label()); gh.append(h)
    leg1 = axes[-1].legend(handles=param_handles, loc="lower right", fontsize=8,
                           facecolor=PAL["panel"], edgecolor=PAL["grid"],
                           labelcolor=PAL["ink"], title="parameter")
    leg1.get_title().set_color(PAL["mute"])
    axes[-1].add_artist(leg1)
    axes[0].legend(handles=gh, loc="upper left", fontsize=7.5,
                   facecolor=PAL["panel"], edgecolor=PAL["grid"], labelcolor=PAL["ink"])
    fig.suptitle("Reliability diagrams — predicted vs empirical coverage",
                 color=PAL["ink"], fontsize=13, y=0.99)
    fig.text(0.5, 0.005, "D* Gaussian SD (Laplace/MCMC) sags below the diagonal; "
             "MCMC quantile interval recovers it.",
             ha="center", color=PAL["mute"], fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(_out("fig_reliability.png"), dpi=300)
    plt.close(fig)
    print("wrote " + _out("fig_reliability.png"), flush=True)


# ============================ heatmap PNG ====================================
def _gap_matrix(df, param):
    sub = df[(df.param == param) & np.isclose(df.nominal, 0.95)].copy()
    sub["row"] = sub["method"].str.replace("OGC_AmsterdamUMC_", "OGC_", regex=False) \
                              .str.replace("_biexp", "", regex=False) + " · " + sub["generator"]
    rows = sorted(sub["row"].unique())
    cells = sorted(sub["cell"].unique())
    M = np.full((len(rows), len(cells)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cells):
            v = sub[(sub.row == r) & (sub.cell == c)]["coverage"].dropna()
            if len(v):
                M[i, j] = float(v.mean()) - 0.95
    return rows, cells, M


def fig_heatmap(df):
    params = ["D", "Dstar", "f"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), squeeze=False)
    axes = axes[0]
    vlim = 0.45
    im = None
    for ax, param in zip(axes, params):
        rows, cells, M = _gap_matrix(df, param)
        im = ax.imshow(M, cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows if ax is axes[0] else [""] * len(rows), fontsize=7)
        ax.set_title(f"{PARAM_LABEL[param]}", color=PAL["amber"], fontsize=12)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                            fontsize=6, color="#101010" if abs(M[i, j]) < 0.28 else "#f5f5f5")
        ax.set_xlabel("cell")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("coverage − nominal (0.95)", color=PAL["ink"], fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=PAL["ink"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=PAL["ink"])
    fig.suptitle("Calibration heatmap — coverage gap at nominal 0.95 "
                 "(red = under-covers, blue = over-covers)",
                 color=PAL["ink"], fontsize=13)
    fig.savefig(_out("fig_calibration_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote " + _out("fig_calibration_heatmap.png"), flush=True)


# ============================ JSX emitters ===================================
def _reliability_data(df):
    out = []
    for name, gens in PARADIGMS:
        gens = [g for g in gens if g in set(df.generator)]
        series = []
        for gen in gens:
            for param in ("D", "Dstar", "f"):
                xs, ys = _curve(df, gen, param)
                if xs:
                    series.append(dict(generator=gen, param=param,
                                       points=[[x, y] for x, y in zip(xs, ys)]))
        if series:
            out.append(dict(paradigm=name, series=series))
    return out


def _heatmap_data(df):
    out = {}
    for param in ("D", "Dstar", "f"):
        rows, cells, M = _gap_matrix(df, param)
        out[param] = dict(rows=rows, cells=cells,
                          matrix=[[None if not np.isfinite(v) else round(float(v), 4)
                                   for v in r] for r in M])
    return out


JSX_RELIABILITY = '''// reliability_diagrams.jsx — Project Fashion W3
// Auto-generated by make_figures.py from calib_w3.csv. Self-contained (pure SVG).
import React from "react";

const PAL = {bg:"#0a0e14", panel:"#11161f", ink:"#e7e3d8", amber:"#e8a33d",
  teal:"#3ec5b6", red:"#e5705a", violet:"#9d8cff", green:"#6fcf73", mute:"#7c8696", grid:"#243042"};
const PARAM_COLOR = {D: PAL.teal, Dstar: PAL.red, f: PAL.violet};
const PARAM_LABEL = {D: "D", Dstar: "D*", f: "f"};
const GEN_DASH = {bootstrap:"", laplace:"6 5", mcmc:"2 4", mcmc_quantile:"",
  input_perturbation:"", ensemble:"2 3"};
const DATA = __DATA__;

function Panel({paradigm, series}) {
  const W = 300, H = 300, m = {l: 46, r: 12, t: 30, b: 40};
  const x0 = 0.4, x1 = 1.0, y0 = 0, y1 = 1.02;
  const sx = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const sy = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  return (
    <svg width={W} height={H} style={{background: PAL.panel, borderRadius: 8}}>
      <text x={W/2} y={18} fill={PAL.amber} fontSize="13" textAnchor="middle"
        fontFamily="JetBrains Mono, monospace">{paradigm}</text>
      <line x1={sx(x0)} y1={sy(x0)} x2={sx(x1)} y2={sy(x1)} stroke={PAL.mute}
        strokeDasharray="4 4" strokeWidth="1"/>
      {series.map((s, i) => {
        const star = s.param === "Dstar";
        const d = s.points.map((p, k) => `${k ? "L" : "M"}${sx(p[0])},${sy(p[1])}`).join(" ");
        return (<g key={i}>
          <path d={d} fill="none" stroke={PARAM_COLOR[s.param]}
            strokeWidth={star ? 2.6 : 1.2} strokeOpacity={star ? 1 : 0.45}
            strokeDasharray={GEN_DASH[s.generator] || ""}/>
          {s.points.map((p, k) => (<circle key={k} cx={sx(p[0])} cy={sy(p[1])}
            r={star ? 3 : 2} fill={PARAM_COLOR[s.param]} fillOpacity={star ? 1 : 0.5}/>))}
        </g>);
      })}
      {[0.5,0.8,0.9,0.95].map(t => (<text key={t} x={sx(t)} y={H-22} fill={PAL.ink}
        fontSize="8" textAnchor="middle" fontFamily="monospace">{t}</text>))}
      {[0,0.5,1].map(t => (<text key={t} x={m.l-6} y={sy(t)+3} fill={PAL.ink}
        fontSize="8" textAnchor="end" fontFamily="monospace">{t}</text>))}
      <text x={W/2} y={H-4} fill={PAL.mute} fontSize="8" textAnchor="middle">nominal</text>
    </svg>
  );
}

export default function ReliabilityDiagrams() {
  return (
    <div style={{background: PAL.bg, padding: 20, color: PAL.ink,
      fontFamily: "JetBrains Mono, monospace"}}>
      <h2 style={{color: PAL.ink, margin: "0 0 4px"}}>Reliability diagrams</h2>
      <p style={{color: PAL.mute, marginTop: 0, fontSize: 12}}>
        Predicted vs empirical coverage. D* (red, bold): Gaussian SD sags below the
        diagonal; MCMC 2.5/97.5 quantile interval recovers it.</p>
      <div style={{display: "flex", gap: 14, flexWrap: "wrap"}}>
        {DATA.map((p, i) => <Panel key={i} {...p}/>)}
      </div>
    </div>
  );
}
'''

JSX_HEATMAP = '''// calibration_heatmap.jsx — Project Fashion W3
// Auto-generated by make_figures.py from calib_w3.csv. Self-contained (pure SVG).
import React from "react";

const PAL = {bg:"#0a0e14", panel:"#11161f", ink:"#e7e3d8", amber:"#e8a33d", mute:"#7c8696"};
const PARAM_LABEL = {D: "D", Dstar: "D*", f: "f"};
const DATA = __DATA__;

// diverging red(-)..white(0)..blue(+), domain +/-0.45 (coverage - 0.95)
// matches matplotlib RdBu: negative gap (under-coverage) = red, positive = blue
function divColor(v) {
  if (v === null || v === undefined) return PAL.panel;
  const t = Math.max(-1, Math.min(1, v / 0.45));
  const lerp = (a, b, u) => Math.round(a + (b - a) * u);
  if (t < 0) { const u = -t; return `rgb(${lerp(247,178,u)},${lerp(247,24,u)},${lerp(247,43,u)})`; }
  const u = t; return `rgb(${lerp(247,49,u)},${lerp(247,130,u)},${lerp(247,189,u)})`;
}

function Heat({param, rows, cells, matrix}) {
  const cw = 54, ch = 22, lw = 150, th = 26;
  const W = lw + cells.length * cw + 12, H = th + rows.length * ch + 36;
  return (
    <div style={{marginBottom: 8}}>
      <div style={{color: PAL.amber, fontSize: 13, margin: "6px 0"}}>{PARAM_LABEL[param]}</div>
      <svg width={W} height={H} style={{background: PAL.panel, borderRadius: 8}}>
        {cells.map((c, j) => (<text key={j} x={lw + j*cw + cw/2} y={th-8} fill={PAL.ink}
          fontSize="8" textAnchor="middle" transform={`rotate(-30 ${lw+j*cw+cw/2} ${th-8})`}>{c}</text>))}
        {rows.map((r, i) => (<g key={i}>
          <text x={lw-6} y={th + i*ch + ch/2 + 3} fill={PAL.ink} fontSize="8" textAnchor="end">{r}</text>
          {cells.map((c, j) => {
            const v = matrix[i][j];
            return (<g key={j}>
              <rect x={lw + j*cw} y={th + i*ch} width={cw-1} height={ch-1} fill={divColor(v)}/>
              {v !== null && <text x={lw + j*cw + cw/2} y={th + i*ch + ch/2 + 3}
                fontSize="7" textAnchor="middle"
                fill={Math.abs(v) < 0.28 ? "#101010" : "#f5f5f5"}>{v>=0?"+":""}{v.toFixed(2)}</text>}
            </g>);
          })}
        </g>))}
      </svg>
    </div>
  );
}

export default function CalibrationHeatmap() {
  const params = ["D", "Dstar", "f"];
  return (
    <div style={{background: PAL.bg, padding: 20, color: PAL.ink,
      fontFamily: "JetBrains Mono, monospace"}}>
      <h2 style={{margin: "0 0 4px"}}>Calibration heatmap</h2>
      <p style={{color: PAL.mute, marginTop: 0, fontSize: 12}}>
        Coverage − nominal at 0.95. Red = under-covers, blue = over-covers, white = calibrated.</p>
      <div style={{display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start"}}>
        {params.map(p => DATA[p] ? <Heat key={p} param={p} {...DATA[p]}/> : null)}
      </div>
    </div>
  );
}
'''


def emit_jsx(df):
    rel = json.dumps(_reliability_data(df))
    heat = json.dumps(_heatmap_data(df))
    with open(_out("reliability_diagrams.jsx"), "w") as f:
        f.write(JSX_RELIABILITY.replace("__DATA__", rel))
    with open(_out("calibration_heatmap.jsx"), "w") as f:
        f.write(JSX_HEATMAP.replace("__DATA__", heat))
    print("wrote reliability_diagrams.jsx + calibration_heatmap.jsx", flush=True)


def main():
    df = load()
    print(f"loaded {CSV}: {len(df)} rows, generators={sorted(df.generator.unique())}",
          flush=True)
    fig_reliability(df)
    fig_heatmap(df)
    emit_jsx(df)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
