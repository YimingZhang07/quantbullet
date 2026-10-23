"""Anchored bilinear correction D-hat on balance x incentive, against the C2 warp and a log-linear tilt.

Main effects stay fixed at the fitted turnover model: the offset is m = pred_turnover.

- A (D-hat): mu = m * D(B, I). D is bilinear on a (balance x incentive) node grid and equals 1 on the reference
  row B* and the reference column I*. The other nodes are fitted by penalized weighted Poisson (Fisher scoring,
  ridge toward 1). SEs are sandwich SEs that ignore within-loan correlation.
- C2: mu = m * [F(k I) / F(k I*)] / [F(I) / F(I*)], k = (B / B*)^gamma, F = the model's fitted incentive curve.
  It has the same anchors as A, so the two surfaces compare node by node. Warped inputs outside the incentive clip
  take the edge value.
- Tilt: log D = theta * log(B / B*) * (I - I*), also 1 on both reference lines.

gamma and theta are profiled on the deviance with the offset fixed. Everything runs on all rows and on reporting
years 2022+. Balance is the nominal c_prev_balance, clipped to the outer balance knots. Reads the frame,
predictions and model under the config's output_root, and writes only to --out (outside the repository):
dhat_vs_c2.json and dhat_vs_c2.png.

Run from the repository root (one line):
    .venv/Scripts/python.exe explorations/2026-10-05-balance-incentive-2d/dhat_vs_c2.py
        --config projects/freddie_prepayment/turnover.example.toml --out <directory>
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projects.freddie_prepayment.config import REPOSITORY, read_config  # noqa: E402
from projects.freddie_prepayment.fit_turnover import CLIP  # noqa: E402

KEYS = ["row_id", "loan_identifier", "d_reporting_month"]
B_KNOTS = np.array([50e3, 150e3, 250e3, 450e3, 800e3])
I_KNOTS = np.array([CLIP["c_incentive"][0], -4., -3., -2., -1., CLIP["c_incentive"][1]])
B_STAR, I_STAR = 250e3, -3.
LAM = 10.     # absolute ridge on (d - 1)^2, in units of the weighted Poisson loss
GAMMA = .2
NB, NI = len(B_KNOTS), len(I_KNOTS)
N = NB * NI
FREE = np.array([B_KNOTS[j] != B_STAR and I_KNOTS[k] != I_STAR for j in range(NB) for k in range(NI)])

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, BLUE_LIGHT, ORANGE, AQUA = "#2a78d6", "#86b6ef", "#eb6834", "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b83232"])


# ---------------------------------------------------------------- bilinear grid
def corners(b, i):
    """The four (node index, interpolation weight) pairs of each row's cell."""
    jb = np.clip(np.searchsorted(B_KNOTS, b, side="right") - 1, 0, NB - 2)
    ki = np.clip(np.searchsorted(I_KNOTS, i, side="right") - 1, 0, NI - 2)
    u = (b - B_KNOTS[jb]) / (B_KNOTS[jb + 1] - B_KNOTS[jb])
    v = (i - I_KNOTS[ki]) / (I_KNOTS[ki + 1] - I_KNOTS[ki])
    base = jb * NI + ki
    return [(base, (1 - u) * (1 - v)), (base + NI, u * (1 - v)), (base + 1, (1 - u) * v), (base + NI + 1, u * v)]


def surface(cl, nodes):
    return sum(wt * nodes[idx] for idx, wt in cl)


def gram(cl, row_w):
    """Z' diag(row_w) Z over all nodes, accumulated without a design matrix."""
    H = np.zeros(N * N)
    for a, wa in cl:
        for b, wb in cl:
            H += np.bincount(a * N + b, weights=row_w * wa * wb, minlength=N * N)
    return H.reshape(N, N)


def proj(cl, row_v):
    return sum(np.bincount(a, weights=row_v * wa, minlength=N) for a, wa in cl)


def deviance(mu, yy, ww):
    """Weighted mean Poisson deviance for a 0/1 response."""
    return float(2 * np.sum(ww * (-yy * np.log(mu) - (yy - mu))) / np.sum(ww))


def fit_anchored(cl, yy, ww, mm):
    nodes = np.ones(N)
    ridge = 2 * LAM * np.eye(FREE.sum())

    def objective(nd):
        mu = mm * surface(cl, nd)
        return float(np.sum(ww * (mu - yy * np.log(mu))) + LAM * np.sum((nd[FREE] - 1) ** 2))

    loss = objective(nodes)
    for it in range(50):
        D = surface(cl, nodes)
        grad = proj(cl, ww * (mm * D - yy) / D)[FREE] + 2 * LAM * (nodes[FREE] - 1)
        H = gram(cl, ww * mm / D)[np.ix_(FREE, FREE)] + ridge
        step, t, accepted = np.linalg.solve(H, grad), 1., False
        while t > 1e-6:
            trial = nodes.copy()
            trial[FREE] -= t * step
            if (trial > 0).all():
                new = objective(trial)
                if new <= loss:
                    accepted = True
                    break
            t /= 2
        if not accepted:
            break
        gain, nodes, loss = loss - new, trial, new
        if gain < 1e-10 * abs(loss):
            break
    D = surface(cl, nodes)
    mu = mm * D
    bread = np.linalg.inv(gram(cl, ww * mm / D)[np.ix_(FREE, FREE)] + ridge)
    meat = gram(cl, (ww * (yy - mu) / D) ** 2)[np.ix_(FREE, FREE)]
    se = np.zeros(N)
    se[FREE] = np.sqrt(np.diag(bread @ meat @ bread))
    print(f"  A fit: {it + 1} Fisher steps")
    return dict(nodes=nodes.reshape(NB, NI), se=se.reshape(NB, NI), events=proj(cl, yy).reshape(NB, NI),
                deviance=deviance(mu, yy, ww), baseline=deviance(mm, yy, ww))


# ---------------------------------------------------------------- C2 and tilt
def incentive_curve(model_path: Path):
    """The fitted incentive factor F on a fine grid, read through ``model.term(...).curve``."""
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)["model"]
    grid = np.linspace(I_KNOTS[0], I_KNOTS[-1], 4501)
    values = np.asarray(model.term("c_incentive_fit").curve(grid), dtype=float)
    if not (values > 0).all():
        raise ValueError("the incentive curve must be positive")
    return grid, values


def make_forms(grid, values):
    def F(x):
        return np.interp(np.clip(x, grid[0], grid[-1]), grid, values)

    def c2(i, b, gamma):
        kappa = (b / B_STAR) ** gamma
        return F(kappa * i) / F(kappa * I_STAR) * F(I_STAR) / F(i)

    def tilt(i, b, theta):
        return np.exp(theta * np.log(b / B_STAR) * (i - I_STAR))

    return F, c2, tilt


# ---------------------------------------------------------------- figure
def draw(results, c2_nodes, c2, tilt, out: Path):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
                         "figure.facecolor": "white"})
    main = results["all"]
    fig = plt.figure(figsize=(12.5, 9.2), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1])
    heat_axes = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:])]
    limit = max(np.log(1.12), np.abs(np.log(main["nodes"])).max(), np.abs(np.log(c2_nodes)).max())
    norm = Normalize(-limit, limit)
    b_labels = [f"{int(b / 1e3)}K" for b in B_KNOTS]
    for ax, grid, title, se in (
            (heat_axes[0], main["nodes"], r"$\hat{D}$: anchored bilinear correction (fitted, main effects fixed)",
             main["se"]),
            (heat_axes[1], c2_nodes, rf"C2 implied correction, $\gamma$ = {GAMMA}", None)):
        image = ax.imshow(np.log(grid), cmap=DIVERGING, norm=norm, origin="lower", aspect="auto")
        for j in range(NB):
            for k in range(NI):
                anchor = not FREE[j * NI + k]
                text = "1\nanchor" if anchor else f"{grid[j, k]:.2f}" + (
                    f"\n±{1.96 * se[j, k]:.2f}" if se is not None else "")
                strong = abs(np.log(grid[j, k])) > .6 * limit
                ax.text(k, j, text, ha="center", va="center", fontsize=8 if not anchor else 7,
                        color="white" if strong else (MUTED if anchor else INK))
        ax.set_xticks(range(NI), [f"{x:g}" for x in I_KNOTS])
        ax.set_yticks(range(NB), b_labels)
        ax.set_xlabel("c_incentive (pp)")
        ax.set_ylabel("current balance")
        ax.set_title(title, loc="left")
        ax.tick_params(length=0)
    ticks = [t for t in (.8, .9, 1, 1.1, 1.25) if abs(np.log(t)) <= limit]
    bar = fig.colorbar(image, ax=heat_axes, ticks=np.log(ticks), shrink=.9, pad=.01)
    bar.ax.set_yticklabels([f"{t:g}" for t in ticks])
    bar.set_label("multiplier on current prediction", color=INK2)
    bar.outline.set_visible(False)

    line_axes = []
    xs = np.linspace(I_KNOTS[0], I_KNOTS[-1], 200)
    rows = [j for j in range(NB) if B_KNOTS[j] != B_STAR]
    for col, j in enumerate(rows):
        ax = fig.add_subplot(gs[1, col], sharey=line_axes[0] if line_axes else None)
        line_axes.append(ax)
        ax.axhline(1, color=AXIS, lw=.8, zorder=0)
        ax.axvline(I_STAR, color=GRID, lw=.8, zorder=0)
        ax.grid(True, color=GRID, lw=.5)
        ax.set_axisbelow(True)
        ax.plot(I_KNOTS, results["2022+"]["nodes"][j], color=BLUE_LIGHT, lw=1.2, marker="o", ms=4, mfc="white",
                label=r"$\hat{D}$, 2022+ only")
        ax.errorbar(I_KNOTS, main["nodes"][j], yerr=1.96 * main["se"][j], color=BLUE, lw=1.6, marker="o", ms=4,
                    capsize=2, elinewidth=1, label=r"$\hat{D}$, all data (±95%)")
        balance = np.full_like(xs, B_KNOTS[j])
        ax.plot(xs, c2(xs, balance, GAMMA), color=ORANGE, lw=1.6, ls="--", label=rf"C2, $\gamma$ = {GAMMA}")
        ax.plot(xs, tilt(xs, balance, main["theta_hat"]), color=AQUA, lw=1.6,
                label=rf"log-linear tilt, best $\theta$ = {main['theta_hat']:.3f}")
        ax.set_title(f"balance = {b_labels[j]}", loc="left")
        ax.set_xlabel("c_incentive (pp)")
        ax.set_xticks(I_KNOTS[:-1])
        if col == 0:
            ax.set_ylabel("correction multiplier D")
        else:
            ax.tick_params(labelleft=True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = line_axes[0].get_legend_handles_labels()
    wanted = [labels.index(name) for name in sorted(
        labels, key=lambda s: (not s.startswith(r"$\hat{D}$, all"), not s.startswith(r"$\hat{D}$"), s))]
    fig.legend([handles[i] for i in wanted], [labels[i] for i in wanted], loc="outside lower center", ncol=4,
               frameon=False)
    base = main["baseline"]
    fig.suptitle(
        "Balance × incentive: free 2D correction vs one-parameter forms  "
        f"(anchors B* = {int(B_STAR / 1e3)}K, I* = {I_STAR:g}; offset = current model)\n"
        f"In-sample deviance change, all data: A ({FREE.sum()} nodes) {main['deviance'] / base - 1:+.3%} · "
        f"C2 γ={GAMMA} {main['dev_c2_fixed'] / base - 1:+.3%} · "
        f"C2 best γ={main['gamma_hat']:.2f} {main['dev_c2_hat'] / base - 1:+.3%} · "
        f"tilt θ={main['theta_hat']:.3f} {main['dev_tilt'] / base - 1:+.3%}.  "
        "Balance = c_prev_balance (nominal), clipped 50K–800K.",
        x=.01, ha="left", fontsize=10, color=INK)
    fig.savefig(out / "dhat_vs_c2.png", dpi=170, bbox_inches="tight")


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="Directory for dhat_vs_c2.json and dhat_vs_c2.png")
    args = parser.parse_args()
    root = read_config(args.config).output_root
    out = Path(args.out).resolve()
    if out.is_relative_to(REPOSITORY):
        raise ValueError("--out must be outside the repository")
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    frame = (pl.scan_parquet(root / "turnover_frame.parquet")
             .select(*KEYS, "c_prev_balance", "c_incentive", "y_full_prepay", "weight")
             .join(pl.scan_parquet(root / "turnover_predictions.parquet"), on=KEYS)
             .collect())
    zero = frame.filter(pl.col("pred_turnover") <= 0)
    print(f"loaded {frame.height:,} rows in {time.perf_counter() - started:.1f}s; "
          f"dropping {zero.height} zero predictions with {int(zero['y_full_prepay'].sum())} events")
    frame = frame.filter(pl.col("pred_turnover") > 0)
    y = frame["y_full_prepay"].to_numpy().astype(float)
    w = frame["weight"].to_numpy().astype(float)
    m = frame["pred_turnover"].to_numpy().astype(float)
    B = np.clip(frame["c_prev_balance"].to_numpy().astype(float), B_KNOTS[0], B_KNOTS[-1])
    I = np.clip(frame["c_incentive"].to_numpy().astype(float), I_KNOTS[0], I_KNOTS[-1])
    recent = (frame["d_reporting_month"].dt.year() >= 2022).to_numpy()
    del frame, zero
    all_corners = corners(B, I)

    grid, values = incentive_curve(root / "turnover_model.pkl")
    F, c2, tilt = make_forms(grid, values)
    print("fitted incentive curve F relative to F(-3):",
          {float(x): round(float(F(x) / F(I_STAR)), 3) for x in (-5, -4, -3, -2, -1.5, -1, -.75, -.5)})

    results = {}
    for label, mask in (("all", np.ones(len(y), dtype=bool)), ("2022+", recent)):
        print(f"== {label}: {mask.sum():,} rows")
        yy, ww, mm, bb, ii = y[mask], w[mask], m[mask], B[mask], I[mask]
        fit = fit_anchored([(a[mask], wa[mask]) for a, wa in all_corners], yy, ww, mm)

        def c2_dev(gamma):
            return deviance(mm * c2(ii, bb, gamma), yy, ww)

        def tilt_dev(theta):
            return deviance(mm * tilt(ii, bb, theta), yy, ww)

        best_c2 = minimize_scalar(c2_dev, bounds=(0, 1.5), method="bounded", options={"xatol": 2e-3})
        best_tilt = minimize_scalar(tilt_dev, bounds=(-.5, .5), method="bounded", options={"xatol": 1e-3})
        fit.update(gamma_hat=float(best_c2.x), dev_c2_hat=float(best_c2.fun), dev_c2_fixed=c2_dev(GAMMA),
                   profile={g: c2_dev(g) for g in (0., .1, .2, .3, .5, 1.)},
                   theta_hat=float(best_tilt.x), dev_tilt=float(best_tilt.fun))
        results[label] = fit
        base = fit["baseline"]
        print(f"  deviance x1e3: baseline {base * 1e3:.5f} | A {fit['deviance'] * 1e3:.5f} "
              f"({fit['deviance'] / base - 1:+.3%}) | C2 g={GAMMA} ({fit['dev_c2_fixed'] / base - 1:+.3%}) | "
              f"C2 g_hat={fit['gamma_hat']:.3f} ({fit['dev_c2_hat'] / base - 1:+.3%}) | "
              f"tilt theta_hat={fit['theta_hat']:.3f} ({fit['dev_tilt'] / base - 1:+.3%})")
        # Gain on the 2*loss scale: the fit weights have mean 1, so sum(w) is about the row count.
        scale = ww.sum()
        print(f"  2*loss gain: A {(base - fit['deviance']) * scale:.1f} ({FREE.sum()} params) | "
              f"C2 best {(base - fit['dev_c2_hat']) * scale:.1f} (1) | tilt {(base - fit['dev_tilt']) * scale:.1f} (1)")
        print("  C2 profile (relative change):", {g: f"{d / base - 1:+.4%}" for g, d in fit["profile"].items()})
        index = (B_KNOTS / 1e3).astype(int)
        for name, digits in (("nodes", 3), ("se", 3), ("events", 0)):
            print(f"  {name}:")
            print(pd.DataFrame(fit[name], index=index, columns=I_KNOTS).round(digits).to_string())

    c2_nodes = np.array([[c2(np.array([ik]), np.array([bk]), GAMMA)[0] for ik in I_KNOTS] for bk in B_KNOTS])
    print(f"C2 (gamma={GAMMA}) at nodes:")
    print(pd.DataFrame(c2_nodes, index=(B_KNOTS / 1e3).astype(int), columns=I_KNOTS).round(3).to_string())
    (out / "dhat_vs_c2.json").write_text(json.dumps(
        {label: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
         for label, r in results.items()}
        | {"c2_nodes": c2_nodes.tolist(), "b_knots": B_KNOTS.tolist(), "i_knots": I_KNOTS.tolist()}, indent=1))
    draw(results, c2_nodes, c2, tilt, out)
    print(f"saved to {out}; total {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
