"""Extra figures connecting the four joint analyses and their controls.

This is presentation-only: every curve comes from saved results; no inference
or smoothing is added. Preview first; final vector export requires --final.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, default=ROOT/"results/science")
    p.add_argument("--output-dir", type=Path, default=ROOT/"results/joint_figures")
    p.add_argument("--skill-dir", type=Path, required=True)
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(args.skill_dir/"scripts"))
    from setup_style import setup_style
    from visual_qa import audit_layout
    from profile_data import profile_data
    setup_style(journal="general", lang="en", use_sciplots=False)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    styles = ["-", "--", "-.", ":", "--"]
    markers = ["o", "s", "^", "D", "v"]
    overview = json.loads((args.analysis_dir/"overview.json").read_text())
    qa, profiles, density_ranges = {}, {}, []

    def zlabel(z):
        return "0" if abs(float(z)) < 1e-6 else f"{float(z):.2g}"

    def curve_axes(ax, title="", zero=True):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([.05, .1, .3, 1, 3, 5]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Separation (cMpc/h, log scale)")
        ax.set_title(title)
        if zero:
            ax.axhline(0, color=".65", ls=":", lw=.6)

    def save(fig, name, caption):
        fig.supxlabel(caption, fontsize=7)
        fig.canvas.draw()
        issues = audit_layout(fig)
        qa[name] = issues
        if any(item[0] == "FAIL" for item in issues):
            raise RuntimeError(f"Figure QA failure {name}: {issues}")
        fig.savefig(args.output_dir/(name+"_preview.png"), dpi=150)
        if args.final:
            for ext in ["png", "pdf", "svg"]:
                fig.savefig(args.output_dir/(name+"."+ext), dpi=300)
            from PIL import Image
            Image.open(args.output_dir/(name+".png")).convert("L").save(
                args.output_dir/(name+"_grayscale.png"), dpi=(300, 300))
        plt.close(fig)

    # Data profiling is done before deciding how to display the distributions.
    for s in overview:
        name = f"snapshot_{s['snapshot']:03d}"
        points = arrays(args.analysis_dir/name/"points.npz")
        frame = pd.DataFrame({"log10_density": np.log10(points["rho"]),
            "log10_mass": np.log10(points["stellar_mass_msun"]),
            "membership_strength": points["membership_strength"],
            "outlier_score": points["outlier_score"],
            "density_group": pd.Categorical(points["density_group"]),
            "is_noise": points["is_noise"]})
        profiles[name] = profile_data(frame, group_cols=["density_group"])
        density_ranges.extend([float(frame.log10_density.min()), float(frame.log10_density.max())])
    (args.output_dir/"feature_profiles.json").write_text(json.dumps(profiles, ensure_ascii=False, indent=2, default=str))

    for suffix, label in [("", "Common mass threshold"), ("_mass_matched", "Matched coarse stellar-mass histograms")]:
        fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
        for j, s in enumerate(overview):
            g = arrays(args.analysis_dir/(f"snapshot_{s['snapshot']:03d}"+suffix)/"global.npz")
            r = np.sqrt(g["edges"][:-1]*g["edges"][1:])
            ax.errorbar(r, g["xi"], yerr=g["se"], color=colors[j], ls=styles[j], marker=markers[j],
                        ms=3, lw=.8, label=f"z={zlabel(s['redshift'])}")
        curve_axes(ax, label, zero=False)
        ax.set_yscale("log")
        ax.set_ylabel("Landy-Szalay correlation (log scale)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        save(fig, "cross_redshift"+suffix, "+/-1 internal spatial JK SE; 27 regions. Same simulation, not four independent volumes.")

    for s in overview:
        name = f"snapshot_{s['snapshot']:03d}"
        d = args.analysis_dir/name
        g = arrays(d/"global.npz")
        r = np.sqrt(g["edges"][:-1]*g["edges"][1:])
        title = f"TNG100-1, z={zlabel(s['redshift'])}, N={s['n']:,}"
        fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
        for j, key in enumerate(["natural", "davis_peebles", "hewett", "hamilton", "landy_szalay"]):
            ax.plot(r, g["xi_"+key], color=colors[j], ls=styles[j], marker=markers[j],
                    ms=3, lw=.8, label=key.replace("_", " "))
        curve_axes(ax, title, zero=False)
        ax.set_yscale("log")
        ax.set_ylabel("Correlation estimate (log scale)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        save(fig, name+"_estimators", "Five estimators share the same normalized DD/DR/RR; they are not independent models.")

        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True, layout="constrained")
        axes[0].errorbar(r, g["xi"], yerr=g["se"], fmt="o", ms=3, color=colors[0], label="Landy-Szalay +/-1 JK SE")
        if s["fit"].get("success"):
            fit = s["fit"]
            axes[0].plot(r, fit["predicted"], color=colors[1], ls="--", label="Shell-averaged power law")
            idx = np.asarray(fit["fit_indices"], dtype=int)
            axes[0].scatter(r[idx], g["xi"][idx], marker="s", s=25, facecolors="none", edgecolors=colors[1], label="Fitted bins")
            axes[1].plot(r, fit["residual"], "o-", color=colors[0], ms=3)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Correlation (log scale)")
        axes[1].set_ylabel("Data minus model")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        for i, ax in enumerate(axes):
            curve_axes(ax, zero=i != 0)
        axes[0].set_title(title)
        axes[0].set_xlabel("")
        save(fig, name+"_fit_diagnostic", "Fit range declared before running; convergence alone is not evidence that one power law is adequate.")

        fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
        for j, subset in enumerate(["cluster", "noise"]):
            sub = arrays(d/f"subset_{subset}.npz")
            se = np.sqrt(np.diag(sub["covariance"])) if "covariance" in sub else None
            ax.errorbar(r, sub["xi"], yerr=se, color=colors[j], ls=styles[j], marker=markers[j], ms=3,
                        label=subset+"-labelled")
        curve_axes(ax, title, zero=False)
        ax.set_yscale("log")
        ax.set_ylabel("Conditional correlation (log scale)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        save(fig, name+"_cluster_noise", "+/-1 internal spatial JK SE; labels fixed. Coordinate-derived selection, not causal evidence.")

        fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
        for j, percent in enumerate([5, 10, 20]):
            a = arrays(d/f"removal_{percent}.npz")
            original = a["original_xi"]
            valid = np.isfinite(original) & (original > 1)
            change = np.where(valid, (a["targeted_xi"]-original)/original*100, np.nan)
            random = (a["random_xi"]-original)/original*100
            mean, sd = np.nanmean(random, axis=0), np.nanstd(random, axis=0, ddof=1)
            ax.plot(r, change, color=colors[j], ls=styles[j], marker=markers[j], ms=3,
                    label=f"Remove top {percent}% density")
            ax.plot(r, np.where(valid, mean, np.nan), color=colors[j], ls=":", lw=.6)
            ax.fill_between(r, np.where(valid, mean-sd, np.nan), np.where(valid, mean+sd, np.nan), color=colors[j], alpha=.12)
        curve_axes(ax, title)
        ax.set_ylabel("Change from full correlation (%)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        save(fig, name+"_removal", "Dotted lines/bands: random-deletion mean +/-1 SD, 10 repeats; show only full xi>1. Not cosmic uncertainty.")

    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    for j, s in enumerate(overview):
        name = f"snapshot_{s['snapshot']:03d}"
        pnt = arrays(args.analysis_dir/name/"points.npz")
        ax.hist(np.log10(pnt["rho"]), bins=np.linspace(min(density_ranges), max(density_ranges), 65), density=True,
                     histtype="step", color=colors[j], ls=styles[j], label=f"z={zlabel(s['redshift'])}")
    ax.set(xlabel="log10 kNN density [(cMpc/h)^-3]", ylabel="Probability density", title="Local density distributions")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    save(fig, "density_distributions", "Histogram includes all selected galaxies in each snapshot; common bins cover the full observed range.")
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    for j, s in enumerate(overview):
        name = f"snapshot_{s['snapshot']:03d}"
        c = arrays(args.analysis_dir/(name+"_poisson_control")/"global.npz")
        r = np.sqrt(c["edges"][:-1]*c["edges"][1:])
        ax.errorbar(r, c["xi"], yerr=c["se"], color=colors[j], ls=styles[j], marker=markers[j], ms=3, label=f"z={zlabel(s['redshift'])}")
    curve_axes(ax, "Uniform matched-N controls")
    ax.set_ylabel("Control correlation")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    save(fig, "uniform_controls", "+/-1 internal spatial JK SE, 27 regions. Uniform controls are synthetic, not TNG data.")
    ensemble_dir = args.analysis_dir.parent/"uniform_null_ensemble"
    if (ensemble_dir/"summary.json").exists():
        ensemble = json.loads((ensemble_dir/"summary.json").read_text())
        fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
        jitter = np.random.default_rng(20261021)
        for j, item in enumerate(ensemble):
            a = arrays(ensemble_dir/f"snapshot_{item['snapshot']:03d}.npz")
            ax.scatter(j+jitter.uniform(-.12, .12, len(a["small_scale_means"])), a["small_scale_means"],
                       color=colors[0], alpha=.65, s=12, label="20 added uniform realisations" if j == 0 else None)
            ax.errorbar(j+.2, item["mean_small_scale_xi"], yerr=item["between_realisation_sd"], fmt="ks",
                        ms=4, lw=.8, capsize=3, label="Added mean +/-1 SD" if j == 0 else None)
            ax.scatter(j, item["original_single_control_mean"], marker="D", facecolors="none", edgecolors=colors[1],
                       s=50, linewidths=1, label="Original full-pipeline control" if j == 0 else None)
        ax.axhline(0, color=".5", ls=":", lw=.8)
        ax.set_xticks(range(4), [f"z={zlabel(s['redshift'])}\nN={s['n']:,}" for s in overview])
        ax.set(xlabel="Matched real-catalogue size (separate null ensembles)",
               ylabel="Small-scale equal-bin mean correlation", title="Uniform-null ensemble and retained original controls")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        save(fig, "uniform_null_ensemble", "Dots: 20 independent D/R realisations per N; bars: between-realisation SD, not cosmic uncertainty.\nOriginal controls are retained; the negative z=2 realisation is not replaced.")
    (args.output_dir/"layout_qa.json").write_text(json.dumps(qa, indent=2))
    print(f"Saved {len(qa)} figure previews to {args.output_dir}")


if __name__ == "__main__":
    main()
