"""Reproducible data plots with separate preview and final export steps.

Optional --skill-dir reuses the provided SciPilot style/profile/layout helpers.
Without it, standard matplotlib output remains portable. No TNG curves are
invented: plot only arrays actually saved by run_analysis/benchmark.
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


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
STYLES = ["-", "--", "-.", ":", "--"]
MARKERS = ["o", "s", "^", "D", "v"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skill-dir", type=Path)
    parser.add_argument("--export-final", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.skill_dir:
        sys.path.insert(0,str(args.skill_dir/"scripts"))
        from setup_style import setup_style
        from visual_qa import audit_layout
        from profile_data import profile_data
        setup_style(journal="general",lang="en",use_sciplots=False)
    else:
        audit_layout = lambda fig: []
        profile_data = None
    plt.rcParams.update({"font.size":8,"axes.labelsize":9,"axes.titlesize":9,
                         "pdf.fonttype":42,"svg.fonttype":"none","figure.constrained_layout.use":True})
    qa = {}

    def save(fig, name):
        # Matplotlib's default log minor labels overlap for the short 0.1--5
        # correlation range. Keep explicitly sparse readable major labels.
        if not name.startswith("benchmark"):
            for ax in fig.axes:
                if ax.get_xscale()=="log":
                    ax.xaxis.set_major_locator(FixedLocator([.05,.1,.3,1,3,5]))
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos: f"{v:g}"))
                    ax.xaxis.set_minor_formatter(NullFormatter())
        fig.canvas.draw()
        issues = audit_layout(fig)
        qa[name] = issues
        if any(severity=="FAIL" for severity,_ in issues):
            raise RuntimeError(f"Figure glyph/layout failure: {name}")
        fig.savefig(args.output_dir/(name+"_preview.png"),dpi=150)
        if args.export_final:
            for fmt in ["png","pdf","svg"]:
                fig.savefig(args.output_dir/(name+"."+fmt),dpi=300)
            from PIL import Image
            Image.open(args.output_dir/(name+".png")).convert("L").save(args.output_dir/(name+"_grayscale.png"),dpi=(300,300))
        plt.close(fig)

    frame = pd.read_csv(args.benchmark_dir/"benchmark_raw.csv")
    ok = frame[frame.status=="ok"]
    if profile_data:
        profile = profile_data(ok,group_cols=["method"])
        (args.output_dir/"benchmark_eda.json").write_text(json.dumps(profile,indent=2,ensure_ascii=False,default=str))
    for value,label,name,scale in [("elapsed_total_s","DD/DR/RR total time (s, log scale)","benchmark_time",1),
                                  ("peak_rss_bytes","Whole-worker peak RSS (MiB)","benchmark_memory",1/1048576)]:
        fig,ax = plt.subplots(figsize=(7.2,3.8))
        for j,method in enumerate(["ckdtree","blocked_brute","dense_brute"]):
            part = ok[ok.method==method]
            xs,ys = [],[]
            for n,group in part.groupby("n"):
                xs.append(n)
                ys.append(group[value].median()*scale)
                ax.scatter(np.full(len(group),n),group[value]*scale,s=10,color=COLORS[j],alpha=.65,marker=MARKERS[j])
            ax.plot(xs,ys,color=COLORS[j],marker=MARKERS[j],linestyle=STYLES[j],label=method.replace("_"," "),lw=1)
        ax.set_xscale("log")
        if value=="elapsed_total_s": ax.set_yscale("log")
        else: ax.set_ylim(bottom=0)
        ax.set_xlabel("N data points and N random points (log scale)")
        ax.set_ylabel(label)
        origin = "Synthetic uniform" if ok.origin.iloc[0]=="SYNTHETIC_UNIFORM" else "Mass-selected TNG catalogue"
        source = str(ok.catalog.iloc[0]) if "catalog" in ok.columns else ""
        if source and Path(source).is_file():
            with np.load(source, allow_pickle=False) as cat:
                if "engineering_only" in cat and bool(cat["engineering_only"]):
                    origin = "Engineering-only real TNG objects"
        ax.set_title(f"{origin} periodic benchmark; {ok['repeat'].nunique()} repeats")
        ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),frameon=False)
        fig.supxlabel("Dots: individual repeats; line: median. Large brute-force sizes intentionally skipped.",fontsize=7)
        save(fig,name)
    if args.benchmark_only:
        (args.output_dir/"layout_qa.json").write_text(json.dumps(qa,indent=2))
        return
    overview = json.loads((args.analysis_dir/"overview.json").read_text())
    spatial_norm_values = []
    for case in overview:
        folder = "synthetic_clustered" if case["data_kind"]=="SYNTHETIC_CLUSTERED" else "synthetic_uniform" if case["data_kind"]=="SYNTHETIC_UNIFORM" else f"snapshot_{case['snapshot']:03d}"
        with np.load(args.analysis_dir/folder/"points.npz") as points:
            finite = np.log10(points["rho"])
            finite = finite[np.isfinite(finite)]
            spatial_norm_values.extend([finite.min(),finite.max()])
    common_spatial_norm = matplotlib.colors.Normalize(min(spatial_norm_values),max(spatial_norm_values))
    for case in overview:
        if case["data_kind"]=="SYNTHETIC_CLUSTERED":
            name = "synthetic_clustered"
        elif case["data_kind"]=="SYNTHETIC_UNIFORM":
            name = "synthetic_uniform"
        else:
            name = f"snapshot_{case['snapshot']:03d}"
        directory = args.analysis_dir/name
        with np.load(directory/"global.npz") as g:
            edges,xi,se = g["edges"],g["xi"],g["se"]
        r = np.sqrt(edges[:-1]*edges[1:])
        title = name.replace("_"," ")+(" (not TNG data)" if case["data_kind"].startswith("SYNTHETIC") else " (TNG100-1)")
        fig,ax = plt.subplots(figsize=(7.2,4.0))
        ax.errorbar(r,xi,yerr=se,fmt="o-",lw=1,ms=3,color=COLORS[0],label="Full sample; +/-1 spatial JK SE")
        zero_line = ax.axhline(0,color="0.5",lw=.6,ls=":")
        ax.set_xscale("log")
        positive = np.all((xi-se)[np.isfinite(xi-se)] > 0)
        if positive:
            zero_line.remove()
        ax.set_yscale("log" if positive else "symlog", **({} if positive else {"linthresh":1}))
        if positive:
            ax.set_ylim(np.nanmin(xi-se)/1.3, np.nanmax(xi+se)*1.3)
        ax.set_xlabel("Pair separation (cMpc/h, log scale)")
        ax.set_ylabel("Landy-Szalay correlation (log scale)" if positive else "Landy-Szalay correlation (symlog)")
        ax.set_title(title)
        ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),frameon=False)
        fig.supxlabel("27 delete-one regions; internal spatial uncertainty. Missing RR bins are not plotted.",fontsize=7)
        save(fig,name+"_global")
        fig,ax = plt.subplots(figsize=(7.2,4.0))
        for j,group in enumerate(["high","medium","low"]):
            with np.load(directory/f"subset_density_{group}.npz") as sub:
                ax.plot(r,sub["xi"],color=COLORS[j],ls=STYLES[j],marker=MARKERS[j],ms=3,lw=1,label=group+" density")
        ax.axhline(0,color="0.5",lw=.6,ls=":")
        ax.set(xscale="log",yscale="symlog",xlabel="Pair separation (cMpc/h, log scale)",ylabel="Conditional correlation (symlog)",title=title)
        ax.set_yscale("symlog",linthresh=1)
        ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),frameon=False)
        fig.supxlabel("Coordinate-derived 20/60/20 density selection; differences are not independent physical proof.",fontsize=7)
        save(fig,name+"_density_subsets")
        parts = json.loads((directory/"dd_decomposition.json").read_text())
        fig,ax = plt.subplots(figsize=(7.2,3.8))
        for j,key in enumerate(["within","between","cluster_noise","noise_noise"]):
            values = np.array([np.nan if v is None else v for v in parts["shares"][key]])
            ax.plot(r,values,color=COLORS[j],ls=STYLES[j],marker=MARKERS[j],ms=3,lw=1,label=key.replace("_","-"))
        ax.set_xscale("log")
        ax.set_ylim(0,1.03)
        ax.set(xlabel="Pair separation (cMpc/h, log scale)",ylabel="Share of raw DD pairs",title=title)
        ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),frameon=False)
        fig.supxlabel("Fractions partition raw DD, not the excess-correlation signal; undefined bins omitted.",fontsize=7)
        save(fig,name+"_dd_shares")
        with np.load(directory/"points.npz") as points:
            count = len(points["coords"])
            index = np.random.default_rng(10).choice(count,min(count,2000),replace=False)
            coords = points["coords"][index]
            log_rho = np.log10(points["rho"][index])
            noise = points["is_noise"][index]
        fig = plt.figure(figsize=(7.2,4.8))
        ax = fig.add_subplot(111,projection="3d")
        norm = common_spatial_norm
        for keep,marker,label in [(~noise,"o","Cluster-labelled"),(noise,"x","Noise-labelled")]:
            ax.scatter(*coords[keep].T,c=log_rho[keep],norm=norm,cmap="viridis",marker=marker,s=5,alpha=.6,label=label)
        mappable = matplotlib.cm.ScalarMappable(norm=norm,cmap="viridis")
        fig.colorbar(mappable,ax=ax,shrink=.6,pad=.12,label="log10 kNN number density [(cMpc/h)^-3]")
        ax.set(xlabel="x (cMpc/h)",ylabel="y (cMpc/h)",zlabel="z (cMpc/h)",title=title)
        ax.set_box_aspect((1,1,1))
        box = case["boxsize_cMpc_h"]
        ax.set(xlim=(0,box),ylim=(0,box),zlim=(0,box))
        ax.set_xticks([0,box/3,2*box/3,box])
        ax.set_yticks([0,box/3,2*box/3,box])
        ax.set_zticks([0,box/3,2*box/3,box])
        ax.legend(loc="upper left",bbox_to_anchor=(-.1,1),frameon=False,markerscale=2)
        fig.supxlabel(f"Displayed {len(index)}/{count} points; density color + label marker. Periodic face neighbours can look separated.",fontsize=7)
        save(fig,name+"_spatial")
    (args.output_dir/"layout_qa.json").write_text(json.dumps(qa,indent=2))


if __name__=="__main__": main()
