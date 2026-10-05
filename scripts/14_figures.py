"""Paper figures (aggregates only; no competition text is drawn).

Figure 1 — forest plot: our graph minus the released graph (or its emulated schema), per split,
           for BM25 Recall@10, BM25+PPR fusion with `calls`-only propagation Recall@10 (the
           pre-registered H1 contrast) and Gemma 4 Hit@5. Development and held-out splits are
           grouped; a filled marker means the 95% CI excludes zero.
Figure 2 — (a) Gemma's Hit@5 gain against the gain in candidate coverage, per split, with the
           identity line; (b) Gemma's Hit@5 when a gold location is among its candidates, per
           split and graph.
All plotted numbers are written to results/figures/figure_data.json (paired bootstrap 95% CIs,
5,000 resamples, seed 0).

Usage:  python scripts/14_figures.py              # recompute from per-task records, then draw
        python scripts/14_figures.py --from-data  # redraw from results/figures/figure_data.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib import patheffects

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

OUT = Path("results/figures")

# Design tokens: one ink, one accent for the pre-registered held-out tests, quiet greys.
INK, INK2, MUTED = "#17212b", "#4a5562", "#8a939d"
ACCENT = "#9e2a3a"          # held-out (pre-registered) rows and points (bordeaux)
RELEASED = "#a3abb4"        # released graph / schema
GRID, ZERO, BAND = "#e3e6ea", "#5d6772", "#f3f5f7"
FONT = ["FreeSans", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

SPLITS = [  # key, label, group, task dir, suffix, official-method name
    ("comp", "Competition tasks", "dev", "results/loc_notest/tasks", "", "official"),
    ("public", "Public history", "dev", "results/public_loc/tasks", "|notest", "official_schema"),
    ("swebl", "SWE-bench Lite", "held", "results/heldout/loc/tasks", "|notest", "official_schema"),
    ("pymatgen", "pymatgen", "held", "results/heldout/loc/tasks", "|notest", "official_schema"),
]
LABEL = {k: lab for k, lab, *_ in SPLITS}
GROUP = {k: g for k, _, g, *_ in SPLITS}


# ----------------------------------------------------------------------------- data
def boot_ci(x, seed=0, b=5000):
    x = np.asarray(x, float)
    bs = x[np.random.default_rng(seed).integers(0, len(x), (b, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def retrieval_contrasts():
    from cgl.bench import metrics_from_ranks

    out = {}
    for split, label, _, d, suf, off in SPLITS:
        recs = [json.load(open(p)) for p in sorted(Path(d).glob("*.json"))]
        if split in ("swebl", "pymatgen"):
            recs = [r for r in recs if r.get("split") == split]
        if not recs:
            continue
        if split == "comp":
            pairs = {"bm25": ("bm25_cgl", "bm25_official"), "fusion_calls": ("rrf_cgl-calls_only", "rrf_official")}
        else:
            pairs = {"bm25": (f"bm25_cgl{suf}", f"bm25_{off}{suf}"),
                     "fusion_calls": (f"rrf_cgl-calls_only{suf}", f"rrf_cgl-{off}{suf}")}
        out[split] = {"label": label, "n": len(recs)}
        for name, (a, b) in pairs.items():
            d_ = [metrics_from_ranks(r["methods"][a]["ranks"])["recall@10"]
                  - metrics_from_ranks(r["methods"][b]["ranks"])["recall@10"] for r in recs]
            out[split][name] = boot_ci(d_)
    return out


def llm_numbers():
    out = {}
    for f in ("results/llm/summary.json", "results/heldout/llm/summary.json"):
        if not Path(f).exists():
            continue
        s = json.load(open(f))
        for split, S in s["splits"].items():
            if split == "all":
                continue
            D = S["paired_differences"]
            out[split] = {"n": S["n_tasks_paired"],
                          "gemma_hit5_diff": D["cgl - official [llm_hit@5]"],
                          "cand_hit_diff": D["cgl - official [cand_hit]"],
                          "selection_given_gold": S["llm_hit@5_given_gold_in_candidates"]}
    return out


# ----------------------------------------------------------------------------- style
def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 0, "xtick.labelsize": 7.5, "ytick.labelsize": 8.5,
        "svg.fonttype": "none", "pdf.fonttype": 42, "figure.dpi": 100,
    })


def save(fig, name):
    for ext, kw in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white", bbox_inches="tight", pad_inches=0.04, **kw)
    plt.close(fig)


def pts(v):
    return [100 * x for x in v]


# ----------------------------------------------------------------------------- figure 1
def figure1(ret, llm):
    rows = [k for k, *_ in SPLITS if k in ret]
    # vertical layout: group headers + rows, top to bottom
    layout, y = [], 0.0
    for g, title in (("dev", "Development repositories"), ("held", "Held-out · pre-registered")):
        ks = [k for k in rows if GROUP[k] == g]
        if not ks:
            continue
        layout.append(("header", title, y, g))
        y += 0.85
        for k in ks:
            layout.append(("row", k, y, g))
            y += 1.0
        y += 0.35
    ymax = y - 0.35 - 0.45
    panels = [("bm25", "BM25", "Recall@10"), ("fusion_calls", "BM25 + PPR", "Recall@10, calls only"),
              ("gemma", "Gemma 4 re-ranking", "Hit@5")]
    fig = plt.figure(figsize=(7.2, 0.36 * ymax + 1.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.08)
    lab_ax = fig.add_subplot(gs[0])
    axes = [fig.add_subplot(gs[i + 1], sharey=lab_ax) for i in range(3)]
    xlim = (-7, 12)
    for ax in [lab_ax] + axes:
        ax.set_ylim(ymax + 0.15, -0.55)
    lab_ax.axis("off")
    for kind, what, yy, g in layout:
        if kind == "header":
            lab_ax.text(0.0, yy, what.upper(), fontsize=6.8, fontweight="bold", color=ACCENT if g == "held" else INK2,
                        va="center", ha="left", transform=lab_ax.get_yaxis_transform())
        else:
            n = ret[what]["n"]
            lab_ax.text(0.04, yy, LABEL[what], fontsize=8.5, color=INK, va="center", ha="left",
                        transform=lab_ax.get_yaxis_transform())
            lab_ax.text(0.98, yy, f"n = {n}", fontsize=7.2, color=MUTED, va="center", ha="right",
                        transform=lab_ax.get_yaxis_transform())
    held = [yy for kind, _, yy, g in layout if g == "held"]
    if held:
        tr = blended_transform_factory(fig.transFigure, lab_ax.transData)
        fig.add_artist(Rectangle((0.0, min(held) - 0.42), 1.0, max(held) - min(held) + 0.42 + 0.5,
                                 transform=tr, facecolor=BAND, edgecolor="none", zorder=-1))
    for ax in [lab_ax] + axes:
        ax.patch.set_alpha(0)
    for ax, (key, title, metric) in zip(axes, panels):
        ax.set_xlim(*xlim)
        for xv in (-5, 5, 10):
            ax.axvline(xv, color=GRID, lw=0.6, zorder=1)
        ax.axvline(0, color=ZERO, lw=0.8, zorder=2)
        for kind, k, yy, g in layout:
            if kind != "row":
                continue
            v = llm.get(k, {}).get("gemma_hit5_diff") if key == "gemma" else ret[k].get(key)
            if not v:
                continue
            m, lo, hi = pts(v)
            col = ACCENT if g == "held" else INK
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.3, solid_capstyle="butt", zorder=3)
            for e in (lo, hi):
                ax.plot([e, e], [yy - 0.13, yy + 0.13], color=col, lw=1.0, zorder=3)
            ax.plot(m, yy, "o", ms=5.6, mfc=col if sig else "white", mec=col, mew=1.3, zorder=4)
            halo = [patheffects.withStroke(linewidth=2.4, foreground=BAND if g == "held" else "white")]
            ax.text(m, yy - 0.3, f"{m:+.1f}".replace("-", "\u2212"), fontsize=6.8, color=col, ha="center",
                    va="bottom", zorder=5, path_effects=halo)
        ax.set_title(title, fontsize=8.6, color=INK, loc="left", pad=14, fontweight="bold")
        ax.text(0, 1.0, metric, transform=ax.transAxes, fontsize=7.2, color=MUTED, va="bottom",
                ha="left")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xticks([-5, 0, 5, 10])
    handles = [Line2D([], [], marker="o", ls="", ms=5.6, mfc=INK, mec=INK, label="95% CI excludes 0"),
               Line2D([], [], marker="o", ls="", ms=5.6, mfc="white", mec=INK, mew=1.3, label="95% CI includes 0")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, -0.03), ncol=2, frameon=False,
               fontsize=7.2, labelcolor=INK2, handletextpad=0.3, columnspacing=1.2)
    fig.text(0.985, -0.005, "Difference in points: codegraph-loc minus the released graph (or its schema)", ha="right",
             va="bottom", fontsize=7.2, color=MUTED)
    save(fig, "fig1_coverage_forest")


# ----------------------------------------------------------------------------- figure 2
def figure2(llm):
    ks = [k for k, *_ in SPLITS if k in llm]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.42})
    # (a) gain tracks coverage
    xl, yl = (-5.5, 13), (-6.5, 11.5)
    a.plot(xl, xl, color="#c9ced4", lw=0.9, ls=(0, (4, 3)), zorder=1)
    a.text(10.0, 8.6, "y = x", rotation=45, transform_rotates_text=True, rotation_mode="anchor",
           fontsize=6.8, color=MUTED, ha="left", va="top")
    a.axhline(0, color=GRID, lw=0.6, zorder=0)
    a.axvline(0, color=GRID, lw=0.6, zorder=0)
    # label placement, chosen so that no label covers a data line:
    # "above"/"below" = centred past the end of the vertical CI; otherwise an offset in points
    place = {"public": "above", "swebl": "below", "comp": (7, -9, "left"), "pymatgen": (-6, 11, "right")}
    for k in ks:
        x, xlo, xhi = pts(llm[k]["cand_hit_diff"])
        y, ylo, yhi = pts(llm[k]["gemma_hit5_diff"])
        col = ACCENT if GROUP[k] == "held" else INK
        a.plot([xlo, xhi], [y, y], color=col, lw=0.9, alpha=0.32, solid_capstyle="butt", zorder=2)
        a.plot([x, x], [ylo, yhi], color=col, lw=0.9, alpha=0.32, solid_capstyle="butt", zorder=2)
        a.plot(x, y, "o", ms=6.2, mfc=col, mec="white", mew=1.2, zorder=4)
        how = place.get(k, (7, 6, "left"))
        kw = dict(fontsize=7.6, color=col, zorder=5)
        if how == "above":
            a.annotate(LABEL[k], (x, yhi), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", **kw)
        elif how == "below":
            a.annotate(LABEL[k], (x, ylo), xytext=(4, -3), textcoords="offset points", ha="center", va="top", **kw)
        else:
            dx, dy, ha = how
            a.annotate(LABEL[k], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", **kw)
    a.set_xlim(*xl)
    a.set_ylim(*yl)
    a.set_xticks([-4, 0, 4, 8, 12])
    a.set_yticks([-4, 0, 4, 8])
    a.set_xlabel("Change in gold among candidates (points)", fontsize=7.6)
    a.set_ylabel("Change in Gemma Hit@5 (points)", fontsize=7.6)
    a.set_title("a   Gemma's gain tracks candidate coverage", fontsize=8.6, color=INK, loc="left",
                fontweight="bold", pad=8)
    # (b) selection accuracy given coverage
    order = list(reversed(ks))
    for i, k in enumerate(order):
        sel = llm[k]["selection_given_gold"]
        col = ACCENT if GROUP[k] == "held" else INK
        for j, (cond, face, edge) in enumerate((("official", "white", RELEASED), ("cgl", col, col))):
            m, lo, hi = pts(sel[cond])
            yy = i + (0.13 if j == 0 else -0.13)
            b.plot([lo, hi], [yy, yy], color=edge, lw=1.1, zorder=2)
            b.plot(m, yy, "o", ms=5.6, mfc=face, mec=edge, mew=1.3, zorder=3)
    b.set_yticks(range(len(order)))
    b.set_yticklabels([LABEL[k] for k in order], fontsize=8)
    for t, k in zip(b.get_yticklabels(), order):
        t.set_color(ACCENT if GROUP[k] == "held" else INK)
    b.set_xlim(80, 100)
    b.set_xticks([80, 85, 90, 95, 100])
    b.set_xticklabels(["80%", "85%", "90%", "95%", "100%"])
    for xv in (85, 90, 95):
        b.axvline(xv, color=GRID, lw=0.6, zorder=0)
    b.set_ylim(-0.6, len(order) - 0.4 + 0.55)
    b.set_xlabel("Gemma Hit@5 when a gold location is among its candidates", fontsize=7.6)
    b.set_title("b   Selection accuracy is the same with both graphs", fontsize=8.6, color=INK, loc="left",
                fontweight="bold", pad=8)
    handles = [Line2D([], [], marker="o", ls="", ms=5.6, mfc="white", mec=RELEASED, mew=1.3,
                      label="Released graph (or schema)"),
               Line2D([], [], marker="o", ls="", ms=5.6, mfc=INK, mec=INK, label="codegraph-loc")]
    b.legend(handles=handles, loc="upper left", ncol=2, fontsize=7, frameon=False, labelcolor=INK2,
             handletextpad=0.3, columnspacing=1.4, borderaxespad=0.1, borderpad=0)
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    b.spines["left"].set_visible(False)
    b.tick_params(axis="y", left=False)
    a.tick_params(axis="y", labelsize=7.5, size=3, width=0.6)
    save(fig, "fig2_coverage_vs_selection")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-data", action="store_true", help="redraw from results/figures/figure_data.json")
    ap.add_argument("--accent", default=None, help="hex colour for the held-out splits (default: ACCENT)")
    a = ap.parse_args()
    if a.accent:
        global ACCENT
        ACCENT = a.accent
    OUT.mkdir(parents=True, exist_ok=True)
    data_file = OUT / "figure_data.json"
    if a.from_data:
        data = json.load(open(data_file))
        ret = data["retrieval_recall@10"]
        llm = llm_numbers() or data["gemma"]
    else:
        ret, llm = retrieval_contrasts(), llm_numbers()
    data_file.write_text(json.dumps({"retrieval_recall@10": ret, "gemma": llm}, indent=1))
    setup()
    figure1(ret, llm)
    if llm:
        figure2(llm)
    print("figures written to", OUT, "| splits:", sorted(set(ret) | set(llm)))


if __name__ == "__main__":
    main()
