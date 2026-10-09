"""Paper figures (aggregates only; no competition text is drawn). Numbered as in the paper.

Figure 1 (fig_profiles)  — share of fixes touching async definitions / module-level code per repository
                           (results/profiles/summary.json from scripts/16_profile_repos.py).
Figure 2 (fig_selection) — (a) Gemma's Hit@5 gain against the gain in candidate coverage, per split, with the
                           identity line; (b) Gemma's Hit@5 when a gold location is among its candidates, per
                           split and graph.
Figure 3 (fig_forest)    — forest plot: our graph minus the released graph (or its emulated schema), per split,
                           for BM25 Recall@10, BM25+PPR fusion with `calls`-only propagation Recall@10 (the
                           pre-registered H1 contrast) and Gemma 4 Hit@5; filled marker = 95% CI excludes zero.
Figure 4 (fig_agent)     — Experiment 5, Gemma 4 as a graph-navigating agent: (a) Hit@5, our graph minus the
                           released graph, per split; (b) the same difference on tasks with an edited location
                           missing from the released graph vs tasks with all of them present (pre-registered H7).
                           From results/agent/gemma-4-31b-it/answers.jsonl, pilot tasks excluded.
Plotted numbers are written to results/figures/figure_data.json (paired bootstrap 95% CIs, 5,000 resamples, seed 0).

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


AGENT = Path("results/agent/gemma-4-31b-it/answers.jsonl")
PILOT = Path("results/agent/pilot/gemma-4-31b-it/answers.jsonl")


def agent_numbers():
    """Per split: Hit@5 difference (cgl - official) and the same split by whether the released graph
    (or its schema) contains every edited location. Definitions as in scripts/18_eval_agent.py."""
    if not AGENT.exists():
        return {}
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_agent", Path(__file__).with_name("18_eval_agent.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    skip = {(r["split"], r["task_id"]) for r in map(json.loads, open(PILOT))} if PILOT.exists() else set()
    recs = {}
    for line in open(AGENT):
        r = json.loads(line)
        if r.get("ok") and (r["split"], r["task_id"]) not in skip:
            recs[r["key"]] = r
    by = {}
    for r in recs.values():
        by.setdefault((r["split"], r["task_id"]), {})[r["condition"]] = r
    rows = [c for c in by.values() if {"official", "cgl"} <= set(c)]
    groups = {k: [c for c in rows if c["cgl"]["split"] == k] for k, *_ in SPLITS}
    groups["heldout"] = groups["swebl"] + groups["pymatgen"]
    groups["all"] = rows
    out = {}
    for k, cs in groups.items():
        if not cs:
            continue
        d = [ev.hit5(c["cgl"]) - ev.hit5(c["official"]) for c in cs]
        miss = [x for x, c in zip(d, cs) if not ev.complete(c)]
        pres = [x for x, c in zip(d, cs) if ev.complete(c)]
        reached = {}
        for stratum, keep in (("missing", False), ("present", True)):
            sub = [c for c in cs if ev.complete(c) == keep]
            if sub:
                reached[stratum] = {cond: round(float(np.mean([bool(c[cond]["gold_seen"]) for c in sub])), 4)
                                    for cond in ("official", "cgl")}
        out[k] = {"n": len(cs), "hit5_diff": boot_ci(d),
                  "n_missing": len(miss), "gain_missing": boot_ci(miss) if miss else None,
                  "n_present": len(pres), "gain_present": boot_ci(pres) if pres else None,
                  "reached_by_stratum": reached}
    # per repository: agent gain vs the cgl-profile share of fixes a released-style graph cannot fully represent
    prof_file = Path("results/profiles/summary.json")
    if prof_file.exists():
        from scipy.stats import spearmanr
        prof = json.load(open(prof_file))
        per = {}
        for c in rows:
            per.setdefault(c["cgl"]["repo"], []).append(ev.hit5(c["cgl"]) - ev.hit5(c["official"]))
        reps = sorted(r for r in per if r in prof)
        gap = [1 - prof[r]["share_of_fixes"]["fully representable by a released-style graph"][0] for r in reps]
        gain = [float(np.mean(per[r])) for r in reps]
        rho, pval = spearmanr(gap, gain)
        out["by_repo"] = {"repos": {r: {"n": len(per[r]), "gain": round(g, 4), "profile_gap": round(x, 4)}
                                    for r, g, x in zip(reps, gain, gap)},
                          "spearman_rho": round(float(rho), 3), "spearman_p": round(float(pval), 3)}
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
def fig_forest(ret, llm):
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
    save(fig, "figure3_coverage_forest")


# ----------------------------------------------------------------------------- figure 2
def fig_selection(llm):
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
    save(fig, "figure2_coverage_vs_selection")



# ----------------------------------------------------------------------------- figure 3
DEV_REPOS = ["fastapi", "httpx", "requests", "rich"]
HELD_REPOS = ["astropy", "django", "flask", "matplotlib", "pylint", "pytest", "scikit-learn", "seaborn",
              "sphinx", "sympy", "xarray", "pymatgen"]


def fig_profiles(prof):
    """Share of fixes touching async definitions / module-level code, per repository (cgl-profile)."""
    layout, y = [], 0.0
    for g, title, names in (("dev", "Development repositories", DEV_REPOS),
                            ("held", "Held-out repositories", HELD_REPOS)):
        names = [n for n in names if n in prof]
        if not names:
            continue
        layout.append(("header", title, y, g))
        y += 0.8
        for n in names:
            layout.append(("row", n, y, g))
            y += 0.62
        y += 0.3
    ymax = y - 0.3 - 0.2
    panels = [("touching async code", "Fixes touching async code"),
              ("touching module-level code", "Fixes touching module-level code")]
    fig = plt.figure(figsize=(7.2, 0.25 * ymax + 1.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.16)
    lab_ax = fig.add_subplot(gs[0])
    axes = [fig.add_subplot(gs[i + 1], sharey=lab_ax) for i in range(2)]
    for ax in [lab_ax] + axes:
        ax.set_ylim(ymax + 0.2, -0.5)
        ax.patch.set_alpha(0)
    lab_ax.axis("off")
    held = [yy for kind, _, yy, g in layout if g == "held"]
    if held:
        tr = blended_transform_factory(fig.transFigure, lab_ax.transData)
        fig.add_artist(Rectangle((0.0, min(held) - 0.4), 1.0, max(held) - min(held) + 0.4 + 0.33,
                                 transform=tr, facecolor=BAND, edgecolor="none", zorder=-1))
    for kind, what, yy, g in layout:
        tr = lab_ax.get_yaxis_transform()
        if kind == "header":
            lab_ax.text(0.0, yy, what.upper(), fontsize=6.8, fontweight="bold", color=ACCENT if g == "held" else INK2,
                        va="center", ha="left", transform=tr)
        else:
            lab_ax.text(0.04, yy, what, fontsize=8, color=INK, va="center", ha="left", transform=tr)
            lab_ax.text(0.98, yy, f"{prof[what]['n_fixes']:,} fixes", fontsize=6.8, color=MUTED, va="center",
                        ha="right", transform=tr)
    for ax, (key, title) in zip(axes, panels):
        ax.set_xlim(0, 50)
        for xv in (10, 20, 30, 40):
            ax.axvline(xv, color=GRID, lw=0.6, zorder=1)
        ax.axvline(0, color=ZERO, lw=0.8, zorder=2)
        for kind, n, yy, g in layout:
            if kind != "row":
                continue
            m, lo, hi = pts(prof[n]["share_of_fixes"][key])
            col = ACCENT if g == "held" else INK
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.2, solid_capstyle="butt", zorder=3)
            ax.plot(m, yy, "o", ms=4.8, mfc=col, mec="white", mew=0.8, zorder=4)
            halo = [patheffects.withStroke(linewidth=2.4, foreground=BAND if g == "held" else "white")]
            ax.text(hi + 1.2, yy, f"{m:.0f}%", fontsize=6.6, color=col, ha="left", va="center", zorder=5,
                    path_effects=halo)
        ax.set_title(title, fontsize=8.6, color=INK, loc="left", pad=6, fontweight="bold")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%"])
    fig.text(0.985, -0.005, "Code+test commits, 2019 to Sep 2026 (pymatgen: to Mar 2026); 95% bootstrap CIs. "
             "Module-level code includes new top-level definitions.", ha="right", va="bottom", fontsize=6.8,
             color=MUTED)
    save(fig, "figure1_fix_profiles")

# ----------------------------------------------------------------------------- figure 4
def fig_agent(ag):
    """Experiment 5: (a) agent Hit@5 difference per split; (b) by whether the released graph has the code."""
    rows_a = [k for k in ("comp", "public", "swebl", "pymatgen", "heldout") if k in ag]
    rows_b = [k for k in ("comp", "public", "swebl", "pymatgen", "all") if k in ag]
    name = dict(LABEL, heldout="Held-out pooled", all="All tasks")
    group = dict(GROUP, heldout="held", all="all")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.6})

    def stylize(ax, order, xlim, ticks):
        ax.set_xlim(*xlim)
        for xv in ticks:
            if xv:
                ax.axvline(xv, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=ZERO, lw=0.8, zorder=1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:+d}".replace("-", "\u2212").replace("+0", "0") for v in ticks])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([name[k] for k in order], fontsize=8)
        for t, k in zip(ax.get_yticklabels(), order):
            t.set_color(ACCENT if group[k] == "held" else INK)
            if k in ("heldout", "all"):
                t.set_fontweight("bold")
        ax.set_ylim(len(order) - 0.22, -0.75)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", left=False)

    def mark(ax, v, yy, col, filled, marker="o", where="above", text_col=None):
        m, lo, hi = pts(v)
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.2, solid_capstyle="butt", zorder=2)
        ax.plot(m, yy, marker, ms=5.6 if marker == "o" else 5.0, mfc=col if filled else "white", mec=col, mew=1.3,
                zorder=3)
        dy, va = (-0.2, "bottom") if where == "above" else (0.2, "top")
        ax.text(m, yy + dy, f"{m:+.1f}".replace("-", "\u2212"), fontsize=6.8, color=text_col or col, ha="center",
                va=va, zorder=4, path_effects=[patheffects.withStroke(linewidth=2.4, foreground="white")])

    # (a)
    for i, k in enumerate(rows_a):
        col = ACCENT if group[k] == "held" else INK
        v = ag[k]["hit5_diff"]
        mark(a, v, i, col, filled=(v[1] > 0 or v[2] < 0))
    stylize(a, rows_a, (-9, 13), [-5, 0, 5, 10])
    a.set_xlabel("Hit@5, our graph minus the released graph (points)", fontsize=7.6)
    a.set_title("a   The agent with each graph", fontsize=8.6, color=INK, loc="left", fontweight="bold", pad=8)

    # (b)
    for i, k in enumerate(rows_b):
        col = ACCENT if group[k] == "held" else INK
        for key, dy, c, mk, where, tcol in (("gain_missing", -0.17, col, "o", "above", None),
                                            ("gain_present", 0.17, RELEASED, "D", "below", MUTED)):
            v = ag[k][key]
            if v:
                mark(b, v, i + dy, c, filled=(v[1] > 0 or v[2] < 0), marker=mk, where=where, text_col=tcol)
    stylize(b, rows_b, (-20, 34), [-10, 0, 10, 20, 30])
    b.set_xlabel("Hit@5, our graph minus the released graph (points)", fontsize=7.6)
    b.set_title("b   The gain comes from code the released graph lacks", fontsize=8.6, color=INK, loc="left",
                fontweight="bold", pad=8)
    handles = [Line2D([], [], marker="o", ls="", ms=5.6, mfc="white", mec=INK, mew=1.3,
                      label="tasks with an edited location missing from the released graph"),
               Line2D([], [], marker="D", ls="", ms=5.0, mfc="white", mec=RELEASED, mew=1.3,
                      label="tasks with every edited location present")]
    b.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.19), ncol=1, fontsize=7, frameon=False,
             labelcolor=INK2, handletextpad=0.3, borderaxespad=0, borderpad=0)
    a.text(0, -0.215, "Both panels: filled marker = 95% CI excludes 0", transform=a.transAxes, fontsize=7,
           color=MUTED, va="top")
    save(fig, "figure4_agent")


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
    ag = agent_numbers() or (json.load(open(data_file)).get("agent", {}) if data_file.exists() else {})
    data_file.write_text(json.dumps({"retrieval_recall@10": ret, "gemma": llm, "agent": ag}, indent=1))
    setup()
    fig_forest(ret, llm)
    if llm:
        fig_selection(llm)
    if ag:
        fig_agent(ag)
    prof_file = Path("results/profiles/summary.json")
    if prof_file.exists():
        fig_profiles(json.load(open(prof_file)))
    print("figures written to", OUT, "| splits:", sorted(set(ret) | set(llm)))


if __name__ == "__main__":
    main()
