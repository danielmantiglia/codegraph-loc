"""Paper figures (aggregates only; no competition text is drawn). Numbered as in the paper.

Figure 1 (fig_profiles)  — share of fixes touching async definitions / module-level code per repository
                           (results/profiles/summary.json from scripts/16_profile_repos.py).
Figure 2 (fig_selection) — (a) Gemma's Hit@5 gain against the gain in candidate coverage, per split, with the
                           identity line; (b) Gemma's Hit@5 when a gold location is among its candidates, per
                           split and graph.
Figure 3 (fig_forest)    — our graph minus the released graph (or its emulated schema), per split, for BM25
                           Recall@10, BM25+PPR fusion with `calls`-only propagation Recall@10 (the pre-registered
                           H1 contrast) and Gemma 4 Hit@5; solid bar = 95% CI excludes zero.
Figure 4 (fig_agent)     — Experiment 5, Gemma 4 as a graph-navigating agent: (a) Hit@5, our graph minus the
                           released graph, per split; (b) the same difference on tasks with an edited location
                           missing from the released graph vs tasks with all of them present (pre-registered H7).
                           From results/agent/gemma-4-31b-it/answers.jsonl, pilot tasks excluded.
Plotted numbers are written to results/figures/figure_data.json (paired bootstrap 95% CIs, 5,000 resamples, seed 0).
Every figure is a 16:9 page (2200 x 1238 px), so the same PNG serves the paper and the Kaggle media gallery.
Font: IBM Plex Sans (SIL Open Font License), shipped in fonts/; without it matplotlib falls back to FreeSans or
DejaVu Sans and only the typeface changes.

Usage:  python scripts/14_figures.py              # recompute from per-task records, then draw
        python scripts/14_figures.py --from-data  # redraw from results/figures/figure_data.json
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager, patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

OUT = Path("results/figures")
FONT_DIR = Path(__file__).resolve().parents[1] / "fonts"

# Design tokens (editorial): a cream page, one orange accent for the code the released graph omits and for
# what including it gains, slate for the released graph, pale tints when a 95% CI includes zero.
BG, BAND, GRID = "#f6f4ef", "#ebe7de", "#e2ded5"
INK, INK2, MUTED = "#1d232b", "#4e5661", "#8d949c"
ACCENT, ACCENT_PALE = "#d9622b", "#efc2a6"
SLATE, SLATE_PALE = "#71819a", "#c9d0d9"
FONT = ["IBM Plex Sans", "FreeSans", "DejaVu Sans"]
FIGSIZE, LEFT = (10, 5.625), 0.045

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
    for f in sorted(FONT_DIR.glob("*.ttf")):          # IBM Plex Sans (OFL), shipped in fonts/
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 10,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9.5, "ytick.labelsize": 10,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def canvas(title, subtitle, note):
    """A 16:9 page (one file serves the paper body and the Kaggle gallery): title, subtitle, source note."""
    fig = plt.figure(figsize=FIGSIZE)
    fig.text(LEFT, 0.94, title, fontsize=15.5, fontweight="bold", color=INK, va="top")
    fig.text(LEFT, 0.872, subtitle, fontsize=10.5, color=INK2, va="top")
    fig.text(LEFT, 0.03, note, fontsize=8.5, color=MUTED, va="bottom")
    return fig


def legend(fig, items, x, y):
    """Swatches laid out left to right in figure coordinates. kind: 'bar', 'pale', 'dot', 'band'."""
    r = fig.canvas.get_renderer()
    for kind, col, lab in items:
        if kind == "dot":
            fig.add_artist(Line2D([x + 0.006], [y], marker="o", ms=7.5, ls="", color=col, mec=BG, mew=1.2,
                                  transform=fig.transFigure))
        else:
            fig.add_artist(Rectangle((x, y - 0.0115), 0.0125, 0.023, transform=fig.transFigure, facecolor=col,
                                     edgecolor=MUTED if kind == "band" else "none", lw=0.4))
        t = fig.text(x + 0.019, y, lab, fontsize=9.5, color=INK2, va="center")
        x = t.get_window_extent(renderer=r).transformed(fig.transFigure.inverted()).x1 + 0.024


def signed(v, nd=1):
    return f"{v:+.{nd}f}".replace("-", "−")


def tick(v):
    return "0" if v == 0 else signed(v, 0)


def pts(v):
    return [100 * x for x in v]


def bar(ax, v, y, h, col, pale, text=None):
    """A bar from 0 to the estimate, ink whiskers for the 95% CI, the value right of the whisker.
    Solid = the CI excludes 0, pale = it includes 0."""
    m, lo, hi = pts(v)
    sig = lo > 0 or hi < 0
    ax.barh(y, m, height=h, color=col if sig else pale, lw=0, zorder=2)
    ax.plot([lo, hi], [y, y], color=INK, lw=0.9, zorder=3, solid_capstyle="butt")
    for e in (lo, hi):
        ax.plot([e, e], [y - h * 0.3, y + h * 0.3], color=INK, lw=0.9, zorder=3)
    ax.annotate(text or signed(m), (max(hi, 0), y), xytext=(4, 0), textcoords="offset points", va="center",
                ha="left", fontsize=8.8, color=INK2, zorder=4, annotation_clip=False)


def xaxis(ax, xlim, ticks, fmt=signed, zero=True):
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt(t) for t in ticks])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if zero:
        ax.axvline(0, color=INK, lw=0.9, zorder=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    ax.patch.set_alpha(0)


def band(fig, ax, y0, y1, x0=LEFT - 0.012, x1=0.985):
    """Shaded rows (held-out splits / repositories), spanning labels and panels."""
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=tr, facecolor=BAND, edgecolor="none",
                             zorder=-1))


def save(fig, name):
    for ext, kw in (("png", {"dpi": 220}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)


def grouped_rows(groups, header_h, row_h, gap):
    """[(kind, item, y, group)] top to bottom for ('dev'|'held', title, items) groups."""
    layout, y = [], 0.0
    for g, title, items in groups:
        if not items:
            continue
        layout.append(("header", title, y, g))
        y += header_h
        for it in items:
            layout.append(("row", it, y, g))
            y += row_h
        y += gap
    return layout, y - gap - row_h / 2


# ----------------------------------------------------------------------------- figure 1
DEV_REPOS = ["fastapi", "httpx", "requests", "rich"]
HELD_REPOS = ["astropy", "django", "flask", "matplotlib", "pylint", "pytest", "scikit-learn", "seaborn",
              "sphinx", "sympy", "xarray", "pymatgen"]


def fig_profiles(prof):
    """Share of fixes touching async definitions / module-level code, per repository (cgl-profile)."""
    fig = canvas("Fixes touch async code in two libraries, module-level code everywhere",
                 "Share of code+test fixes that touch each kind of code, per repository; whiskers: 95% bootstrap CI",
                 "Source: cgl-profile on each repository's history, 2019 to Sep 2026 (pymatgen: to Mar 2026). "
                 "Module-level code includes new top-level definitions.")
    layout, ymax = grouped_rows([("dev", "Development repositories", [n for n in DEV_REPOS if n in prof]),
                                 ("held", "Held-out repositories", [n for n in HELD_REPOS if n in prof])],
                                header_h=0.95, row_h=0.72, gap=0.35)
    bottom, top = 0.115, 0.80
    lab = fig.add_axes([LEFT, bottom, 0.26, top - bottom])
    axes = [fig.add_axes([0.335, bottom, 0.29, top - bottom]), fig.add_axes([0.68, bottom, 0.29, top - bottom])]
    for ax in [lab] + axes:
        ax.set_ylim(ymax + 0.25, -0.55)
    lab.axis("off")
    held = [y for kind, _, y, g in layout if g == "held"]
    band(fig, lab, min(held) - 0.5, max(held) + 0.45)
    tr = lab.get_yaxis_transform()
    for kind, what, y, g in layout:
        if kind == "header":
            lab.text(0, y, what.upper(), fontsize=8.3, fontweight="semibold", color=INK2, va="center", transform=tr)
        else:
            lab.text(0.03, y, what, fontsize=9.6, color=INK, va="center", transform=tr)
            lab.text(0.97, y, f"{prof[what]['n_fixes']:,} fixes", fontsize=8.5, color=MUTED, va="center",
                     ha="right", transform=tr)
    for ax, key, title in zip(axes, ("touching async code", "touching module-level code"),
                              ("Async code", "Module-level code")):
        for kind, n, y, g in layout:
            if kind == "row":
                v = prof[n]["share_of_fixes"][key]
                bar(ax, v, y, 0.5, ACCENT, ACCENT, text=f"{100 * v[0]:.0f}%")
        xaxis(ax, (0, 50), [0, 10, 20, 30, 40], fmt=lambda t: f"{t}%", zero=False)
        ax.axvline(0, color=INK, lw=0.9, zorder=3)
        ax.set_title(title, fontsize=10.5, fontweight="semibold", color=INK, loc="left", pad=6)
    save(fig, "figure1_fix_profiles")


# ----------------------------------------------------------------------------- figure 2
def fig_selection(llm):
    ks = [k for k, *_ in SPLITS if k in llm]
    fig = canvas("The graph sets the ceiling for Gemma 4",
                 "Gemma 4 picks five of about 30 graph candidates; our graph vs the released graph (or its schema), "
                 "with 95% CIs",
                 "Source: codegraph-loc, Experiments 3 and 4 (Gemma 4 31B re-ranking). Points in panel (a) are "
                 "paired differences, in percentage points.")
    a = fig.add_axes([0.105, 0.17, 0.34, 0.56])
    b = fig.add_axes([0.66, 0.17, 0.31, 0.56])
    # (a) gain tracks coverage
    xl, yl = (-5.5, 13), (-6.5, 11.5)
    a.plot(xl, xl, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    a.text(10.3, 8.9, "y = x", rotation=45, transform_rotates_text=True, rotation_mode="anchor", fontsize=8.5,
           color=MUTED, ha="left", va="top")
    place = {"public": "above", "swebl": "below", "comp": (7, -9, "left"), "pymatgen": (-6, 11, "right")}
    for k in ks:
        x, xlo, xhi = pts(llm[k]["cand_hit_diff"])
        y, ylo, yhi = pts(llm[k]["gemma_hit5_diff"])
        a.plot([xlo, xhi], [y, y], color=ACCENT_PALE, lw=2.2, solid_capstyle="round", zorder=2)
        a.plot([x, x], [ylo, yhi], color=ACCENT_PALE, lw=2.2, solid_capstyle="round", zorder=2)
        a.plot(x, y, "o", ms=8, mfc=ACCENT, mec=BG, mew=1.4, zorder=4)
        how = place.get(k, (7, 6, "left"))
        kw = dict(fontsize=9.3, color=INK, zorder=5, path_effects=[patheffects.withStroke(linewidth=3, foreground=BG)])
        if how == "above":
            a.annotate(LABEL[k], (x, yhi), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", **kw)
        elif how == "below":
            a.annotate(LABEL[k], (x, ylo), xytext=(4, -4), textcoords="offset points", ha="center", va="top", **kw)
        else:
            dx, dy, ha = how
            a.annotate(LABEL[k], (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", **kw)
    a.set_xlim(*xl)
    a.set_ylim(*yl)
    a.set_xticks([-4, 0, 4, 8, 12])
    a.set_xticklabels([tick(t) for t in [-4, 0, 4, 8, 12]])
    a.set_yticks([-4, 0, 4, 8])
    a.set_yticklabels([tick(t) for t in [-4, 0, 4, 8]])
    a.grid(color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.axhline(0, color=INK, lw=0.9, zorder=1)
    a.axvline(0, color=INK, lw=0.9, zorder=1)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_xlabel("Change in gold among candidates", fontsize=9.5, labelpad=6)
    a.set_ylabel("Change in Gemma Hit@5", fontsize=9.5, labelpad=4)
    a.set_title("a   Gemma's gain tracks candidate coverage", fontsize=10.5, fontweight="semibold", color=INK,
                loc="left", pad=10, x=(LEFT - 0.105) / 0.34)
    # (b) selection accuracy given coverage
    for i, k in enumerate(ks):
        sel = llm[k]["selection_given_gold"]
        for cond, dy, col in (("official", -0.14, SLATE), ("cgl", 0.14, ACCENT)):
            m, lo, hi = pts(sel[cond])
            b.plot([lo, hi], [i + dy] * 2, color=col, lw=1.1, alpha=0.55, zorder=2)
            b.plot(m, i + dy, "o", ms=7.5, mfc=col, mec=BG, mew=1.3, zorder=3)
        b.text(79.3, i, LABEL[k], ha="right", va="center", fontsize=9.6, color=INK)
    xaxis(b, (80, 100), [80, 85, 90, 95, 100], fmt=lambda t: f"{t}%", zero=False)
    b.set_ylim(len(ks) - 0.45, -0.55)
    b.set_xlabel("Gemma Hit@5 when a gold location is among its candidates", fontsize=9.5, labelpad=6)
    b.set_title("b   Selection accuracy is the same with both graphs", fontsize=10.5, fontweight="semibold",
                color=INK, loc="left", pad=10, x=(0.53 - 0.66) / 0.31)
    legend(fig, [("dot", SLATE, "released graph (or its schema)"), ("dot", ACCENT, "our graph (codegraph-loc)")],
           0.53, 0.80)
    save(fig, "figure2_coverage_vs_selection")


# ----------------------------------------------------------------------------- figure 3
def fig_forest(ret, llm):
    rows = [k for k, *_ in SPLITS if k in ret]
    fig = canvas("Clear gains on the development repositories, small ones held out",
                 "Our graph minus the released graph (or its schema), percentage points; whiskers: paired 95% CI",
                 "Source: codegraph-loc, Experiments 2 to 4. Retrieval excludes test code; BM25 + PPR uses calls-only "
                 "propagation on both graphs (the pre-registered H1 contrast).")
    layout, ymax = grouped_rows([("dev", "Development repositories", [k for k in rows if GROUP[k] == "dev"]),
                                 ("held", "Held out · pre-registered", [k for k in rows if GROUP[k] == "held"])],
                                header_h=0.9, row_h=1.0, gap=0.45)
    bottom, top = 0.12, 0.70
    lab = fig.add_axes([LEFT, bottom, 0.22, top - bottom])
    panels = [("bm25", "BM25", "Recall@10"), ("fusion_calls", "BM25 + PPR", "Recall@10, calls only"),
              ("gemma", "Gemma 4 re-ranking", "Hit@5")]
    axes = [fig.add_axes([0.30 + i * 0.235, bottom, 0.19, top - bottom]) for i in range(3)]
    for ax in [lab] + axes:
        ax.set_ylim(ymax + 0.4, -0.5)
    lab.axis("off")
    held = [y for kind, _, y, g in layout if g == "held"]
    band(fig, lab, min(held) - 0.55, max(held) + 0.55)
    tr = lab.get_yaxis_transform()
    for kind, what, y, g in layout:
        if kind == "header":
            lab.text(0, y, what.upper(), fontsize=8.3, fontweight="semibold", color=INK2, va="center", transform=tr)
        else:
            lab.text(0.04, y, LABEL[what], fontsize=9.8, color=INK, va="center", transform=tr)
            lab.text(0.97, y, f"n = {ret[what]['n']}", fontsize=8.5, color=MUTED, va="center", ha="right",
                     transform=tr)
    for ax, (key, title, metric) in zip(axes, panels):
        for kind, k, y, g in layout:
            if kind != "row":
                continue
            v = llm.get(k, {}).get("gemma_hit5_diff") if key == "gemma" else ret[k].get(key)
            if v:
                bar(ax, v, y, 0.5, ACCENT, ACCENT_PALE)
        xaxis(ax, (-7, 12), [-5, 0, 5, 10], fmt=tick)
        ax.text(0, 1.075, title, transform=ax.transAxes, fontsize=10.5, fontweight="semibold", color=INK, va="bottom")
        ax.text(0, 1.02, metric, transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")
    legend(fig, [("bar", ACCENT, "95% CI excludes 0"), ("pale", ACCENT_PALE, "95% CI includes 0"),
                 ("band", BAND, "held-out splits")], LEFT, 0.80)
    save(fig, "figure3_coverage_forest")


# ----------------------------------------------------------------------------- figure 4
def fig_agent(ag):
    """Experiment 5: (a) agent Hit@5 difference per split; (b) by whether the released graph has the code."""
    rows_a = [k for k in ("comp", "public", "swebl", "pymatgen", "heldout") if k in ag]
    rows_b = [k for k in ("comp", "public", "swebl", "pymatgen", "all") if k in ag]
    name = dict(LABEL, heldout="Held-out pooled", all="All tasks")
    group = dict(GROUP, heldout="held", all="all")
    fig = canvas("The agent gains where the released graph lacks the edited code",
                 "Gemma 4 exploring the graph with tools: Hit@5 with our graph minus the released graph (or its "
                 "schema), points",
                 "Source: codegraph-loc, Experiment 5 (Gemma 4 31B, 2,718 pre-registered episodes). Whiskers: paired "
                 "95% CI. Panel (b): “All tasks” is H7; the other rows are exploratory.")
    bottom, top = 0.12, 0.66
    a = fig.add_axes([0.19, bottom, 0.27, top - bottom])
    b = fig.add_axes([0.665, bottom, 0.305, top - bottom])
    for ax, order, x0, x1, xlab in ((a, rows_a, LEFT - 0.012, 0.485, 0.18), (b, rows_b, 0.51, 0.985, 0.655)):
        ax.set_ylim(len(order) - 0.45, -0.6)
        held = [i for i, k in enumerate(order) if group[k] == "held"]
        band(fig, ax, min(held) - 0.5, max(held) + 0.5, x0=x0, x1=x1)
        for i, k in enumerate(order):
            fig.text(xlab, i, name[k], fontsize=9.8, color=INK, ha="right", va="center",
                     fontweight="semibold" if k in ("heldout", "all") else "normal",
                     transform=blended_transform_factory(fig.transFigure, ax.transData))
    for i, k in enumerate(rows_a):
        bar(a, ag[k]["hit5_diff"], i, 0.48, ACCENT, ACCENT_PALE)
    for i, k in enumerate(rows_b):
        for key, dy, col, pale in (("gain_missing", -0.19, ACCENT, ACCENT_PALE),
                                   ("gain_present", 0.19, SLATE, SLATE_PALE)):
            if ag[k].get(key):
                bar(b, ag[k][key], i + dy, 0.32, col, pale)
    xaxis(a, (-9, 13), [-5, 0, 5, 10], fmt=tick)
    xaxis(b, (-20, 36), [-10, 0, 10, 20, 30], fmt=tick)
    a.set_title("a   Per split", fontsize=10.5, fontweight="semibold", color=INK, loc="left", pad=10, x=(LEFT - 0.19) / 0.27)
    b.set_title("b   By whether the released graph has the edited code", fontsize=10.5, fontweight="semibold",
                color=INK, loc="left", pad=10, x=(0.52 - 0.665) / 0.305)
    legend(fig, [("bar", ACCENT, "an edited location is missing from the released graph"),
                 ("bar", SLATE, "every edited location is present"),
                 ("pale", ACCENT_PALE, "pale: 95% CI includes 0"), ("band", BAND, "held out")], LEFT, 0.80)
    save(fig, "figure4_agent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-data", action="store_true", help="redraw from results/figures/figure_data.json")
    a = ap.parse_args()
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
