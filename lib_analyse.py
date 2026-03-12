"""
Bibliothèque d'analyse statistique — Projet BI
Centralise visualisation et analyses univariée / bivariée
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, pearsonr
from itertools import combinations

# ── Style global ──────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_PALETTE     = "Set2"
_COLOR_MAIN  = "#4C72B0"
_COLOR_MEAN  = "#e74c3c"
_COLOR_MED   = "#f39c12"

# ── Helpers ───────────────────────────────────────────────────────────────────

def log_graph_generated(path: str) -> None:
    print(f"  → {path}")


def _mkdir(base: str, sub: str) -> str:
    """Crée base/sub et retourne le chemin complet."""
    target = os.path.join(base, sub)
    os.makedirs(target, exist_ok=True)
    return target


def _save(fig: plt.Figure, base: str, sub: str, filename: str) -> str:
    folder = _mkdir(base, sub)
    path   = os.path.join(folder, filename)
    fig.savefig(path)
    plt.close(fig)
    log_graph_generated(path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE UNIVARIÉE
# ══════════════════════════════════════════════════════════════════════════════

def analyse_univariee(df: pd.DataFrame,
                      quant_vars: list,
                      qual_vars: list,
                      dossier: str) -> pd.DataFrame:
    """
    Analyse univariée complète.
    - Quantitatives : histogramme + KDE + lignes moyenne / médiane
    - Qualitatives  : barres horizontales avec annotations fréquence + %
    Retourne un DataFrame de statistiques descriptives.
    """
    os.makedirs(dossier, exist_ok=True)
    stats_list = []

    # ── Quantitatives ─────────────────────────────────────────────────────────
    for col in quant_vars:
        if col not in df.columns:
            continue

        data   = df[col].dropna()
        mean   = data.mean()
        median = data.median()
        std    = data.std()
        q1     = data.quantile(0.25)
        q3     = data.quantile(0.75)
        iqr    = q3 - q1

        stats_list.append({
            "variable": col, "type": "quantitative",
            "mean": mean, "median": median, "std": std,
            "variance": data.var(), "min": data.min(), "max": data.max(),
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "n_valides": len(data), "n_manquants": df[col].isna().sum(),
        })

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data, bins=40, kde=True, ax=ax,
                     color=_COLOR_MAIN, edgecolor="white", alpha=0.8,
                     line_kws={"linewidth": 2})
        ax.axvline(mean,   color=_COLOR_MEAN, linestyle="--",
                   linewidth=1.8, label=f"Moyenne : {mean:.2f}")
        ax.axvline(median, color=_COLOR_MED,  linestyle="-",
                   linewidth=1.8, label=f"Médiane : {median:.2f}")
        ax.set_title(f"Distribution de {col}", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Fréquence", fontsize=12)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

        _save(fig, dossier, "histogrammes", f"{col}_hist.png")

    # ── Qualitatives ──────────────────────────────────────────────────────────
    for col in qual_vars:
        if col not in df.columns:
            continue

        vc    = df[col].value_counts()
        n_cat = len(vc)
        total = vc.sum()

        stats_list.append({
            "variable": col, "type": "qualitative",
            "mode": vc.index[0], "n_categories": n_cat,
            "n_valides": total, "n_manquants": df[col].isna().sum(),
        })

        fig_h  = min(max(4, n_cat * 0.38), 22)
        fig, ax = plt.subplots(figsize=(11, fig_h))
        palette = sns.color_palette(_PALETTE, n_colors=min(n_cat, 8))
        colors  = [palette[i % len(palette)] for i in range(n_cat)]

        bars = ax.barh(vc.index.astype(str), vc.values,
                       color=colors, edgecolor="white", height=0.65)
        ax.invert_yaxis()

        for bar, count in zip(bars, vc.values):
            pct = 100 * count / total
            ax.text(
                bar.get_width() + total * 0.004,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}  ({pct:.1f} %)",
                va="center", ha="left", fontsize=9, color="#333333",
            )

        ax.set_title(f"Distribution de {col}", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_xlabel("Nombre d'occurrences", fontsize=12)
        ax.set_ylabel(col, fontsize=11)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        ax.margins(x=0.18)

        _save(fig, dossier, "barres", f"{col}_bar.png")

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(os.path.join(dossier, "statistiques_univariees.csv"),
                    index=False)
    return stats_df


# ══════════════════════════════════════════════════════════════════════════════
# DÉTECTION DE TYPE
# ══════════════════════════════════════════════════════════════════════════════

def detect_type(series: pd.Series) -> str:
    return "quantitative" if pd.api.types.is_numeric_dtype(series) else "qualitative"


def get_var_types(df: pd.DataFrame) -> tuple[list, list]:
    quant = [c for c in df.columns if detect_type(df[c]) == "quantitative"]
    qual  = [c for c in df.columns if detect_type(df[c]) == "qualitative"]
    return quant, qual


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE BIVARIÉE
# ══════════════════════════════════════════════════════════════════════════════

# ── V de Cramér ───────────────────────────────────────────────────────────────

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2  = chi2_contingency(ct)[0]
    n     = ct.sum().sum()
    r, k  = ct.shape
    denom = n * (min(k - 1, r - 1))
    return np.nan if denom == 0 else float(np.sqrt(chi2 / denom))


# ── Qual vs Qual ──────────────────────────────────────────────────────────────

def analyse_qual_qual(df, var1, var2, folder,
                      generate_plot_only_if_associated=False,
                      association_threshold=0.3):
    """
    Heatmap de contingence (comptes absolus + intensité proportionnelle par ligne)
    + V de Cramér en titre.
    """
    val = cramers_v(df[var1], df[var2])
    ct  = pd.crosstab(df[var1], df[var2])
    has_association = (val is not None and not pd.isna(val)
                       and val >= association_threshold)
    graph_path = None

    if (not generate_plot_only_if_associated) or has_association:
        ct_pct  = ct.div(ct.sum(axis=1), axis=0)   # proportion par ligne
        n_rows, n_cols = ct.shape
        fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.9),
                                        max(5, n_rows * 0.6)))

        sns.heatmap(ct_pct, annot=ct.values, fmt="d",
                    cmap="Blues", vmin=0, vmax=1,
                    linewidths=0.4, linecolor="white",
                    annot_kws={"size": 9}, ax=ax, cbar_kws={"label": "Proportion"})

        v_str = f"{val:.3f}" if not pd.isna(val) else "N/A"
        ax.set_title(f"{var1} vs {var2}  —  V de Cramér = {v_str}",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(var2, fontsize=11)
        ax.set_ylabel(var1, fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)

        graph_path = _save(fig, folder, "qual_qual", f"{var1}_{var2}.png")

    return {"var1": var1, "var2": var2, "type": "qual-qual",
            "metric": "cramers_v", "value": val,
            "has_association": has_association, "graph_path": graph_path}


# ── Qual vs Quant ─────────────────────────────────────────────────────────────

def analyse_qual_quant(df, qual, quant, folder,
                       generate_plot_only_if_significant=False,
                       anova_pvalue_threshold=0.05):
    """
    Violin plot (avec boîte intérieure) par modalité + test ANOVA.
    Remplace les boîtes à moustaches pour une lecture plus riche de la distribution.
    """
    groups = [
        df[df[qual] == g][quant].dropna()
        for g in df[qual].dropna().unique()
        if len(df[df[qual] == g][quant].dropna()) > 1
    ]
    if len(groups) < 2:
        return None

    stat, p = f_oneway(*groups)
    has_significant_difference = p < anova_pvalue_threshold
    graph_path = None

    if (not generate_plot_only_if_significant) or has_significant_difference:
        n_cat   = df[qual].nunique()
        fig_w   = max(9, n_cat * 1.1)
        fig, ax = plt.subplots(figsize=(fig_w, 6))

        sns.violinplot(
            x=df[qual].astype(str), y=df[quant],
            inner="box",        # boîte IQR + médiane à l'intérieur du violon
            palette=_PALETTE,
            linewidth=1.2,
            cut=0,              # limite le violon à l'étendue des données
            ax=ax,
        )

        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        ax.set_title(f"{qual} vs {quant}  —  ANOVA p = {p_str}",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(qual, fontsize=11)
        ax.set_ylabel(quant, fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)

        graph_path = _save(fig, folder, "qual_quant", f"{qual}_{quant}.png")

    return {"var1": qual, "var2": quant, "type": "qual-quant",
            "metric": "anova_pvalue", "value": p,
            "has_significant_difference": has_significant_difference,
            "graph_path": graph_path}


# ── Quant vs Quant ────────────────────────────────────────────────────────────

def analyse_quant_quant(df, var1, var2, folder,
                        generate_plot_only_if_correlated=False,
                        corr_threshold=0.3,
                        pvalue_threshold=0.05):
    """
    Nuage de points + droite de régression OLS.
    Corrélations Pearson, Spearman, Kendall affichées en annotation.
    """
    data = df[[var1, var2]].dropna()
    if len(data) < 3:
        return None

    pearson_r  = data[var1].corr(data[var2], method="pearson")
    spearman_r = data[var1].corr(data[var2], method="spearman")
    kendall_r  = data[var1].corr(data[var2], method="kendall")
    _, p_value = pearsonr(data[var1], data[var2])

    has_correlation = (
        p_value < pvalue_threshold and (
            abs(pearson_r)  >= corr_threshold or
            abs(spearman_r) >= corr_threshold or
            abs(kendall_r)  >= corr_threshold
        )
    )
    graph_path = None

    if (not generate_plot_only_if_correlated) or has_correlation:
        n     = len(data)
        alpha = max(0.05, min(0.6, 2000 / n)) if n > 500 else 0.6

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(data[var1], data[var2],
                   alpha=alpha, s=16, color=_COLOR_MAIN, edgecolors="none")

        # droite de régression
        coef   = np.polyfit(data[var1], data[var2], 1)
        x_line = np.linspace(data[var1].min(), data[var1].max(), 300)
        ax.plot(x_line, np.poly1d(coef)(x_line),
                color=_COLOR_MEAN, linestyle="--", linewidth=2, label="Régression OLS")

        p_str = f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}"
        stats_txt = (
            f"Pearson r = {pearson_r:.3f}   |   "
            f"Spearman ρ = {spearman_r:.3f}   |   "
            f"Kendall τ = {kendall_r:.3f}   |   "
            f"p = {p_str}"
        )
        ax.set_title(f"{var1} vs {var2}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(var1, fontsize=11)
        ax.set_ylabel(var2, fontsize=11)
        ax.legend(fontsize=9)
        ax.text(0.5, -0.14, stats_txt,
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4",
                          edgecolor="#cccccc", alpha=0.9))

        graph_path = _save(fig, folder, "quant_quant", f"{var1}_{var2}.png")

    return {"var1": var1, "var2": var2, "type": "quant-quant",
            "metric": "pearson_corr", "value": pearson_r,
            "p_value": p_value, "spearman_corr": spearman_r,
            "kendall_corr": kendall_r, "has_correlation": has_correlation,
            "graph_path": graph_path}


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING DES RELATIONS INTÉRESSANTES
# ══════════════════════════════════════════════════════════════════════════════

def _format_interesting_entry(res: dict, graph_path: str) -> str:
    rt = res["type"]
    if rt == "quant-quant":
        return (f"[FORT] {res['var1']} vs {res['var2']} | "
                f"Pearson={res['value']:.3f} | p={res['p_value']:.4g} | "
                f"graph={graph_path}")
    if rt == "qual-quant":
        return (f"[SIGNIFICATIF] {res['var1']} vs {res['var2']} | "
                f"ANOVA p={res['value']:.4g} | graph={graph_path}")
    return (f"[FORT] {res['var1']} vs {res['var2']} | "
            f"CramersV={res['value']:.3f} | graph={graph_path}")


def _is_interesting(res: dict) -> bool:
    rt    = res["type"]
    value = res.get("value")
    if value is None or pd.isna(value):
        return False
    if rt == "quant-quant":
        p = res.get("p_value")
        return p is not None and not pd.isna(p) and abs(value) >= 0.5 and p < 0.05
    if rt == "qual-quant":
        return value < 0.05
    if rt == "qual-qual":
        return value >= 0.3
    return False


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION BIVARIÉE
# ══════════════════════════════════════════════════════════════════════════════

def analyse_table(df: pd.DataFrame,
                  folder: str,
                  progress_callback=None,
                  interesting_file: str | None = None,
                  excluded_columns: list | None = None,
                  generate_bivariate_plots_only_if_significant: bool = False,
                  generate_quant_plots_only_if_correlated: bool = False,
                  cramers_v_threshold: float = 0.3,
                  anova_pvalue_threshold: float = 0.05,
                  corr_threshold: float = 0.3,
                  pvalue_threshold: float = 0.05) -> pd.DataFrame:
    """Analyse bivariée automatique de toutes les paires de variables d'une table."""
    results  = []
    types    = {col: detect_type(df[col]) for col in df.columns}
    excluded = set(excluded_columns or [])
    cols     = [col for col in df.columns if col not in excluded]
    pairs    = list(combinations(cols, 2))
    total    = len(pairs)

    if interesting_file:
        os.makedirs(os.path.dirname(os.path.abspath(interesting_file)),
                    exist_ok=True)
        with open(interesting_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== {folder} ===\n")

    for idx, (v1, v2) in enumerate(pairs, start=1):
        t1, t2 = types[v1], types[v2]
        res = None

        if t1 == "qualitative" and t2 == "qualitative":
            res = analyse_qual_qual(
                df, v1, v2, folder,
                generate_plot_only_if_associated=generate_bivariate_plots_only_if_significant,
                association_threshold=cramers_v_threshold,
            )
        elif t1 == "qualitative" and t2 == "quantitative":
            res = analyse_qual_quant(
                df, v1, v2, folder,
                generate_plot_only_if_significant=generate_bivariate_plots_only_if_significant,
                anova_pvalue_threshold=anova_pvalue_threshold,
            )
        elif t1 == "quantitative" and t2 == "qualitative":
            res = analyse_qual_quant(
                df, v2, v1, folder,
                generate_plot_only_if_significant=generate_bivariate_plots_only_if_significant,
                anova_pvalue_threshold=anova_pvalue_threshold,
            )
        elif t1 == "quantitative" and t2 == "quantitative":
            res = analyse_quant_quant(
                df, v1, v2, folder,
                generate_plot_only_if_correlated=(
                    generate_quant_plots_only_if_correlated
                    or generate_bivariate_plots_only_if_significant
                ),
                corr_threshold=corr_threshold,
                pvalue_threshold=pvalue_threshold,
            )

        if res:
            results.append(res)
            if interesting_file and _is_interesting(res):
                gp   = res.get("graph_path") or os.path.join(folder, f"{v1}_{v2}.png")
                line = _format_interesting_entry(res, gp)
                with open(interesting_file, "a", encoding="utf-8") as f:
                    f.write(f"{line}\n")

        if progress_callback:
            progress_callback(idx, total, v1, v2)

    return pd.DataFrame(results) if results else pd.DataFrame()
