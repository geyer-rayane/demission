"""
Bibliothèque d'analyse statistique pour le projet BI
Centralise les méthodes de visualisation et d'analyse univariée et bivariée
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency, f_oneway, pearsonr

# ============================================================
# ANALYSE UNIVARIEE
# ============================================================

def analyse_univariee(df, quant_vars, qual_vars, dossier):
    """
    Effectue une analyse univariée des variables quantitatives et qualitatives.
    Génère des statistiques descriptives et des visualisations.
    """
    stats_list = []

    # VARIABLES QUANTITATIVES
    for col in quant_vars:
        if col in df.columns:
            mean = df[col].mean()
            median = df[col].median()
            std = df[col].std()
            var = df[col].var()
            min_val = df[col].min()
            max_val = df[col].max()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)

            stats_list.append({
                "variable": col,
                "type": "quantitative",
                "mean": mean,
                "median": median,
                "std": std,
                "variance": var,
                "min": min_val,
                "max": max_val,
                "IQR": iqr
            })

            plt.figure()
            sns.histplot(df[col], bins=30)
            plt.title(f"Histogramme {col}")
            plt.xlabel(col)
            plt.ylabel("Frequence")
            plt.savefig(f"{dossier}/{col}_hist.png")
            plt.close()

    # VARIABLES QUALITATIVES
    for col in qual_vars:
        if col in df.columns:
            mode = df[col].mode()[0]

            stats_list.append({
                "variable": col,
                "type": "qualitative",
                "mode": mode
            })

            plt.figure()
            df[col].value_counts().plot(kind="bar")
            plt.title(f"Diagramme barres {col}")
            plt.xlabel(col)
            plt.ylabel("Nombre")
            plt.savefig(f"{dossier}/{col}_bar.png")
            plt.close()

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(f"{dossier}/statistiques_univariees.csv", index=False)
    
    return stats_df


# ============================================================
# ANALYSE BIVARIEE
# ============================================================

def detect_type(series):
    """Détecte si une variable est quantitative ou qualitative."""
    if pd.api.types.is_numeric_dtype(series):
        return "quantitative"
    else:
        return "qualitative"


def get_var_types(df):
    """Sépare les variables d'une dataframe en quantitatives et qualitatives."""
    quant = [col for col in df.columns if detect_type(df[col]) == "quantitative"]
    qual = [col for col in df.columns if detect_type(df[col]) == "qualitative"]
    return quant, qual


def cramers_v(x, y):
    """Calcule le V de Cramér (association entre deux variables qualitatives)."""
    confusion_matrix = pd.crosstab(x, y)

    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return np.nan

    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    denom = n * (min(k-1, r-1))

    if denom == 0:
        return np.nan

    return np.sqrt(chi2 / denom)


def analyse_qual_qual(df, var1, var2, folder):
    """Analyse la relation entre deux variables qualitatives (V de Cramér)."""
    val = cramers_v(df[var1], df[var2])
    ct = pd.crosstab(df[var1], df[var2])

    plt.figure(figsize=(8, 6))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{var1} vs {var2}")
    plt.tight_layout()

    path = f"{folder}/{var1}_{var2}.png"
    plt.savefig(path)
    plt.close()

    return {
        "var1": var1,
        "var2": var2,
        "type": "qual-qual",
        "metric": "cramers_v",
        "value": val
    }


def analyse_qual_quant(df, qual, quant, folder):
    """Analyse la relation entre une variable qualitative et une quantitative (ANOVA)."""
    groups = []

    for g in df[qual].dropna().unique():
        data = df[df[qual] == g][quant].dropna()
        if len(data) > 1:
            groups.append(data)

    if len(groups) < 2:
        return None

    stat, p = f_oneway(*groups)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[qual], y=df[quant])
    plt.title(f"{qual} vs {quant}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = f"{folder}/{qual}_{quant}.png"
    plt.savefig(path)
    plt.close()

    return {
        "var1": qual,
        "var2": quant,
        "type": "qual-quant",
        "metric": "anova_pvalue",
        "value": p
    }


def analyse_quant_quant(df, var1, var2, folder):
    """Analyse la relation entre deux variables quantitatives (Corrélation)."""
    # Remove NaN values
    data = df[[var1, var2]].dropna()

    if len(data) < 3:
        return None

    # Pearson correlation
    pearson_corr = data[var1].corr(data[var2], method='pearson')

    # Spearman correlation (rank-based)
    spearman_corr = data[var1].corr(data[var2], method='spearman')

    # Kendall correlation (rank-based)
    kendall_corr = data[var1].corr(data[var2], method='kendall')

    # P-value for Pearson using scipy
    _, p_value = pearsonr(data[var1], data[var2])

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data[var1], data[var2], alpha=0.6)

    # Add regression line
    z = np.polyfit(data[var1], data[var2], 1)
    p = np.poly1d(z)
    plt.plot(data[var1], p(data[var1]), "r--", alpha=0.8, linewidth=2)

    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(f"{var1} vs {var2} (r={pearson_corr:.3f}, p={p_value:.4f})")

    plt.tight_layout()

    path = f"{folder}/{var1}_{var2}.png"
    plt.savefig(path)
    plt.close()

    return {
        "var1": var1,
        "var2": var2,
        "type": "quant-quant",
        "metric": "pearson_corr",
        "value": pearson_corr,
        "p_value": p_value,
        "spearman_corr": spearman_corr,
        "kendall_corr": kendall_corr
    }


def analyse_table(df, folder):
    """Analyse automatique bivariée complète d'une table."""
    results = []
    types = {col: detect_type(df[col]) for col in df.columns}
    cols = list(df.columns)

    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            v1 = cols[i]
            v2 = cols[j]
            t1 = types[v1]
            t2 = types[v2]

            if t1 == "qualitative" and t2 == "qualitative":
                res = analyse_qual_qual(df, v1, v2, folder)
                if res:
                    results.append(res)

            elif t1 == "qualitative" and t2 == "quantitative":
                res = analyse_qual_quant(df, v1, v2, folder)
                if res:
                    results.append(res)

            elif t1 == "quantitative" and t2 == "qualitative":
                res = analyse_qual_quant(df, v2, v1, folder)
                if res:
                    results.append(res)
            
            elif t1 == "quantitative" and t2 == "quantitative":
                res = analyse_quant_quant(df, v1, v2, folder)
                if res:
                    results.append(res)

    return pd.DataFrame(results) if results else pd.DataFrame()


