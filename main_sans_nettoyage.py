"""
Pipeline d'analyse avant nettoyage — données brutes
1. Analyse univariée des tables brutes
2. Analyse bivariée des tables brutes
"""

import os
import pandas as pd
from lib_analyse import analyse_univariee, analyse_table, get_var_types

# ── Chemins ────────────────────────────────────────────────────────────────────

DATA_DIR    = "data"
OUTPUT_ROOT = "analyse_avant_nettoyage"

UNI_TABLE1  = os.path.join(OUTPUT_ROOT, "analyse_univariee", "table1")
UNI_TABLE2  = os.path.join(OUTPUT_ROOT, "analyse_univariee", "table2")
BIV_TABLE1  = os.path.join(OUTPUT_ROOT, "analyse_bivariee",  "table1")
BIV_TABLE2  = os.path.join(OUTPUT_ROOT, "analyse_bivariee",  "table2")
INTERESTING = os.path.join(OUTPUT_ROOT, "analyse_bivariee",  "graphiques_interessants.txt")

# ── Vérifications préliminaires ────────────────────────────────────────────────

print("Vérification des fichiers requis...")
assert os.path.exists(os.path.join(DATA_DIR, "table1.csv")), \
    "[ERREUR] data/table1.csv introuvable"
assert os.path.exists(os.path.join(DATA_DIR, "table2.csv")), \
    "[ERREUR] data/table2.csv introuvable"
assert os.path.exists("lib_analyse.py"), "[ERREUR] lib_analyse.py introuvable"
print("[OK] Tous les fichiers requis trouvés\n")

# ── Chargement ────────────────────────────────────────────────────────────────

print("Chargement des données brutes...")
table1 = pd.read_csv(os.path.join(DATA_DIR, "table1.csv"))
table2 = pd.read_csv(os.path.join(DATA_DIR, "table2.csv"))
print(f"[OK] table1 : {table1.shape[0]:,} lignes × {table1.shape[1]} colonnes")
print(f"[OK] table2 : {table2.shape[0]:,} lignes × {table2.shape[1]} colonnes\n")

# ── Détection des types de variables ──────────────────────────────────────────

quant_t1, qual_t1 = get_var_types(table1)
quant_t2, qual_t2 = get_var_types(table2)

print("=" * 70)
print("VARIABLES DÉTECTÉES")
print("=" * 70)
print(f"\nTable1  —  quantitatives ({len(quant_t1)}) : {quant_t1}")
print(f"         qualitatives  ({len(qual_t1)}) : {qual_t1}")
print(f"\nTable2  —  quantitatives ({len(quant_t2)}) : {quant_t2}")
print(f"         qualitatives  ({len(qual_t2)}) : {qual_t2}")


def _remove_variables(quant: list, qual: list, label: str) -> tuple[list, list]:
    """Propose interactivement de retirer des variables avant analyse."""
    print(f"\n{label} — Retirer des variables ? (o/n) : ", end="")
    if input().strip().lower() not in ("o", "oui", "y", "yes"):
        return quant, qual
    q, ql = quant.copy(), qual.copy()
    while True:
        print(f"  Quantitatives : {q}")
        print(f"  Qualitatives  : {ql}")
        print("  Variable à retirer (ou 'fin') : ", end="")
        var = input().strip()
        if var.lower() in ("fin", "done", "stop", ""):
            break
        if var in q:
            q.remove(var); print(f"  [OK] {var} retiré des quantitatives")
        elif var in ql:
            ql.remove(var); print(f"  [OK] {var} retiré des qualitatives")
        else:
            print(f"  [?] Variable '{var}' non trouvée")
    return q, ql


print("\n" + "=" * 70)
print("CONFIGURATION DES VARIABLES")
print("=" * 70)

quant_t1, qual_t1 = _remove_variables(quant_t1, qual_t1, "TABLE1")
quant_t2, qual_t2 = _remove_variables(quant_t2, qual_t2, "TABLE2")

# ID systématiquement exclu des graphiques
for lst in (quant_t1, qual_t1, quant_t2, qual_t2):
    if "ID" in lst:
        lst.remove("ID")

print(f"\n[OK] Table1 : {len(quant_t1)} quant., {len(qual_t1)} qual.")
print(f"[OK] Table2 : {len(quant_t2)} quant., {len(qual_t2)} qual.\n")

# ── Création des dossiers de sortie ───────────────────────────────────────────

for path in (UNI_TABLE1, UNI_TABLE2, BIV_TABLE1, BIV_TABLE2):
    os.makedirs(path, exist_ok=True)

with open(INTERESTING, "w", encoding="utf-8") as f:
    f.write("Graphiques bivariés à interpréter en priorité\n")
    f.write("=" * 60 + "\n")

# ── Phase 1 : Analyse univariée ───────────────────────────────────────────────

print("=" * 70)
print("PHASE 1 — ANALYSE UNIVARIÉE")
print("=" * 70 + "\n")

stats_t1 = analyse_univariee(table1, quant_t1, qual_t1, UNI_TABLE1)
print(f"[OK] Table1 : {len(stats_t1)} variables analysées\n")

stats_t2 = analyse_univariee(table2, quant_t2, qual_t2, UNI_TABLE2)
print(f"[OK] Table2 : {len(stats_t2)} variables analysées\n")

# ── Phase 2 : Analyse bivariée ────────────────────────────────────────────────

print("=" * 70)
print("PHASE 2 — ANALYSE BIVARIÉE")
print("=" * 70 + "\n")


def _progress(current, total, v1, v2):
    print(f"  [{current:>4}/{total}] {v1} vs {v2}")


res_t1 = analyse_table(
    table1, BIV_TABLE1,
    progress_callback=_progress,
    interesting_file=INTERESTING,
    excluded_columns=["ID"],
    generate_bivariate_plots_only_if_significant=True,
    generate_quant_plots_only_if_correlated=True,
)
res_t1.to_csv(os.path.join(OUTPUT_ROOT, "analyse_bivariee",
                            "resultats_table1.csv"), index=False)
print(f"\n[OK] Table1 : {len(res_t1)} paires analysées\n")

res_t2 = analyse_table(
    table2, BIV_TABLE2,
    progress_callback=_progress,
    interesting_file=INTERESTING,
    excluded_columns=["ID"],
    generate_bivariate_plots_only_if_significant=True,
    generate_quant_plots_only_if_correlated=True,
)
res_t2.to_csv(os.path.join(OUTPUT_ROOT, "analyse_bivariee",
                            "resultats_table2.csv"), index=False)
print(f"\n[OK] Table2 : {len(res_t2)} paires analysées\n")

# ── Fin ────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("[OK] ANALYSES TERMINÉES")
print(f"     Résultats dans : {OUTPUT_ROOT}/")
print(f"     Priorités      : {INTERESTING}")
print("=" * 70)
