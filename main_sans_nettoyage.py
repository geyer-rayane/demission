"""
Main - Orchestrer toutes les analyses statistiques du projet BI
"""

import pandas as pd
import os
import subprocess
from lib_analyse import analyse_univariee, analyse_table, detect_type, get_var_types

# VÉRIFICATION DES CHEMINS

print("Vérification des chemins...")
assert os.path.exists("data/table1.csv"), "✗ data/table1.csv manquant"
assert os.path.exists("data/table2.csv"), "✗ data/table2.csv manquant"
assert os.path.exists("nettoyage.py"), "✗ nettoyage.py manquant"
assert os.path.exists("lib_analyse.py"), "✗ lib_analyse.py manquant"
print("✓ Tous les fichiers requis trouvés\n")

# CHARGEMENT

print("Chargement des données (non nettoyées)..")
table1 = pd.read_csv("data/table1.csv")
table2 = pd.read_csv("data/table2.csv")
print(f"✓ Table1: {table1.shape}")
print(f"✓ Table2: {table2.shape}\n")

# CONFIGURATION DES VARIABLES

quant_table1, qual_table1 = get_var_types(table1)
quant_table2, qual_table2 = get_var_types(table2)

print("=" * 70)
print("VARIABLES DÉTECTÉES")
print("=" * 70)
print(f"\nTable1:")
print(f"  Quantitatives ({len(quant_table1)}): {quant_table1}")
print(f"  Qualitatives ({len(qual_table1)}): {qual_table1}")
print(f"\nTable2:")
print(f"  Quantitatives ({len(quant_table2)}): {quant_table2}")
print(f"  Qualitatives ({len(qual_table2)}): {qual_table2}")

# Fonction pour proposer retrait de variables
def remove_variables(quant, qual, table_name):
    all_vars = {"quant": quant.copy(), "qual": qual.copy()}
    
    print(f"\n{table_name} - Retirer des variables? (y/n): ", end="")
    if input().lower() != "y":
        return all_vars["quant"], all_vars["qual"]
    
    while True:
        print(f"\nVariables actuelles:")
        print(f"  Quantitatives: {all_vars['quant']}")
        print(f"  Qualitatives: {all_vars['qual']}")
        print(f"\nEntrez le nom de la variable à retirer (ou 'done' pour continuer): ", end="")
        var = input().strip()
        
        if var.lower() == "done":
            break
        
        if var in all_vars["quant"]:
            all_vars["quant"].remove(var)
            print(f"✓ {var} retiré des quantitatives")
        elif var in all_vars["qual"]:
            all_vars["qual"].remove(var)
            print(f"✓ {var} retiré des qualitatives")
        else:
            print(f"✗ Variable {var} non trouvée")
    
    return all_vars["quant"], all_vars["qual"]

print("\n" + "=" * 70)
print("CONFIGURATION DES VARIABLES")
print("=" * 70)

quant_table1, qual_table1 = remove_variables(quant_table1, qual_table1, "TABLE1")
quant_table2, qual_table2 = remove_variables(quant_table2, qual_table2, "TABLE2")

print(f"\n✓ Configuration finalisée:")
print(f"  Table1 - Quant: {len(quant_table1)}, Qual: {len(qual_table1)}")
print(f"  Table2 - Quant: {len(quant_table2)}, Qual: {len(qual_table2)}\n")



# CRÉATION DES DOSSIERS

os.makedirs("analyse_avant_nettoyage/analyse_univariee/table1", exist_ok=True)
os.makedirs("analyse_avant_nettoyage/analyse_univariee/table2", exist_ok=True)
os.makedirs("analyse_avant_nettoyage/analyse_bivariee/table1", exist_ok=True)
os.makedirs("analyse_avant_nettoyage/analyse_bivariee/table2", exist_ok=True)

# PHASE 1: ANALYSE UNIVARIÉE

print("=" * 70)
print("PHASE 1: ANALYSE UNIVARIÉE")
print("=" * 70 + "\n")

stats_table1 = analyse_univariee(table1, quant_table1, qual_table1, "analyse_avant_nettoyage/analyse_univariee/table1")
print(f"Table1: {len(stats_table1)} variables analysées\n")

stats_table2 = analyse_univariee(table2, quant_table2, qual_table2, "analyse_avant_nettoyage/analyse_univariee/table2")
print(f"Table2: {len(stats_table2)} variables analysées\n")

# PHASE 2: ANALYSE BIVARIÉE

print("=" * 70)
print("PHASE 2: ANALYSE BIVARIÉE")
print("=" * 70 + "\n")

resultats_table1 = analyse_table(table1, "analyse_avant_nettoyage/analyse_bivariee/table1")
resultats_table1.to_csv("analyse_avant_nettoyage/analyse_bivariee/resultats_table1.csv", index=False)
print(f"Table1: {len(resultats_table1)} associations\n")

resultats_table2 = analyse_table(table2, "analyse_avant_nettoyage/analyse_bivariee/table2")
resultats_table2.to_csv("analyse_avant_nettoyage/analyse_bivariee/resultats_table2.csv", index=False)
print(f"Table2: {len(resultats_table2)} associations\n")

print("=" * 70)
print("✓ ANALYSES TERMINÉES")
print("=" * 70)
