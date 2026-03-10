import pandas as pd

# Charger le fichier de stats
stats = pd.read_csv("analyse_univariee/table1/statistiques_univariees.csv")

# Séparer les types
stats_quant = stats[stats["type"] == "quantitative"].copy()
stats_qual = stats[stats["type"] == "qualitative"].copy()

# Arrondir les valeurs numériques
stats_quant = stats_quant.round({
    "mean":2,
    "median":2,
    "std":2,
    "variance":2,
    "min":2,
    "max":2,
    "IQR":2
})

# Garder seulement les colonnes utiles
stats_quant = stats_quant[[
    "variable","mean","median","std","variance","min","max","IQR"
]]

stats_qual = stats_qual[[
    "variable","mode"
]]

# Affichage
print("\n===== VARIABLES QUANTITATIVES =====\n")
print(stats_quant.to_string(index=False))

print("\n===== VARIABLES QUALITATIVES =====\n")
print(stats_qual.to_string(index=False))