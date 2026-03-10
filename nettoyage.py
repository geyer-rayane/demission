import pandas as pd
import numpy as np


# 1. Charger les données


table1 = pd.read_csv("data/table1.csv", sep=",")
table2 = pd.read_csv("data/table2.csv", sep=",")

print("Table1 shape :", table1.shape)
print("Table2 shape :", table2.shape)



# 2. Corriger les valeurs invalides


table2["DTNAIS"] = table2["DTNAIS"].replace("0000-00-00", np.nan)

table1["DTDEM"] = table1["DTDEM"].replace("31/12/1900", np.nan)
table2["DTDEM"] = table2["DTDEM"].replace("31/12/1900", np.nan)



# 3. Convertir les dates


date_columns_table1 = ["DTADH", "DTDEM"]
date_columns_table2 = ["DTADH", "DTDEM", "DTNAIS"]

for col in date_columns_table1:
    table1[col] = pd.to_datetime(table1[col], errors="coerce", dayfirst=True)

for col in date_columns_table2:
    table2[col] = pd.to_datetime(table2[col], errors="coerce", dayfirst=True)



# 4. Supprimer variables redondantes


cols_to_drop = [
    "RANGAGEAD",
    "RANGAGEDEM",
    "RANGADH","ANNEEDEM","RANGDEM","AGEDEM","ADH"
]

table1 = table1.drop(columns=[c for c in cols_to_drop if c in table1.columns])
table2 = table2.drop(columns=[c for c in cols_to_drop if c in table2.columns])



# 5. Gérer les valeurs manquantes


# Variables catégorielles → remplacer par "Unknown"
categorical_cols = ["CDMOTDEM"]

for col in categorical_cols:
    if col in table1.columns:
        table1[col] = table1[col].fillna("Unknown")

    if col in table2.columns:
        table2[col] = table2[col].fillna("Unknown")


# Variables numériques → remplacer par une valeur aléatoire sur la colonne
numeric_cols_table1 = table1.select_dtypes(include=np.number).columns
numeric_cols_table2 = table2.select_dtypes(include=np.number).columns

for col in numeric_cols_table1:
    table1[col] = table1[col].apply(
        lambda x: table1[col].dropna().sample(1).iloc[0] if pd.isna(x) else x
    )

for col in numeric_cols_table2:
    table2[col] = table2[col].apply(
        lambda x: table2[col].dropna().sample(1).iloc[0] if pd.isna(x) else x
    )



# 6. Créer la variable cible (ISDEM)

# table2 : démission si DTDEM existe
table2["ISDEM"] = table2["DTDEM"].apply(lambda x: 0 if pd.isna(x) else 1)




print("table1 shape :", table1.shape)
print("table2 shape :", table2.shape)




# 7. Vérifier que DTADH < DTDEM sur table2
invalid_dates = table2[table2["DTADH"] >= table2["DTDEM"]].copy()
print(f"\nNombre de lignes où DTADH >= DTDEM : {len(invalid_dates)}")
if len(invalid_dates) > 0:
    print(invalid_dates[["DTADH", "DTDEM", "ISDEM"]])


# 9. Sauvegarder dataset propre
table1.to_csv("table1_clean.csv", index=False)
table2.to_csv("table2_clean.csv", index=False)