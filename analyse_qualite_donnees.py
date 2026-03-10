import pandas as pd
import numpy as np

# 1 Charger les données

table1 = pd.read_csv("data/table1.csv")
table2 = pd.read_csv("data/table2.csv")

print("Table1 shape:", table1.shape)
print("Table2 shape:", table2.shape)

print("\nColonnes table1:")
print(table1.columns)

print("\nColonnes table2:")
print(table2.columns)


# 2 Aperçu des données

print("\nAperçu table1:")
print(table1.head())

print("\nAperçu table2:")
print(table2.head())


# 3 Informations générales

print("\nInfo table1")
print(table1.info())

print("\nInfo table2")
print(table2.info())


# 4 Valeurs manquantes

print("\nValeurs manquantes table1:")
print(table1.isnull().sum())

print("\nValeurs manquantes table2:")
print(table2.isnull().sum())


# 5 Valeurs uniques (catégorielles)

print("\nValeurs uniques par colonne (table1)")
for col in table1.columns:
    print(f"\n{col}")
    print(table1[col].unique()[:20])  # limite affichage


print("\nValeurs uniques par colonne (table2)")
for col in table2.columns:
    print(f"\n{col}")
    print(table2[col].unique()[:20])


# 6 Statistiques descriptives

print("\nStatistiques table1:")
print(table1.describe(include='all'))

print("\nStatistiques table2:")
print(table2.describe(include='all'))


# 7 Détection de valeurs suspectes

print("\nValeurs suspectes possibles")

# dates suspectes
print("\nDates DTNAIS = 0000-00-00")
if "DTNAIS" in table2.columns:
    print((table2["DTNAIS"] == "0000-00-00").sum())

print("\nDates DTDEM = 31/12/1900 (clients actifs)")
if "DTDEM" in table2.columns:
    print((table2["DTDEM"] == "31/12/1900").sum())


# 8 Vérification valeurs numériques

numeric_cols1 = table1.select_dtypes(include=np.number).columns
numeric_cols2 = table2.select_dtypes(include=np.number).columns

print("\nColonnes numériques table1:", numeric_cols1)
print("\nColonnes numériques table2:", numeric_cols2)

print("\nValeurs min/max table1")
for col in numeric_cols1:
    print(col, "min:", table1[col].min(), "max:", table1[col].max())

print("\nValeurs min/max table2")
for col in numeric_cols2:
    print(col, "min:", table2[col].min(), "max:", table2[col].max())


# 9 Détection valeurs aberrantes simples

print("\nVérification valeurs négatives")

for col in numeric_cols1:
    if (table1[col] < 0).any():
        print("Valeurs négatives dans:", col)

for col in numeric_cols2:
    if (table2[col] < 0).any():
        print("Valeurs négatives dans:", col)


# 10 Vérification doublons

print("\nDoublons table1:", table1.duplicated().sum())
print("Doublons table2:", table2.duplicated().sum())


print("\nAnalyse terminée")