import pandas as pd
import numpy as np

table1 = pd.read_csv("table1.csv")
table2 = pd.read_csv("table2.csv")


print("Colonnes table1")
print(table1.columns)

print("\nColonnes table2")
print(table2.columns)


table2["DTDEM"] = pd.to_datetime(table2["DTDEM"], format="%d/%m/%Y", errors="coerce")
table2["DTADH"] = pd.to_datetime(table2["DTADH"], format="%d/%m/%Y", errors="coerce")
table2["DTNAIS"] = pd.to_datetime(table2["DTNAIS"], errors="coerce")

table2["CDDEM"] = np.nan
table2["ANNEEDEM"] = table2["DTDEM"].dt.year
table2["AGEAD"] = (table2["DTADH"] - table2["DTNAIS"]).dt.days / 365
table2["AGEDEM"] = (table2["DTDEM"] - table2["DTNAIS"]).dt.days / 365
table2["ADH"] = (table2["DTDEM"] - table2["DTADH"]).dt.days / 365

table2["RANGAGEAD"] = np.nan
table2["RANGAGEDEM"] = np.nan
table2["RANGDEM"] = np.nan
table2["RANGADH"] = np.nan



table2_alignee = table2[[
    "ID",
    "CDSEXE",
    "MTREV",
    "NBENF",
    "CDSITFAM",
    "DTADH",
    "CDTMT",
    "CDDEM",
    "DTDEM",
    "ANNEEDEM",
    "CDMOTDEM",
    "CDCATCL",
    "AGEAD",
    "RANGAGEAD",
    "AGEDEM",
    "RANGAGEDEM",
    "RANGDEM",
    "ADH",
    "RANGADH"
]]

data_final = pd.concat([table1, table2_alignee], ignore_index=True)

print("Dimensions finale :", data_final.shape)

print("Apercu")
print(data_final.head())

print(" valeurs manquantes")
print(data_final.isnull().sum())


data_final.to_csv("table_finale.csv", index=False)

print("\ntable finale sauvegardée : table_finale.csv")