import pandas as pd

# Charger le dataset
df = pd.read_parquet("data/uscrn_clean.parquet")

# 1. Afficher un aperçu des stations
print("--- Stations (10 premières) ---")
print(df["station"].unique()[:10])

# 2. Vérifier les valeurs du flag d'incohérence verticale
print("\n--- Flag d'incohérence verticale ---")
print(df["flag_incoherence_verticale"].value_counts(dropna=False))

# 3. Vérifier les valeurs manquantes sur les colonnes clés
print("\n--- Valeurs manquantes (moyenne) ---")
print(df[["time", "SOIL_MOISTURE_5", "P_CALC", "SOIL_TEMP_5"]].isna().mean())

# 4. Vérifier l'intervalle de temps pour une station
s = df[df.station == df.station.iloc[0]].sort_values("time")
print("\n--- Intervalles de temps (doit être 1 heure en principe) ---")
print(s["time"].diff().value_counts().head(3))

# 5. Statistiques sur les variations de l'humidité du sol
print("\n--- Variations de l'humidité du sol (SOIL_MOISTURE_5) ---")
print(s["SOIL_MOISTURE_5"].diff().abs().describe())