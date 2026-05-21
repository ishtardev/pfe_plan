"""Détection d'anomalies sur les lignes budgétaires.

Deux approches complémentaires :

1. Z-score par ligne (anomalie temporelle)
   Pour chaque ligne, on calcule la moyenne et l'écart-type du taux d'engagement
   sur toutes les années. Une année est flaggée si |z| > 2.
   → Détecte les chocs ponctuels sur une ligne (ex: chute de taux en 2023).

2. Isolation Forest (anomalie structurelle)
   On représente chaque ligne par son vecteur de taux [t_2021, …, t_2025].
   L'algorithme détecte les lignes dont le profil global est statistiquement
   différent des autres.
   → Détecte les lignes structurellement atypiques.

Output : SituationChap_ANOMALIES.xlsx
  Sheet "Anomalies_Annuelles"  -- Z-score, une ligne par (budget_line × année)
  Sheet "Lignes_Anormales"     -- Isolation Forest, une ligne par budget_line
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent
SRC      = DATA_DIR / "03_extracted" / "SituationChap_LIGNES_BUDGETAIRES_COMMUNS.xlsx"
OUT_DIR  = DATA_DIR / "03_extracted"

ID_COLS = ["Chap", "Chap_Intitule", "Prog", "Prog_Intitule",
           "Reg",  "Reg_Intitule",  "Proj", "Proj_Intitule",
           "Lb",   "Intitule"]


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def compute_taux(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne Taux_Engagement = Total_Engage / Credits_Ouverts."""
    df = df.copy()
    mask = (df["Credits_Ouverts_Vises"] != 0) & df["Credits_Ouverts_Vises"].notna()
    df["Taux_Engagement"] = np.nan
    df.loc[mask, "Taux_Engagement"] = (
        df.loc[mask, "Total_Engage_Vises"] / df.loc[mask, "Credits_Ouverts_Vises"]
    )
    return df


# ------------------------------------------------------------------ #
#  1. Z-score par ligne — anomalies temporelles                       #
# ------------------------------------------------------------------ #

def zscore_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby("Line_Key"):
        meta = {c: grp[c].iloc[0] for c in ID_COLS}
        meta["Line_Key"] = key

        taux_series = grp.set_index("Year")["Taux_Engagement"]
        valid = taux_series.dropna()
        if len(valid) < 2:
            continue

        mean_t = valid.mean()
        std_t  = valid.std(ddof=1)

        for year, taux in taux_series.items():
            z = (taux - mean_t) / std_t if (not np.isnan(taux) and std_t > 0) else np.nan
            rows.append({
                **meta,
                "Year":             year,
                "Taux_Engagement":  round(taux, 4) if not np.isnan(taux) else np.nan,
                "Taux_Moyen":       round(mean_t, 4),
                "Taux_EcartType":   round(std_t, 4),
                "Z_Score":          round(z, 2) if not np.isnan(z) else np.nan,
                "Anomalie_Annee":   "Oui" if (not np.isnan(z) and abs(z) > 1.5) else "Non",
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  2. Isolation Forest — anomalies structurelles                      #
# ------------------------------------------------------------------ #

def isolation_forest_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df["Year"].unique())

    # Pivot : une ligne par budget_line, une colonne par année
    pivot = df.pivot_table(index="Line_Key", columns="Year",
                           values="Taux_Engagement", aggfunc="first")
    pivot = pivot.reindex(columns=years)

    # Imputation des NaN par la médiane de la colonne (pour que IF puisse tourner)
    pivot_filled = pivot.copy()
    for col in pivot_filled.columns:
        median_val = pivot_filled[col].median()
        pivot_filled[col] = pivot_filled[col].fillna(median_val)

    # Isolation Forest
    clf = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    clf.fit(pivot_filled.values)

    scores     = clf.decision_function(pivot_filled.values)   # plus négatif = plus anormal
    predictions = clf.predict(pivot_filled.values)             # -1 = anomalie, 1 = normal

    # Rang d'anomalie (1 = le plus anormal)
    rank = pd.Series(scores).rank(method="min").astype(int)

    meta_df = (df.drop_duplicates("Line_Key")
                 .set_index("Line_Key")[ID_COLS]
                 .reindex(pivot.index))

    result = meta_df.copy()
    result["Score_Anomalie"]    = scores.round(4)
    result["Est_Anomalie_IF"]   = ["Oui" if p == -1 else "Non" for p in predictions]
    result["Rang_Anomalie"]     = rank.values   # 1 = plus anormal parmi les 109 lignes

    # Ajouter les taux historiques pour contexte
    for y in years:
        result[f"Taux_{y}"] = pivot[y].values.round(4)

    return result.reset_index()


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    df = pd.read_excel(SRC)
    df["Line_Key"] = (df["Chap"].astype(str) + "-" + df["Prog"].astype(str) + "-" +
                      df["Reg"].astype(str)  + "-" + df["Proj"].astype(str) + "-" +
                      df["Lb"].astype(str))
    df = compute_taux(df)

    # -- Z-score --
    print("Calcul Z-score par ligne…")
    zscore_df = zscore_anomalies(df)
    n_anom = (zscore_df["Anomalie_Annee"] == "Oui").sum()
    print(f"  {n_anom} anomalies temporelles détectées sur {len(zscore_df)} observations")

    # -- Isolation Forest --
    print("Calcul Isolation Forest…")
    if_df = isolation_forest_anomalies(df)
    n_if = (if_df["Est_Anomalie_IF"] == "Oui").sum()
    print(f"  {n_if} lignes structurellement anormales détectées sur {len(if_df)} lignes")
    print("\n  Top 10 lignes les plus anormales :")
    top10 = if_df.nsmallest(10, "Score_Anomalie")[["Intitule", "Score_Anomalie", "Rang_Anomalie"] +
                                                    [c for c in if_df.columns if c.startswith("Taux_")]]
    print(top10.to_string(index=False))

    # -- Export --
    out_path = OUT_DIR / "SituationChap_ANOMALIES.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        zscore_df.to_excel(writer, sheet_name="Anomalies_Annuelles", index=False)
        if_df.to_excel(writer, sheet_name="Lignes_Anormales",        index=False)
    print(f"\nRésultats -> {out_path}")


if __name__ == "__main__":
    main()
