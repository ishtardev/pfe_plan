"""Visualisations des anomalies budgétaires.

Outputs dans 04_visuals/ :
  1. heatmap_zscore.png          -- Z-score de toutes les lignes × années
  2. isolation_forest_top11/     -- Graphe de taux par ligne anormale (IF)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

DATA_DIR  = Path(__file__).resolve().parent.parent
ANOM_FILE = DATA_DIR / "03_extracted" / "SituationChap_ANOMALIES.xlsx"
OUT_DIR   = DATA_DIR / "04_visuals"
IF_DIR    = OUT_DIR / "isolation_forest_top11"
OUT_DIR.mkdir(exist_ok=True)
IF_DIR.mkdir(exist_ok=True)

ZSCORE_THRESHOLD = 1.5


# ------------------------------------------------------------------ #
#  1. Heatmap Z-score                                                 #
# ------------------------------------------------------------------ #

def plot_heatmap(zscore_df: pd.DataFrame):
    pivot = zscore_df.pivot_table(
        index="Intitule", columns="Year", values="Z_Score", aggfunc="first"
    )
    pivot = pivot.sort_index()
    years = sorted(pivot.columns)

    fig, ax = plt.subplots(figsize=(len(years) * 1.4, max(12, len(pivot) * 0.32)))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-3, vmax=3,
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7},
        cbar_kws={"label": "Z-score", "shrink": 0.6},
    )

    # Encadrer les cellules anormales
    for i, intitule in enumerate(pivot.index):
        for j, year in enumerate(years):
            z = pivot.loc[intitule, year]
            if not np.isnan(z) and abs(z) > ZSCORE_THRESHOLD:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j, i), 1, 1,
                        fill=False, edgecolor="black", lw=1.5
                    )
                )

    ax.set_title(
        f"Z-score du taux d'engagement par ligne et par année\n"
        f"(encadré = |Z| > {ZSCORE_THRESHOLD})",
        fontsize=13, pad=14
    )
    ax.set_xlabel("Année", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out = OUT_DIR / "heatmap_zscore.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ------------------------------------------------------------------ #
#  2. Graphes individuels — lignes Isolation Forest                   #
# ------------------------------------------------------------------ #

def plot_if_lines(zscore_df: pd.DataFrame, if_df: pd.DataFrame):
    # Prendre les lignes flaggées ET le top selon le rang
    anomalies = if_df[if_df["Est_Anomalie_IF"] == "Oui"].copy()
    if len(anomalies) == 0:
        # fallback : top 11 par score
        anomalies = if_df.nsmallest(11, "Score_Anomalie")

    taux_cols = sorted([c for c in if_df.columns if c.startswith("Taux_")])
    years     = [int(c.replace("Taux_", "")) for c in taux_cols]

    for _, row in anomalies.iterrows():
        key       = row["Line_Key"]
        intitule  = row["Intitule"]
        rang      = int(row["Rang_Anomalie"])
        score     = round(row["Score_Anomalie"], 4)

        taux_vals = [row[c] for c in taux_cols]

        # Z-scores pour cette ligne
        line_z = zscore_df[zscore_df["Line_Key"] == key].set_index("Year")

        fig, ax = plt.subplots(figsize=(8, 4))

        # Fond coloré pour les années anormales (Z-score)
        for y in years:
            if y in line_z.index:
                z = line_z.loc[y, "Z_Score"]
                if not np.isnan(z) and abs(z) > ZSCORE_THRESHOLD:
                    ax.axvspan(y - 0.4, y + 0.4, color="salmon", alpha=0.3, label="_nolegend_")

        ax.plot(years, taux_vals, marker="o", linewidth=2,
                color="#2563EB", markersize=7, label="Taux d'engagement")

        # Points anomalies Z-score en rouge
        for i, y in enumerate(years):
            if y in line_z.index:
                z = line_z.loc[y, "Z_Score"]
                if not np.isnan(z) and abs(z) > ZSCORE_THRESHOLD:
                    ax.plot(y, taux_vals[i], "ro", markersize=10, zorder=5,
                            label=f"Anomalie Z ({y})")

        ax.set_title(
            f"{intitule}\nIF Rang #{rang}  |  Score: {score}",
            fontsize=10, pad=10
        )
        ax.set_xlabel("Année")
        ax.set_ylabel("Taux d'engagement")
        ax.set_xticks(years)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=8)

        plt.tight_layout()
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in intitule)[:60]
        out = IF_DIR / f"rang{rang:02d}_{safe_name}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out.name}")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    print("Chargement des données d'anomalies…")
    zscore_df = pd.read_excel(ANOM_FILE, sheet_name="Anomalies_Annuelles")
    if_df     = pd.read_excel(ANOM_FILE, sheet_name="Lignes_Anormales")

    print("\n1. Heatmap Z-score…")
    plot_heatmap(zscore_df)

    print("\n2. Graphes Isolation Forest…")
    plot_if_lines(zscore_df, if_df)

    print(f"\nTous les visuels sont dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
