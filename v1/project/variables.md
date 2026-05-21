# Variables du jeu de données — Explication simplifiée

## Identifiants (qui est cette ligne budgétaire ?)

| Variable | Explication |
|---|---|
| `year` | L'année budgétaire (2020 à 2025) |
| `quarter` | Le trimestre : 1 = janvier-mars, 2 = avril-juin, 3 = juillet-sept, 4 = octobre-déc |
| `programme` | Numéro du programme budgétaire (301, 302 ou 303) |
| `programme_label` | Nom du programme (ex : "Accès à la Justice") |
| `region` | Code à 2 chiffres de la région (00 = services centraux, 01 à 12 = régions) |
| `region_label` | Nom de la région (ex : "CASABLANCA-SETTAT") |
| `projet` | Numéro du projet au sein du programme |
| `projet_label` | Nom du projet (ex : "Construction et réhabilitation") |
| `ligne` | Numéro de la ligne budgétaire au sein du projet |
| `ligne_label` | Nom de la ligne (ex : "Travaux de construction de tribunaux") |
| `type_ligne` | Catégorie de la dépense : `travaux`, `equipements`, `etudes`, `acquisitions`, `fournitures` |
| `line_id` | Identifiant unique de la ligne, format : P301-R04-Pj10-L10 |

---

## Montants et taux (combien ? à quel niveau ?)

| Variable | Explication |
|---|---|
| `lf_mdh` | Budget alloué à cette ligne cette année-là, en **millions de dirhams (MDH)**. Vient de la Loi de Finances. |
| `taux` | **Taux d'exécution** : part du budget réellement dépensée à ce trimestre. 0.15 = 15% consommé. C'est la valeur qu'on cherche à prédire. |

---

## Variables calculées (utilisées par le modèle ML)

| Variable | Explication |
|---|---|
| `lf_ratio` | Budget de cette ligne cette année ÷ sa moyenne sur toutes les années. Indique si la ligne a eu plus ou moins de budget que d'habitude. 1.2 = 20% de plus que la normale. |
| `lf_share` | Budget de cette ligne ÷ budget total annuel de toutes les lignes. Indique le poids relatif de cette ligne dans l'enveloppe globale. |

---

## Résumé rapide

```
data_lines.csv
├── 2616 lignes
├── 109 lignes budgétaires distinctes
│     ├── 13 lignes centrales  (région 00 — SERVICES COMMUNS)
│     └── 96 lignes régionales (8 types × 12 régions)
└── 6 années × 4 trimestres par ligne
```

**Données simulées — calibrées sur les agrégats réels TGR 2020–2025.**  
La somme pondérée des taux par trimestre correspond aux chiffres officiels publiés.
