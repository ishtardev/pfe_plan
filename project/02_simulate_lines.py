"""
Simulation de données budgétaires au niveau ligne
Ministère de la Justice — Dépenses d'investissement 2020-2025
Structure : Programme → Région → Projet → Ligne

Contrainte : la somme pondérée des taux par ligne = agrégat TGR réel par trimestre
Output     : data_lines.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ── Agrégats TGR réels (investment, from data_long.csv) ──────────────────────
TGR = {
    2020: {"lf": 270.49, 1: 0.000, 2: 0.160, 3: 0.277, 4: 0.708},
    2021: {"lf": 200.50, 1: 0.009, 2: 0.116, 3: 0.262, 4: 0.763},
    2022: {"lf": 198.60, 1: 0.026, 2: 0.302, 3: 0.579, 4: 0.932},
    2023: {"lf": 230.60, 1: 0.022, 2: 0.151, 3: 0.215, 4: 0.787},
    2024: {"lf": 449.60, 1: 0.008, 2: 0.067, 3: 0.505, 4: 0.923},
    2025: {"lf": 420.60, 1: 0.009, 2: 0.082, 3: 0.597, 4: 1.065},
}

# ── Régions réelles (source : GeneralBudgetClassifier.py) ──────────────────
# Code 00 = services centraux ; codes 03/04/06/07 = 4 régions représentatives
REGIONAL_ZONES = [
    ("01", "TANGER-TÉTOUAN-AL HOCEÏMA"),
    ("02", "L'ORIENTAL"),
    ("03", "FÈS-MEKNÈS"),
    ("04", "RABAT-SALÉ-KÉNITRA"),
    ("05", "BÉNI MELLAL-KHÉNIFRA"),
    ("06", "CASABLANCA-SETTAT"),
    ("07", "MARRAKECH-SAFI"),
    ("08", "DRÂA-TAFILALET"),
    ("09", "SOUSS-MASSA"),
    ("10", "GUELMIM-OUED NOUN"),
    ("11", "LAÂYOUNE-SAKIA AL HAMRA"),
    ("12", "DAKHLA-OUED ED-DAHAB"),
]

# ── Structure budgétaire : Programme → Région → Projet → Ligne ───────────────
# Source : Morasse budgétaire Ministère de la Justice — Budget d'investissement
# Part relatives calibrées sur les crédits de paiement 2026 :
#   P300 (174.6 MDH) ≈ 46 %  |  P301 (150.3 MDH) ≈ 40 %
#   P302  (23.5 MDH) ≈  6 %  |  P303  (28.0 MDH) ≈  7 %
# Régions centrales → code "00" / "SERVICES COMMUNS"
# Régions de terrain → code "RG" (marqueur) → expansé en REGIONAL_ZONES
LINES_RAW = [
    # ── Programme 300 : Soutien et Pilotage (46 %) ───────────────────────────
    # Projet 10 : Infrastructure (89.6 MDH)
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "10","Infrastructure",
     "11","Extension et équipement du siège du Ministère",  "travaux",     0.050),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "10","Infrastructure",
     "12","Construction et équipement des cours d'appel",   "travaux",     0.095),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "10","Infrastructure",
     "13","Construction et équipement cour de cassation",   "travaux",     0.050),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "10","Infrastructure",
     "15","Construction et équipement tribunaux 1ère inst.", "travaux",     0.105),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "10","Infrastructure",
     "16","Construction Centres de Juges Résidents",        "travaux",     0.040),
    # Projet 20 : Gestion des ressources (84.6 MDH)
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "20","Gestion des ressources",
     "21","Guichets d'information judiciaire et de plaintes","equipements", 0.080),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "20","Gestion des ressources",
     "22","Dotation des bibliothèques",                     "equipements", 0.030),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "20","Gestion des ressources",
     "23","Atelier corps de métiers et garage parc auto",   "equipements", 0.040),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "20","Gestion des ressources",
     "24","Formation",                                      "etudes",      0.020),
    # Projet 40 : Gestion des archives (0.5 MDH)
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "40","Gestion des archives",
     "40","Sécurisation des bâtiments administratifs",      "travaux",     0.010),
    # Projet 50 : Efficacité Énergétique
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "50","Efficacité Énergétique",
     "50","Efficacité Énergétique dans les bâtiments publics","etudes",    0.008),

    # ── Programme 301 : Performance de l'Administration Judiciaire (40 %) ─────
    # Projet 10 : Performance des tribunaux en matière civile (36.3 MDH)
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "10","Performance tribunaux — matière civile",
     "11","Amélioration qualité et délais décisions civiles","etudes",     0.020),
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "10","Performance tribunaux — matière civile",
     "12","Renforcement des tribunaux de commerce",         "equipements", 0.025),
    # Projet 20 : Performance des tribunaux en matière pénale (74.0 MDH)
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "20","Performance tribunaux — matière pénale",
     "21","Amélioration qualité et délais décisions pénales","etudes",     0.020),
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "20","Performance tribunaux — matière pénale",
     "22","Construction et équipement tribunaux pénaux",    "travaux",     0.060),
    # Projet 30 : Accès au droit et à la justice (40.0 MDH)
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "30","Accès au droit et à la justice",
     "30","Développement de l'accès à la justice de proximité","etudes",  0.050),
    # Projet 40 : Projet Mahkamati (Banque Mondiale)
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "40","Projet Mahkamati (Banque Mondiale)",
     "41","Amélioration de la performance des juridictions","equipements", 0.025),
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "40","Projet Mahkamati (Banque Mondiale)",
     "42","Mise à niveau planification stratégique",        "etudes",      0.010),
    # Projet 50 : Projet Adl (Maroc-Espagne)
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "50","Projet Adl (coopération Maroc-Espagnole)",
     "50","Renforcement de la justice de proximité",        "equipements", 0.010),
    # Lignes régionales — déploiement physique des juridictions
    (301,"Performance Admin. Judiciaire","RG","RG",
     "10","Construction et réhabilitation des juridictions",
     "10","Construction et équipement des juridictions rég.","travaux",    0.095),
    (301,"Performance Admin. Judiciaire","RG","RG",
     "10","Construction et réhabilitation des juridictions",
     "20","Réhabilitation des juridictions régionales",     "travaux",     0.050),
    (301,"Performance Admin. Judiciaire","RG","RG",
     "20","Équipements des juridictions locales",
     "10","Équipements des juridictions régionales",        "equipements", 0.020),

    # ── Programme 302 : Modernisation du Système Judiciaire et Juridique (6 %)
    # Projet 10 : Performance législative et organisationnelle (14.5 MDH)
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "10","Performance législative et organisationnelle",
     "10","Tribunal numérique",                             "etudes",      0.010),
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "10","Performance législative et organisationnelle",
     "11","Informatique",                                   "equipements", 0.010),
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "10","Performance législative et organisationnelle",
     "12","Informatisation des juridictions",               "equipements", 0.012),
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "10","Performance législative et organisationnelle",
     "13","Base de données juridiques et judiciaires",      "etudes",      0.008),
    # Projet 20 : Tribunal numérique (9.0 MDH)
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "20","Tribunal numérique",
     "22","Construction et équipement centres d'archivage", "equipements", 0.008),
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "20","Tribunal numérique",
     "23","Archivage électronique",                        "etudes",      0.006),
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "20","Tribunal numérique",
     "24","Formation",                                      "etudes",      0.005),
    # Projet 30 : Programme MEDA
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "30","Programme MEDA",
     "20","Modernisation administration et communication",  "etudes",      0.008),
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "30","Programme MEDA",
     "28","Soutien à la cellule de gestion du projet",      "etudes",      0.005),

    # ── Programme 303 : Renforcement des Droits et des Libertés (7 %) ─────────
    # Projet 10 : Protection des droits des détenus (14.0 MDH)
    (303,"Renforcement des Droits et Libertés","00","SERVICES COMMUNS",
     "10","Protection des droits des détenus",
     "10","Travaux établissements de protection des détenus","travaux",    0.030),
    (303,"Renforcement des Droits et Libertés","00","SERVICES COMMUNS",
     "10","Protection des droits des détenus",
     "20","Équipements pénitentiaires",                    "equipements", 0.015),
    (303,"Renforcement des Droits et Libertés","00","SERVICES COMMUNS",
     "10","Protection des droits des détenus",
     "30","Équipements et matériel de réinsertion",        "equipements", 0.008),
    # Projet 20 : Détention provisoire (14.0 MDH)
    (303,"Renforcement des Droits et Libertés","00","SERVICES COMMUNS",
     "20","Détention provisoire",
     "10","Travaux des établissements de détention prov.",  "travaux",     0.030),
    (303,"Renforcement des Droits et Libertés","00","SERVICES COMMUNS",
     "20","Détention provisoire",
     "20","Équipements des établissements de détention",   "equipements", 0.010),
    # Lignes régionales — établissements pénitentiaires
    (303,"Renforcement des Droits et Libertés","RG","RG",
     "10","Établissements pénitentiaires régionaux",
     "10","Construction établissements pénitentiaires rég.","travaux",     0.025),
    (303,"Renforcement des Droits et Libertés","RG","RG",
     "10","Établissements pénitentiaires régionaux",
     "20","Équipements pénitentiaires régionaux",          "equipements", 0.008),
    (303,"Renforcement des Droits et Libertés","RG","RG",
     "20","Fournitures",
     "10","Fournitures et matières consommables",          "fournitures", 0.005),
]

# Expansion des lignes régionales ("RG") vers les 4 régions réelles
LINES_EXPANDED = []
for r in LINES_RAW:
    if r[2] == "RG":
        n = len(REGIONAL_ZONES)
        for reg_code, reg_label in REGIONAL_ZONES:
            LINES_EXPANDED.append((*r[:2], reg_code, reg_label, *r[4:9], r[9] / n))
    else:
        LINES_EXPANDED.append(r)

# Normaliser les parts
total_share = sum(r[9] for r in LINES_EXPANDED)
LINES = [(*r[:9], r[9] / total_share) for r in LINES_EXPANDED]
n_lines = len(LINES)

# ── Profils d'exécution par type de ligne (T1, T2, T3, T4) ──────────────────
# Reflètent les délais typiques du cycle de la commande publique marocaine
PROFILES = {
    "travaux":      [0.012, 0.095, 0.310, 0.850],  # lancement marché T1, paiement massif T4
    "acquisitions": [0.020, 0.130, 0.300, 0.780],  # délais livraison/réception
    "equipements":  [0.080, 0.290, 0.570, 0.930],  # commandes passées T1-T2, livraison T3-T4
    "etudes":       [0.120, 0.370, 0.670, 0.940],  # études lancées tôt, paiements réguliers
    "fournitures":  [0.180, 0.450, 0.720, 0.970],  # consommables, exécution régulière
}

YEARS = list(range(2020, 2026))
records = []

for year in YEARS:
    lf_total = TGR[year]["lf"]
    shares   = np.array([r[9] for r in LINES])

    # Facteurs de performance ligne-spécifiques (variabilité inter-annuelle)
    perf = np.random.normal(1.0, 0.10, n_lines).clip(0.75, 1.35)

    # Génération brute par ligne × trimestre
    raw = np.zeros((n_lines, 4))
    for i, row in enumerate(LINES):
        profile = np.array(PROFILES[row[8]])
        noise   = np.random.normal(0, [0.008, 0.025, 0.045, 0.035])
        raw[i]  = np.clip(profile * perf[i] + noise, 0.0, 1.40)

    # Calibration : pour chaque trimestre, ajuster pour coller à l'agrégat TGR
    calibrated = raw.copy()
    for q in range(4):
        current_mean = shares @ raw[:, q]
        target = TGR[year][q + 1]
        if current_mean > 1e-6:
            scale = target / current_mean
            calibrated[:, q] = np.clip(raw[:, q] * scale, 0.0, 1.50)
        else:
            calibrated[:, q] = 0.0

    # Forcer la monotonicité (taux cumulatif croissant)
    for i in range(n_lines):
        for q in range(1, 4):
            calibrated[i, q] = max(calibrated[i, q], calibrated[i, q - 1])

    # Enregistrer
    for i, row in enumerate(LINES):
        prog, prog_l, reg, reg_l, proj, proj_l, lig, lig_l, type_l, share = row
        lf_ligne = round(lf_total * share, 3)
        for q in range(4):
            records.append({
                "year": year, "quarter": q + 1,
                "programme": prog,      "programme_label": prog_l,
                "region":    reg,       "region_label":    reg_l,
                "projet":    proj,      "projet_label":    proj_l,
                "ligne":     lig,       "ligne_label":     lig_l,
                "type_ligne": type_l,
                "lf_mdh":    lf_ligne,
                "taux":      round(calibrated[i, q], 4),
                "line_id":   f"P{prog}-R{reg}-Pj{proj}-L{lig}",
                "budget_type": "INVESTISSEMENT",
            })

df_inv = pd.DataFrame(records)

# ── MATERIEL (Fonctionnement) simulation ─────────────────────────────────────
# TGR aggregates from data_long.csv — category MATERIEL
TGR_MAT = {
    2020: {"lf": 317.4, 1: 0.220, 2: 0.360, 3: 0.640, 4: 0.939},
    2021: {"lf": 272.5, 1: 0.124, 2: 0.498, 3: 0.611, 4: 0.956},
    2022: {"lf": 283.9, 1: 0.387, 2: 0.532, 3: 0.672, 4: 0.978},
    2023: {"lf": 333.9, 1: 0.351, 2: 0.520, 3: 0.672, 4: 0.974},
    2024: {"lf": 304.1, 1: 0.288, 2: 0.668, 3: 0.878, 4: 1.200},
    2025: {"lf": 352.8, 1: 0.370, 2: 0.500, 3: 0.691, 4: 1.021},
}

# Structure from Morasse Budgétaire — Fonctionnement Matériel & Dépenses Div.
# Total 2026 = 376.449 MDH  (P300:46.4%  P301:39.9%  P302:6.2%  P303:7.4%)
MATERIEL_LINES_RAW = [
    # P300 Soutien et Pilotage ─────────────────────────────────────────────────
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "10","Infrastructure",
     "11","Entretien et maintenance bâtiments administratifs",  "fournitures", 0.120),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "10","Infrastructure",
     "12","Travaux d'entretien courant et sécurité",            "acquisitions",0.075),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "10","Infrastructure",
     "13","Charges locatives et co-propriété",                  "acquisitions",0.043),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "20","Gestion des ressources",
     "21","Fournitures de bureau, informatique et imprimerie",  "fournitures", 0.090),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "20","Gestion des ressources",
     "22","Achats matériel et équipements administratifs",      "acquisitions",0.080),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "20","Gestion des ressources",
     "23","Entretien et réparation véhicules — parc auto",      "fournitures", 0.055),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "30","Contribution",
     "30","Contribution organismes et associations nationales", "fournitures", 0.015),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "40","Gestion des archives",
     "40","Entretien et numérisation des archives",             "fournitures", 0.008),
    (300,"Soutien et Pilotage","00","SERVICES COMMUNS",
     "50","Efficacité Énergétique",
     "50","Consommation énergie et efficacité énergétique",     "acquisitions",0.014),
    # P301 Performance Admin. Judiciaire ─────────────────────────────────────
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "10","Performance tribunaux — matière civile",
     "11","Fournitures et matériel juridictions civiles",       "fournitures", 0.050),
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "10","Performance tribunaux — matière civile",
     "12","Matériel informatique et logiciels tribunaux civils","acquisitions",0.046),
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "20","Performance tribunaux — matière pénale",
     "21","Fournitures et matériel juridictions pénales",       "fournitures", 0.100),
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "20","Performance tribunaux — matière pénale",
     "22","Frais de fonctionnement parquets et juridictions",   "fournitures", 0.097),
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "30","Accès au droit et à la justice",
     "30","Fournitures bureaux d'aide juridique",               "fournitures", 0.106),
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "40","Projet Mahkamati (Banque Mondiale)",
     "40","Matériel et équipements Mahkamati",                  "acquisitions",0.020),
    (301,"Performance Admin. Judiciaire","00","SERVICES COMMUNS",
     "50","Projet Adl (coopération Maroc-Espagnole)",
     "50","Matériel et fournitures Adl",                        "acquisitions",0.010),
    # P302 Modernisation ─────────────────────────────────────────────────────
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "10","Performance législative et organisationnelle",
     "11","Matériel et équipements SI judiciaire",              "acquisitions",0.032),
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "10","Performance législative et organisationnelle",
     "12","Licences et maintenance logiciels judiciaires",      "acquisitions",0.018),
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "20","Tribunal numérique",
     "21","Matériel serveurs et infrastructure numérique",      "acquisitions",0.012),
    (302,"Modernisation Système Judiciaire","00","SERVICES COMMUNS",
     "30","Programme MEDA",
     "30","Fournitures et matériel programme MEDA",             "fournitures", 0.010),
    # P303 Renforcement des Droits ───────────────────────────────────────────
    (303,"Renforcement des Droits et Libertés","00","SERVICES COMMUNS",
     "10","Protection des droits des détenus",
     "11","Fournitures établissements pénitentiaires",          "fournitures", 0.030),
    (303,"Renforcement des Droits et Libertés","00","SERVICES COMMUNS",
     "10","Protection des droits des détenus",
     "12","Matériel médical et sanitaire prisons",              "acquisitions",0.014),
    (303,"Renforcement des Droits et Libertés","00","SERVICES COMMUNS",
     "20","Détention provisoire",
     "21","Fournitures centres détention provisoire",           "fournitures", 0.025),
    (303,"Renforcement des Droits et Libertés","00","SERVICES COMMUNS",
     "20","Détention provisoire",
     "22","Matériel et équipements centres détention",          "acquisitions",0.013),
]

# Normaliser parts MATERIEL
total_share_mat = sum(r[9] for r in MATERIEL_LINES_RAW)
MAT_LINES = [(*r[:9], r[9] / total_share_mat) for r in MATERIEL_LINES_RAW]
n_mat_lines = len(MAT_LINES)

# Profils MATERIEL — exécution plus régulière et front-loaded
PROFILES_MAT = {
    "fournitures":  [0.330, 0.550, 0.760, 0.975],
    "acquisitions": [0.230, 0.480, 0.720, 0.965],
    "equipements":  [0.200, 0.440, 0.700, 0.955],
}

mat_records = []
np.random.seed(99)
for year in YEARS:
    lf_total = TGR_MAT[year]["lf"]
    shares   = np.array([r[9] for r in MAT_LINES])
    perf = np.random.normal(1.0, 0.08, n_mat_lines).clip(0.80, 1.30)
    raw = np.zeros((n_mat_lines, 4))
    for i, row in enumerate(MAT_LINES):
        profile = np.array(PROFILES_MAT[row[8]])
        noise   = np.random.normal(0, [0.015, 0.025, 0.030, 0.020])
        raw[i]  = np.clip(profile * perf[i] + noise, 0.0, 1.60)
    calibrated = raw.copy()
    for q in range(4):
        current_mean = shares @ raw[:, q]
        target = TGR_MAT[year][q + 1]
        if current_mean > 1e-6:
            scale = target / current_mean
            calibrated[:, q] = np.clip(raw[:, q] * scale, 0.0, 1.60)
        else:
            calibrated[:, q] = 0.0
    # Soft monotonicity (allow tiny reductions as MATERIEL taux can vary)
    for i in range(n_mat_lines):
        for q in range(1, 4):
            calibrated[i, q] = max(calibrated[i, q], calibrated[i, q - 1] * 0.98)
    for i, row in enumerate(MAT_LINES):
        prog, prog_l, reg, reg_l, proj, proj_l, lig, lig_l, type_l, share = row
        lf_ligne = round(lf_total * share, 3)
        for q in range(4):
            mat_records.append({
                "year": year, "quarter": q + 1,
                "programme": prog,      "programme_label": prog_l,
                "region":    reg,       "region_label":    reg_l,
                "projet":    proj,      "projet_label":    proj_l,
                "ligne":     lig,       "ligne_label":     lig_l,
                "type_ligne": type_l,
                "lf_mdh":    lf_ligne,
                "taux":      round(calibrated[i, q], 4),
                "line_id":   f"M{prog}-R{reg}-Pj{proj}-L{lig}",
                "budget_type": "MATERIEL",
            })

df_mat = pd.DataFrame(mat_records)

# ── PERSONNEL (Fonctionnement) simulation ───────────────────────────────────
# Source : data_long.csv — catégorie PERSONNEL
# Une seule ligne : P300 / R00 / Proj10 / «Soutien des missions» (4 120 MDH)
TGR_PERS = {
    2020: {"lf": 4748.1, 1: 0.240, 2: 0.490, 3: 0.744, 4: 0.988},
    2021: {"lf": 4717.2, 1: 0.240, 2: 0.502, 3: 0.753, 4: 1.014},
    2022: {"lf": 5035.7, 1: 0.256, 2: 0.502, 3: 0.745, 4: 0.998},
    2023: {"lf": 3189.9, 1: 0.252, 2: 0.492, 3: 0.725, 4: 0.966},
    2024: {"lf": 3192.6, 1: 0.229, 2: 0.486, 3: 0.737, 4: 0.997},
    2025: {"lf": 3431.5, 1: 0.234, 2: 0.492, 3: 0.746, 4: 1.000},
}

# La ligne personnel est unique — le taux de la ligne = l’agrégat TGR directement
# Faible bruit (salaires très prévisibles)
pers_records = []
np.random.seed(55)
for year in YEARS:
    lf_val = TGR_PERS[year]["lf"]
    noise = np.random.normal(0, 0.003, 4)          # bruit minimal ±0.3 pp
    vals  = [round(np.clip(TGR_PERS[year][q + 1] + noise[q], 0.0, 1.50), 4)
             for q in range(4)]
    for q in range(1, 4):                           # monotonicite
        vals[q] = max(vals[q], vals[q - 1])
    for q in range(4):
        pers_records.append({
            "year": year, "quarter": q + 1,
            "programme": 300, "programme_label": "Soutien et Pilotage",
            "region":    "00", "region_label":    "SERVICES COMMUNS",
            "projet":    "10", "projet_label":    "Soutien des missions",
            "ligne":     "10", "ligne_label":     "Dépenses de personnel — Soutien des missions",
            "type_ligne": "personnel",
            "lf_mdh":    lf_val,
            "taux":      vals[q],
            "line_id":   "RH300-R00-Pj10-L10",
            "budget_type": "PERSONNEL",
        })

df_pers = pd.DataFrame(pers_records)
df = pd.concat([df_inv, df_mat, df_pers], ignore_index=True)

# Features dérivées (lf_ratio per line_id ; lf_share per budget_type × year)
mean_lf        = df.groupby("line_id")["lf_mdh"].transform("mean")
df["lf_ratio"] = (df["lf_mdh"] / mean_lf).round(4)
year_budget_lf = df.groupby(["budget_type", "year"])["lf_mdh"].transform("sum")
df["lf_share"] = (df["lf_mdh"] / year_budget_lf).round(4)

df.sort_values(["budget_type","programme","region","projet","ligne","year","quarter"],
               inplace=True)
df.reset_index(drop=True, inplace=True)

# ── Vérification calibration ─────────────────────────────────────────────────
for budget, TGR_ref in [("INVESTISSEMENT", TGR), ("MATERIEL", TGR_MAT), ("PERSONNEL", TGR_PERS)]:
    print(f"\nVérification {budget} :")
    db = df[df["budget_type"] == budget]
    for year in YEARS:
        dy = db[db["year"] == year]
        for q in range(1, 5):
            dq = dy[dy["quarter"] == q]
            sim = (dq["taux"] * dq["lf_mdh"]).sum() / dq["lf_mdh"].sum()
            tgt = TGR_ref[year][q]
            flag = "  ✓" if abs(sim - tgt) < 0.020 else "  !"
            print(f"  {year} T{q} : sim={sim:.3f}  cible={tgt:.3f}  Δ={sim-tgt:+.4f}{flag}")

df.to_csv("data_lines.csv", index=False)
print(f"\nSauvegardé data_lines.csv  ({len(df)} lignes, {df['line_id'].nunique()} lignes budgétaires)")
print(df[["year","quarter","programme","region","ligne_label","taux"]].head(8).to_string())
