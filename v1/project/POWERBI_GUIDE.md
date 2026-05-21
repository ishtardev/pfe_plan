# Guide Power BI — Tableau de bord prévisions budgétaires
**Pour première utilisation. Suivre les étapes dans l'ordre.**

Source de données : `previsions_2025_powerbi.xlsx` (6 onglets, généré par `07_export_powerbi.py`)

Temps estimé : 45–60 min la première fois.

---

## 0. Installation (5 min)

1. Télécharger **Power BI Desktop** (gratuit) :
   https://powerbi.microsoft.com/desktop/
   → Choisir « Télécharger gratuitement » (version Microsoft Store ou installeur `.exe`).
2. Lancer Power BI Desktop. Fermer la fenêtre de bienvenue (« Get started »).
3. Si on te demande de te connecter, tu peux **cliquer sur "Sign in later"** (pas obligatoire pour travailler en local).

L'interface a 3 zones principales :
- **Gauche** : 3 icônes verticales = Report (graphiques), Table (données), Model (relations)
- **Centre** : le canvas où tu places tes visuels
- **Droite** : panneau **Visualizations** (types de graphiques) et **Data** (champs/colonnes)

---

## 1. Importer les données (5 min)

1. Ruban du haut → **Home** → **Get data** → **Excel workbook**
2. Naviguer vers `C:\Users\Inann\Desktop\pfe_plan\project\previsions_2025_powerbi.xlsx`
3. Une fenêtre **Navigator** s'ouvre. Cocher les 6 onglets :
   - ☑ Prévisions
   - ☑ Synthèse
   - ☑ Anomalies
   - ☑ Historique catégorie
   - ☑ Prédit vs Réel
   - ☑ Historique lignes
4. Cliquer **Load** (pas *Transform Data* pour l'instant — les types sont déjà propres).

Tu vois maintenant les 6 tables à droite dans le panneau **Data**.

---

## 2. Vérifier les types de colonnes (3 min)

1. Cliquer l'icône **Table view** (2ème icône à gauche, ressemble à un tableau).
2. Sélectionner la table **Prévisions** dans le panneau Data.
3. Vérifier que :
   - `LF (MDH)`, `T2 realise (%)`, `T4 predit T3 (%)`, etc. → type **Decimal Number** (icône `1.2`)
   - `Annee`, `Annees d'historique` → type **Whole Number** (icône `123`)
   - `Programme`, `Ligne budgetaire`, `Region`, `Type budget`, `Risque` → type **Text** (icône `ABC`)

Si un type est faux : clic droit sur la colonne → **Change Type** → choisir le bon.

---

## 3. Créer les relations entre tables (5 min)

1. Cliquer l'icône **Model view** (3ème icône à gauche, ressemble à un schéma).
2. Tu vois les 6 tables flottantes. Tu vas connecter celles qui ont des champs en commun.

**Relations à créer** (glisser-déposer une colonne sur l'autre) :

| De (table)                | Colonne          | Vers (table)         | Colonne          |
|---------------------------|------------------|----------------------|------------------|
| Prévisions                | Type budget      | Historique catégorie | Type budget      |
| Prévisions                | Type budget      | Prédit vs Réel       | Type budget      |
| Prévisions                | Ligne budgetaire | Historique lignes    | Ligne budgetaire |

Pour chaque relation : Power BI ouvre une fenêtre. Vérifier :
- **Cardinality** : *Many to one* ou *One to many* (Power BI choisit tout seul)
- **Cross filter direction** : *Single* (par défaut, OK)
- Cliquer **OK**

> **Si Power BI dit "no unique values" :** ce n'est pas grave. Cliquer "Many to many" et OK. Pour la démo, ça marche.

---

## 4. Créer les mesures DAX (15 min)

Les **mesures** sont des calculs réutilisables. On va en créer 6.

**Comment créer une mesure :**
1. Dans le panneau Data, clic droit sur la table **Prévisions** → **New measure**
2. Une barre de formule s'ouvre en haut. Coller la formule.
3. Appuyer sur Entrée. La mesure apparaît avec une icône calculatrice.

### Mesure 1 — Nombre de lignes
```DAX
Nb lignes = COUNTROWS('Prévisions')
```

### Mesure 2 — Budget total
```DAX
Budget total (MDH) = SUM('Prévisions'[LF (MDH)])
```

### Mesure 3 — Taux moyen pondéré prédit
```DAX
Taux moyen prédit (%) =
DIVIDE(
    SUMX('Prévisions', 'Prévisions'[T4 predit T3 (%)] * 'Prévisions'[LF (MDH)]),
    SUM('Prévisions'[LF (MDH)])
)
```
> Pondération par le poids de chaque ligne (LF). Plus correct qu'une moyenne simple.

### Mesure 4 — Lignes à risque
```DAX
Lignes à risque =
CALCULATE(
    COUNTROWS('Prévisions'),
    'Prévisions'[Risque] IN { "Alerte", "Attention" }
)
```

### Mesure 5 — Montant à risque
```DAX
Montant à risque (MDH) = SUM('Prévisions'[MDH a risque])
```

### Mesure 6 — Anomalies détectées
```DAX
Nb anomalies =
CALCULATE(
    COUNTROWS('Prévisions'),
    NOT(ISBLANK('Prévisions'[Anomalie]))
)
```

**Bon réflexe :** sélectionner chaque mesure dans le panneau Data → onglet **Measure tools** dans le ruban → **Format** : choisir le format d'affichage (`Whole Number` pour Nb, `Decimal Number` avec 1 décimale pour les %, etc.).

---

## 5. Construire la Page 1 — Vue d'ensemble (10 min)

Cliquer l'icône **Report view** (1ère icône à gauche). Renommer la page en bas : double-clic sur "Page 1" → taper **« Vue d'ensemble »**.

### 5.1 — KPI cards (en haut)

Tu vas ajouter 4 cartes côte à côte.

Pour chaque carte :
1. Dans Visualizations (à droite), cliquer l'icône **Card** (un grand chiffre `123`)
2. Un visuel vide apparaît. Le redimensionner (~200×120 px).
3. Glisser la mesure dans le champ **Fields** :
   - Carte 1 : `Nb lignes`
   - Carte 2 : `Budget total (MDH)`
   - Carte 3 : `Taux moyen prédit (%)`
   - Carte 4 : `Lignes à risque`

**Customisation de chaque carte :**
- Cliquer la carte → onglet **Format your visual** (icône pinceau)
- **Callout value** → Font size : 32, Color : `#1F3864` (bleu foncé)
- **Category label** → On (affiche le nom)
- **Effects → Background** : couleur claire (gris très pâle ou blanc)

### 5.2 — Évolution historique par catégorie

1. Cliquer dans une zone vide du canvas.
2. Visualizations → **Line chart** (icône courbe)
3. Glisser dans :
   - **X-axis** : `Historique catégorie` → `Annee`
   - **Y-axis** : `Historique catégorie` → `Taux execution (%)` (drop-down → *Average*)
   - **Legend** : `Historique catégorie` → `Type budget`
4. Filtre : à droite dans **Filters on this visual**, glisser `Trimestre` et cocher uniquement **T4** (on veut l'exécution annuelle).
5. Format → Title → taper **« Taux d'exécution T4 par catégorie (2020–2025) »**

### 5.3 — Top 10 lignes à risque

1. Visualizations → **Bar chart** (barres horizontales)
2. Glisser :
   - **Y-axis** : `Prévisions` → `Ligne budgetaire`
   - **X-axis** : `Prévisions` → `MDH a risque` (Sum)
3. Filtre : `MDH a risque` → *is greater than* → 0
4. Visual → **Top N** filter : `Ligne budgetaire` by `MDH a risque` → Top 10
5. Format → Title : **« Top 10 lignes à risque (MDH) »**

### 5.4 — Slicers (filtres)

En bas de la page, ajouter 2 slicers :
1. Visualizations → **Slicer**
   - Field : `Prévisions` → `Type budget`
   - Format → Slicer settings → Style : *Tile* (boutons)
2. Deuxième slicer :
   - Field : `Prévisions` → `Programme`
   - Style : *Dropdown*

> Tester : clique sur un bouton "INVESTISSEMENT" → tous les visuels se filtrent.

---

## 6. Construire la Page 2 — Validation modèle (10 min)

En bas, clic droit sur l'onglet → **New page**. Renommer **« Prédit vs Réel »**.

### 6.1 — Bar chart comparaison

1. Visualizations → **Clustered column chart**
2. Glisser :
   - **X-axis** : `Prédit vs Réel` → `Annee`
   - **Y-axis** : `Prédit vs Réel` → `T4 reel (%)` ET `T4 predit (%)` (les deux)
   - **Small multiples** : `Type budget` (créera 3 sous-graphiques côte à côte, un par catégorie)
3. Format → Title : **« Taux T4 réel vs prédit par catégorie »**
4. Format → Y-axis → Range : Min 0, Max 110 (pour bien voir)
5. Format → Data labels → On (affiche les valeurs)

### 6.2 — Table d'écart

1. Visualizations → **Table**
2. Glisser : `Annee`, `Type budget`, `T4 reel (%)`, `T4 predit (%)`, `Ecart (pp)`
3. Format → **Conditional formatting** sur `Ecart (pp)` :
   - Background color → Rules
   - If value `< -5` → rouge clair
   - If value between `-5 and 5` → vert clair
   - If value `> 5` → orange clair
4. Title : **« Détail écarts par année »**

### 6.3 — Carte texte (commentaire)

1. Ruban **Insert** → **Text box**
2. Taper :
   > « Le modèle est validé sur les années 2022–2024 (back-testing). L'écart entre la prévision et la réalisation est de l'ordre de quelques points de pourcentage. La prédiction 2025 utilise les données disponibles au T2 et T3. »

---

## 7. Construire la Page 3 — Détail par ligne (10 min)

Nouvelle page → renommer **« Détail ligne »**.

### 7.1 — Slicers à gauche (colonne)

3 slicers empilés verticalement :
- `Programme` (style Dropdown)
- `Region` (style Dropdown)
- `Type budget` (style Tile)

### 7.2 — Table principale (centre)

1. Visualizations → **Table**
2. Champs : `Ligne budgetaire`, `Region`, `LF (MDH)`, `T2 realise (%)`, `T4 predit T3 (%)`, `Risque`, `MDH a risque`, `Anomalie`
3. **Conditional formatting** sur `Risque` :
   - Background → Rules : "Alerte" → rouge, "Attention" → jaune, "OK" → vert
4. Trier par `MDH a risque` décroissant (clic sur l'entête).

### 7.3 — Historique de la ligne sélectionnée (droite)

1. Visualizations → **Line chart**
2. Champs :
   - X-axis : `Historique lignes` → `Annee`
   - Y-axis : `Historique lignes` → `Taux (%)` (Average)
3. Filtre : `Trimestre` = T4
4. Title : **« Historique T4 — ligne sélectionnée »**

> **Important** : pour que ce graphique se filtre quand on clique une ligne dans la table, il faut que la relation `Prévisions[Ligne budgetaire] → Historique lignes[Ligne budgetaire]` soit créée (étape 3).

### 7.4 — Card "Anomalie"

1. Card visual → Field : `Anomalie`
2. Title : **« Motif d'anomalie (si applicable) »**

---

## 8. Mise en forme finale (5 min)

### Thème
- Ruban **View** → **Themes** → choisir un thème sobre (ex. *Executive*) ou *Customize current theme* pour mettre les couleurs du Ministère.

### Titre du rapport
- Page 1, en haut, ajouter un **Text box** : « Prévisions d'exécution budgétaire 2025 — Ministère de la Justice »
- Police 20, gras, couleur `#1F3864`

### En-tête commun
- Ajouter en bas de chaque page un petit text box : « Source : modèle ML, prévisions générées le 29/04/2026 »

---

## 9. Sauvegarder (1 min)

- **File → Save as** → `previsions_2025_dashboard.pbix`
- Pour partager une version réutilisable : **File → Export → Power BI template** → `.pbit` (ne contient pas les données, juste la structure).

---

## 10. Pendant la démo à ton encadrant

**Ouverture (30 s)**
> « Voici le tableau de bord. Trois pages : vue d'ensemble, validation du modèle, et détail par ligne. »

**Page 1 (3 min)**
- Pointer les KPI : 130 lignes, X MDH de budget, taux moyen prédit Y%.
- Cliquer sur un bouton catégorie : « Si je filtre Investissement, voilà la dynamique. »
- Pointer le top 10 : « Ces lignes sont à surveiller en priorité. »

**Page 2 (3 min) — la plus importante**
- « Pour vérifier que le modèle est fiable, on l'a testé sur 2022, 2023, 2024 en lui cachant la donnée. »
- Pointer les écarts : « En moyenne, X points d'écart, ce qui est inférieur à la méthode actuelle. »

**Page 3 (3 min)**
- « Maintenant, si vous voulez creuser une ligne précise, je sélectionne le programme XYZ, la région ABC… »
- Cliquer une ligne dans la table : « Et là, on voit son historique 2020–2024 et la prédiction 2025. »

**Conclusion (1 min)**
- « Ce tableau est rafraîchi automatiquement quand de nouvelles données sont disponibles. La prochaine étape serait de le brancher directement sur le GID. »

---

## Pièges courants

- **« Cannot create relationship »** : vérifier que les noms de colonnes sont exactement les mêmes (Power BI est sensible aux espaces).
- **Slicer ne filtre pas un visuel** : vérifier la relation entre les tables (étape 3).
- **Valeurs en %.** Si Power BI affiche 0.045 au lieu de 4.5% : sélectionner la colonne dans Table view → ruban **Column tools** → Format → Percentage. Ou les colonnes sont déjà en % (0–100), auquel cas garder Decimal Number et juste mettre l'unité dans le titre.
- **« No data »** : vérifier que les filtres ne sont pas trop restrictifs (clic droit sur le visuel → Clear filters).

---

## Si tu bloques

Dis-moi à quelle étape, je te débugge. Les pièges classiques pour un débutant : les relations (étape 3), les types de colonnes (étape 2), et les agrégations (Sum vs Average dans les visuels).
