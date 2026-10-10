# Guide Étape par Étape — Mini-Projet Séries Temporelles (TEK-UP)
## Modélisation, Prévision et Diagnostic de Séries Temporelles Multidimensionnelles
### *Coca-Cola (KO) & PepsiCo (PEP) — 2019-2024*

Ce guide explique pas à pas comment installer, exécuter, reproduire et défendre l'intégralité du mini-projet sur machine locale ou serveur.

---

## 1. Structure du Projet et Fichiers Livrables

```text
projet/
├── data/
│   ├── data_ko_pep_daily.csv       # Données brutes réelles téléchargées depuis Yahoo Finance
│   ├── data_processed.csv          # Données nettoyées avec log-rendements et indicateurs techniques
│   └── test_forecast_comparison.csv# Prévisions hors-échantillon (151 jours) de tous les modèles
├── presentation/
│   ├── figures/                    # 14 graphiques haute résolution (Dark theme, 200 DPI)
│   └── index.html                  # Copie locale de la présentation web interactive
├── scripts/
│   ├── download_and_preprocess.py  # Téléchargement et ingénierie des caractéristiques réelles
│   ├── run_econometrics_test.py    # Validation expérimentale des tests économétriques
│   ├── test_dl_and_evaluation.py   # Entraînement Deep Learning et calcul du test de Diebold-Mariano
│   ├── build_notebook.py           # Générateur du notebook Jupyter
│   ├── execute_notebook.py         # Exécuteur automatisé de bout en bout
│   ├── generate_presentation_assets.py # Générateur des 14 figures professionnelles
│   ├── generate_pptx.py            # Générateur de la présentation PowerPoint officielle
│   ├── generate_html_presentation.py # Générateur de la présentation HTML interactive
│   ├── inspect_coint_details.py    # Script de diagnostic cointégration Python
│   ├── inspect_coint_r.R           # Script de diagnostic cointégration R (Engle-Granger, Johansen, VECM)
│   └── render_rmd.R                # Script de compilation RMarkdown vers HTML
├── presentation_soutenance.pptx    # Diaporama PowerPoint officiel (24 slides 16:9 + Speaker Notes)
├── presentation_soutenance.html    # Diaporama Web interactif (Plein écran, Zoom figures, Export PDF)
├── Mini_Projet_Series_Temporelles.ipynb # Notebook Jupyter principal entièrement pré-calculé (45 cellules)
├── socle_classique_R.Rmd           # Document RMarkdown du socle classique
├── socle_classique_R.html          # Rapport HTML généré avec succès par R / Pandoc
├── requirements.txt                # Dépendances logicielles Python
├── rapport_structure_detaillee.md  # Rapport scientifique complet (15-25 pages)
└── README_Guide_etape_par_etape.md # Le présent guide
```

---

## 2. Installation de l'Environnement

### 2.1 Prérequis Python (Version 3.10 à 3.13)
Dans le terminal (PowerShell ou Bash), installez les dépendances :
```bash
pip install -r requirements.txt
```

### 2.2 Prérequis R (Version 4.2+)
Dans une console R ou RStudio, installez les packages économétriques :
```R
install.packages(c("readr", "ggplot2", "tseries", "urca", "forecast", "vars", "fGarch", "lmtest", "FinTS", "rmarkdown"))
```

---

## 3. Téléchargement et Préparation des Données Réelles

Pour rafraîchir ou retélécharger les cours réels de Coca-Cola (KO) et PepsiCo (PEP) :
```bash
python scripts/download_and_preprocess.py
```
Ce script :
1. Télécharge les cotations réelles 2019-2024 via Yahoo Finance.
2. Calcule les log-rendements : $r_t = 100 \times \ln(P_t / P_{t-1})$.
3. Calcule la volatilité réalisée glissante annualisée sur 20 jours.
4. Calcule les moyennes mobiles (SMA 20, SMA 50) et le RSI (14 jours).
5. Sauvegarde le dataset vérifié dans `data/data_processed.csv`.

---

## 4. Exécution du Projet

### 4.1 Exécution du Notebook Python (`Mini_Projet_Series_Temporelles.ipynb`)
Le notebook est déjà fourni **entièrement pré-calculé** avec tous ses graphiques, diagnostics et métriques.

Pour l'ouvrir et l'exécuter interactivement :
```bash
jupyter lab Mini_Projet_Series_Temporelles.ipynb
```
Ou via Jupyter Notebook :
```bash
jupyter notebook Mini_Projet_Series_Temporelles.ipynb
```

Pour exécuter toutes les cellules de manière automatisée en ligne de commande :
```bash
python scripts/execute_notebook.py
```

### 4.2 Exécution et Rendu du Socle R (`socle_classique_R.Rmd`)
Pour compiler le document RMarkdown et générer le rapport interactif `socle_classique_R.html` :
```bash
Rscript scripts/render_rmd.R
```
Le fichier HTML généré (`socle_classique_R.html`) s'ouvre dans n'importe quel navigateur web (Chrome, Edge, Firefox).

---

## 5. Ordre Méthodologique des Étapes du Projet

1. **Source & Découpage** : 90 % Train ($N=1357$), 10 % Test ($N=151$). Aucun shuffle, zéro fuite temporelle.
2. **Analyse Exploratoire** :
   - Graphiques des prix, rendements, volatilité, volume.
   - Décomposition STL : Absence de saisonnalité déterministe cyclique ($<0.1\%$ de variance). SARIMA rejeté avec justification empirique.
   - Outliers IQR : Conservation justifiée (la volatilité boursière extrême est une réalité économique à modéliser, pas un bruit à effacer).
   - ACF / PACF : Décroissance lente sur les prix ($I(1)$), absence d'autocorrélation linéaire sur les rendements ($I(0)$).
3. **Stationnarité conjointe** :
   - ADF et KPSS concordent : Cours $I(1)$, Rendements $I(0)$. Ordre de différenciation $d=1$.
4. **Modélisation Multivariée** :
   - Cointégration préalable sur log-prix :
     - Engle-Granger : Stat = -3.5413, $p = 0.0289 < 0.05$ (Python) ; Phillips-Ouliaris $p = 0.01899 < 0.05$ (R).
     - Johansen : Trace stat pour $r=0$ = 20.12 > 19.96 (R `ecdet='const'`), rejet de l'absence de cointégration à 5% ; Trace pour $r \le 1$ = 6.98 < 9.24.
     - Correspondance mathématique : Python `coint_johansen(..., det_order=0)` et R `ca.jo(..., ecdet='none')` fournissent la même statistique de trace : 17.2539.
     - $\implies$ VECM pleinement justifié et estimé dans les deux langages (`cajorls()` sous R, `VECM()` sous Python), validant une élasticité $\beta \approx 0.80$ et un rappel significatif de Coca-Cola ($\alpha_{KO} \approx -0.021$, $p < 0.001$).
   - VAR sur rendements stationnaires (Arbitrage théorique : AIC choisit $p=9$, BIC choisit $p=1$ parsimonieux).
   - Granger Causality : Causalité bilatérale ($KO \leftrightarrow PEP$, $p<0.01$) démontrant une antériorité prédictive temporelle.
5. **ARIMA Univarié & Diagnostics** :
   - Confrontation systématique des ordres manuels vs sélection automatique `auto.arima` $\implies \text{ARIMA}(1,1,4)$.
   - Ljung-Box : $p > 0.99 \implies$ Résidus bruits blancs (dynamique de moyenne bien capturée).
   - Jarque-Bera : Statistique brute de 32M expliquée par l'artefact d'initialisation à $t=0$ ($e_0 = y_0 - 0 = 36.89$). Statistique économétrique épurée = $2967.01$ ($p < 10^{-16}$, kurtosis = 10.03). Rejet sain de la normalité dû à la leptokurticité financière.
   - ARCH-LM : $p = 3.77 \times 10^{-39} \implies$ Rejet massif de l'homoscédasticité conditionnelle.
6. **Modélisation de la Volatilité GARCH(1,1)** :
   - Paramètres $\omega = 0.046$, $\alpha_1 = 0.056$, $\beta_1 = 0.911$ ($\alpha_1 + \beta_1 = 0.967 < 1$, stationnaire).
   - Test ARCH-LM post-GARCH sur résidus standardisés : $p = 0.9907 \implies$ **Homoscédasticité conditionnelle entièrement restaurée**.
7. **Modélisation Récurrente Deep Learning (LSTM & GRU)** :
   - Normalisation `MinMaxScaler` calée strictement sur le train.
   - Fenêtres glissantes $W=20$ jours ouvrés.
   - Validation temporelle interne sans mélange via `TimeSeriesSplit` (5 folds).
8. **Approche Hybride (Bonus)** :
   - ARIMA modélise la composante linéaire, LSTM modélise les résidus non linéaires.
9. **Évaluation Hors-Échantillon (Test Set, 151 Jours)** :
   - Benchmark Naïf (Random Walk : $\hat{P}_t = P_{t-1}$) : RMSE = 0.5374, MAE = 0.4142.
   - ARIMA(1,1,4) : RMSE = 0.5475, MAE = 0.4159 (Diebold-Mariano $p=0.2221$, statistiquement équivalent au modèle naïf).
   - GRU : RMSE = 0.7015, MAE = 0.5594.
   - LSTM : RMSE = 1.4334, MAE = 1.2682.
   - Conclusion : Efficience des marchés d'Eugene Fama confirmée. Les modèles neuronaux complexes surapprennent le bruit.
10. **IA Générative et LLM (4 points)** :
    - Génération d'hypothèses à partir des vrais tests statistiques.
    - Vulgarisation pour un décideur non-statisticien.
    - Recommandations d'investissement simulées et audit critique des risques.
    - Tableau d'audit des hallucinations statistiques (détection de confusions causalité/corrélation et d'engouement injustifié pour le Deep Learning).

---

## 6. Points Clés pour la Soutenance Orale

Lors de la soutenance, l'enseignant évalue la **maîtrise conceptuelle et l'honnêteté scientifique** :
- **Pourquoi l'ARIMA et le Naïf battent-ils le LSTM ?**
  *Réponse* : Sur les marchés boursiers hautement liquides, les cours suivent approximativement une sous-martingale (marche aléatoire). Le signal prévisible est très faible devant la variance du bruit. Le modèle naïf minimise l'erreur quadratique sans introduire de variance d'estimation. Le LSTM, avec ses milliers de paramètres, surapprend le bruit d'entraînement (*overfitting*), ce qui génère une erreur supérieure hors-échantillon.
- **Que signifie le test ARCH-LM et pourquoi faire un GARCH ?**
  *Réponse* : Le test ARCH-LM vérifie si les résidus au carré sont autocorrélés. Même si les résidus ne sont pas corrélés en moyenne ($E[e_t e_{t-k}] = 0$), leur variance dépend des chocs passés ($E[e_t^2 | e_{t-1}^2] \ne \sigma^2$). Le GARCH(1,1) modélise cette variance conditionnelle dans le temps. Le test post-GARCH avec $p = 0.9907$ prouve que l'homoscédasticité a été rétablie.
- **Quelle est la différence entre causalité de Granger et causalité réelle ?**
  *Réponse* : La causalité de Granger est une antériorité temporelle prédictive dans un modèle autorégressif linéaire : elle prouve qu'une information passée de KO devance chronologiquement celle de PEP. Elle ne prouve aucun lien de cause à effet direct dans l'économie réelle.
- **Pourquoi la statistique brute de Jarque-Bera était-elle de 32 millions ?**
  *Réponse* : Lors de la différenciation $d=1$, l'algorithme initialise l'espace d'état à 0 au temps $t=0$, ce qui produit un premier résidu artificiel $e_0 = y_0 - 0 = 36.89$, alors que tous les autres résidus sont centrés autour de 0 ($\pm 0.5$). Ce point aberrant unique a propulsé la skewness à 23.8 et la kurtosis à 763. En supprimant cette observation transitoire ($t \ge 1$), la véritable statistique économétrique de Jarque-Bera est de **2,967.01** ($p < 10^{-16}$, kurtosis = 10.03), confirmant de façon saine la leptokurticité des rendements financiers.
- **Pourquoi la cointégration doit-elle être testée sur les log-prix et non sur les prix nominaux ?**
  *Réponse* : Coca-Cola et PepsiCo ont des niveaux de prix nominaux très différents (~50$ vs ~140$). Un spread linéaire nominal $P_{KO} - \beta P_{PEP}$ intègre des distorsions d'échelle non stationnaires. En revanche, les log-prix $\ln(P_{KO}) - \beta \ln(P_{PEP})$ modélisent l'élasticité et le ratio de prix relatif. Tant sous R (`ca.jo` avec $ecdet='const'$) que sous Python (Engle-Granger $p=0.0289$), la cointégration sur log-prix est robuste et confirme exactement $r = 1$ relation d'équilibre stable de long terme, ce qui justifie formellement l'estimation du VECM (`cajorls()` sous R, `VECM()` sous Python).
- **Pourquoi l'AIC a-t-il sélectionné VAR(9) alors que le BIC a sélectionné VAR(1) ?**
  *Réponse* : La pénalité du BIC est $\ln(T) \approx \ln(1348) \approx 7.21$, soit 3.6 fois plus sévère que la pénalité de 2 de l'AIC. Chaque retard ajoute 4 coefficients dans un VAR bivarié. Le BIC privilégie la **parsimonie** et élimine le surapprentissage du bruit journalier, tandis que l'AIC capture des variations mineures de calendrier bi-hebdomadaires.
- **Pourquoi une validation TimeSeriesSplit est-elle obligatoire pour le Deep Learning ?**
  *Réponse* : Un K-Fold classique brise la dépendance temporelle et utilise le futur pour prédire le passé (fuite de données). TimeSeriesSplit utilise une fenêtre expansive qui respecte strictement la flèche du temps.
- **Comment l'audit final garantit-il l'honnêteté scientifique ?**
  *Réponse* : Au lieu de cocher artificiellement tout en vert, l'audit explicite les nuances économétriques : la sensibilité de la cointégration à l'échelle logarithmique, l'arbitrage AIC/BIC du VAR, l'explication de l'artefact Jarque-Bera et la reconnaissance que le modèle naïf égale ou bat le Deep Learning conformément à l'efficience des marchés.
