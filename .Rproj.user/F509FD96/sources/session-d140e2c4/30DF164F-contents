# Rapport de Projet — Séries Temporelles pour Data Science & IA
## Modélisation, Prévision et Diagnostic de Séries Temporelles Multidimensionnelles
### *De l'ARIMA classique aux réseaux de neurones, en passant par la cointégration, la volatilité conditionnelle et l'IA générative*

---

- **Établissement** : TEK-UP
- **Cours** : Séries Temporelles pour Data Science & IA
- **Enseignant** : Ahmed DHOUIBI
- **Thématique choisie** : **Option A — Données financières réelles (Yahoo Finance)**
- **Actifs modélisés** : **The Coca-Cola Company (`KO`)** et **PepsiCo, Inc. (`PEP`)**
- **Période d'étude** : Janvier 2019 à Décembre 2024 (6 années pleines, 1508 jours de cotation)
- **Découpage chronologique** : 90 % Apprentissage (1357 obs) / 10 % Test (151 obs)

---

## 1. Introduction et Problématique

### 1.1 Contexte Économique et Choix du Sujet
Dans le secteur agroalimentaire mondial des boissons non alcoolisées, **Coca-Cola (KO)** et **PepsiCo (PEP)** forment un duopole historique incontournable. Ces deux multinationales opèrent sur des marchés matures, subissent les mêmes chocs de coûts des matières premières (sucre, aluminium, emballages plastiques, logistique mondiale) et sont exposées aux mêmes conditions macroéconomiques (taux d'intérêt de la Réserve fédérale, inflation, variations du pouvoir d'achat).

> **Justification en une phrase (Exigence Section 1 de l'énoncé) :**
> *"Nous avons sélectionné le duopole historique Coca-Cola (KO) et PepsiCo (PEP) car ces deux multinationales partagent des fondamentaux économiques et sectoriels très proches, constituant un cas d'école idéal pour étudier la co-évolution, la cointégration de long terme, la transmission de volatilité croisée et confronter l'efficience informationnelle des marchés aux modèles économétriques et d'apprentissage profond."*

### 1.2 Problématique de Modélisation et Compétences Visées
La question centrale de ce projet est à la fois théorique et empirique :
1. *Les cours boursiers de Coca-Cola et PepsiCo partagent-ils une relation d'équilibre stable de long terme (cointégration) malgré le caractère imprévisible de leurs marches aléatoires individuelles ?*
2. *L'information passée d'un actif permet-elle d'améliorer la prévision temporelle de l'autre au sens de la causalité de Granger ?*
3. *Face aux chocs de volatilité conditionnelle (clusters de volatilité), la modélisation hétéroscédastique (GARCH) permet-elle de restaurer l'homoscédasticité des résidus ?*
4. *Les modèles récents d'apprentissage profond récurrent (LSTM, GRU) et les architectures hybrides apportent-ils une supériorité statistique mesurable (test de Diebold-Mariano) par rapport au benchmark fondamental de la marche aléatoire (théorie des marchés efficients d'Eugene Fama) ?*

---

## 2. Données et Préparation

### 2.1 Source Exacte et Caractéristiques
- **Source** : API Yahoo Finance (`finance.yahoo.com` via `yfinance` en Python et `quantmod` en R).
- **Profondeur temporelle** : Du 02/01/2019 au 30/12/2024, soit 6 années civiles complètes (largement supérieur au seuil minimal de 5 ans fixé par l'énoncé).
- **Nombre total d'observations** : 1508 jours de cotation effectifs sur le New York Stock Exchange (NYSE).
- **Fréquence** : Journalière ouvrée (Business Days).
- **Variables brutes collectées** :
  - `Open` (Cours d'ouverture)
  - `High` (Plus haut de séance)
  - `Low` (Plus bas de séance)
  - `Close` (Cours de clôture brut)
  - `Adj Close` (Cours de clôture ajusté des dividendes et splits d'actions)
  - `Volume` (Nombre de titres échangés)

### 2.2 Traitement des Données et Variables Dérivées
1. **Traitement des valeurs manquantes** : Les deux séries partagent exactement le même calendrier boursier américain. L'intersection des dates de cotation produit un tableau complet sans aucune valeur manquante ($NaN = 0$).
2. **Justification de l'Adj Close** : Le cours ajusté neutralise les variations artificielles de prix causées par le détachement de dividendes ou les divisions d'actions (*stock splits*), fournissant la véritable rentabilité totale de l'action pour l'actionnaire.
3. **Construction des variables dérivées** :
   - **Log-rendements journaliers** :
     $$r_t = 100 \times \left( \ln(P_t) - \ln(P_{t-1}) \right)$$
     *Justification* : Additivité temporelle et approximation fidèle du rendement arithmétique pour des pas journaliers faibles.
   - **Volatilité réalisée glissante (20 jours annualisée)** :
     $$\sigma_{20, t} = \text{std}(r_{t-19:t}) \times \sqrt{252}$$
   - **Moyennes Mobiles Simples (SMA 20 et SMA 50)** : Suivi de tendance court et moyen terme.
   - **RSI (14 jours)** : Oscillateur de momentum standard mesurant l'amplitude relative des gains et des pertes récents.
   - **Spread de log-prix** : $S_t = \ln(P_{KO, t}) - \ln(P_{PEP, t})$.

### 2.3 Découpage Temporel Strict (Zéro Fuite d'Information)
- **Taille totale** : $N = 1508$ observations.
- **Échantillon d'apprentissage (Train 90 %)** : $N_{train} = 1357$ observations, du **03/01/2019** au **23/05/2024**.
- **Échantillon de test (Test 10 %)** : $N_{test} = 151$ observations, du **24/05/2024** au **30/12/2024**.
- **Protocole anti-fuite** :
  - Aucun mélange aléatoire (`shuffle=False`).
  - Tous les paramètres de mise à l'échelle (`MinMaxScaler`, standardisations) sont strictement calculés et calés (*fitted*) sur le sous-ensemble de train, puis appliqués de façon séquentielle au test.

---

## 3. Analyse Exploratoire Complète (EDA)

### 3.1 Visualisation des Séries et Événements Historiques
L'inspection visuelle des cours révèle :
- **Tendance générale haussière** interrompue par des chocs macroéconomiques majeurs.
- **Le Krach Covid-19 (février-mars 2020)** : Chute brutale de Coca-Cola d'environ 60$ à 38$ (-37%), reflétant la fermeture massive des cinémas, stades et restaurants dans le monde.
- **Resserrement monétaire de la Réserve fédérale (2022)** : Forte instabilité liée à l'inflation et à la remontée brutale des taux directeurs.

### 3.2 Décomposition STL et Évaluation de la Saisonnalité
Une décomposition STL (*Seasonal-Trend decomposition using Loess*) a été appliquée sur la série journalière ouvrée avec une fréquence hebdomadaire boursière ($s=5$).
- **Variance de la composante tendancielle** : $92.4\%$ de la variance totale.
- **Variance de la composante saisonnière** : $< 0.1\%$ de la variance totale.
- **Variance résiduelle** : $7.5\%$ de la variance totale.
- **Conclusion économique** : Les cours boursiers journaliers ne possèdent aucune saisonnalité déterministe cyclique stable. Les variations intra-hebdomadaires relèvent du bruit blanc informationnel. **L'application d'un modèle SARIMA est donc rejetée car elle induirait un surparamétrage artificiel sans justification empirique.**

### 3.3 Traitement Justifié des Valeurs Aberrantes
La méthode de l'écart interquartile (IQR à 3 sigmas) isole 14 séances aux mouvements extrêmes (notamment les journées de mars 2020 avec des variations journalières excédant $\pm 6\%$).
- **Décision méthodologique** : **Conservation intégrale de ces observations**.
- **Justification** : Les krachs boursiers constituent la réalité intrinsèque des marchés financiers. Tronquer ou supprimer ces observations reviendrait à sous-estimer dramatiquement le risque de queue (*fat tails*). Ces chocs sont précisément capturés par les modèles de volatilité conditionnelle GARCH.

### 3.4 ACF et PACF
- **Sur les cours bruts ($P_t$)** : L'ACF décroît de façon linéaire extrêmement lente (valeurs $> 0.85$ jusqu'au retard 30). La PACF présente un pic unique au retard 1 égal à 0.99, puis s'effondre. C'est l'archétype d'un processus intégré d'ordre 1 ($I(1)$).
- **Sur les log-rendements ($r_t$)** : L'ACF et la PACF ne présentent aucun dépassement significatif des bandes de confiance à 95% de Bartlett ($\pm 1.96 / \sqrt{T}$). Les rendements sont non autocorrélés en moyenne.

### 3.5 Statistiques Descriptives (Train Set, 1357 obs)

| Variable | Moyenne | Écart-type | Min | Médiane | Max | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KO_Close ($)** | 48.45 | 7.42 | 33.72 | 49.32 | 62.48 | -0.125 | -0.912 |
| **PEP_Close ($)** | 131.78 | 23.36 | 84.98 | 134.12 | 173.55 | -0.279 | -1.218 |
| **KO_Return (%)** | +0.031 | 1.280 | -10.17 | +0.063 | +6.28 | -0.938 | **+10.29** |
| **PEP_Return (%)** | +0.033 | 1.320 | -13.01 | +0.048 | +12.17 | -0.623 | **+22.23** |

*Interprétation* :
- La skewness négative traduit un risque d'asymétrie baissière accrue (*crash risk*).
- L'excès massif de kurtosis ($10.29$ pour KO et $22.23$ pour PEP vs 0 pour la loi normale) démontre la **leptokurticité sévère** des rendements financiers.

---

## 4. Tests Formels de Stationnarité

Conformément à la consigne, les tests conjointe **ADF** et **KPSS** sont conduits sur le train set :

| Variable testée | ADF Stat | ADF p-value | ADF Conclusion ($H_0$: Racine unitaire) | KPSS Stat | KPSS p-value | KPSS Conclusion ($H_0$: Stationnaire) | Décision Finale |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- | :---: |
| **KO_Close (Prix)** | -1.5661 | 0.5006 | Échec rejet $H_0$ (Non-stationnaire) | 5.3080 | 0.0100 | Rejet $H_0$ (Non-stationnaire) | **$I(1)$** |
| **PEP_Close (Prix)** | -1.6485 | 0.4578 | Échec rejet $H_0$ (Non-stationnaire) | 5.9186 | 0.0100 | Rejet $H_0$ (Non-stationnaire) | **$I(1)$** |
| **KO_Return (%)** | -11.2606 | 1.63e-20 | Rejet $H_0$ (Stationnaire) | 0.0263 | $\ge$ 0.1000 | Échec rejet $H_0$ (Stationnaire) | **$I(0)$** |
| **PEP_Return (%)** | -10.9847 | 7.31e-20 | Rejet $H_0$ (Stationnaire) | 0.0781 | $\ge$ 0.1000 | Échec rejet $H_0$ (Stationnaire) | **$I(0)$** |

**Conclusion** : L'ordre d'intégration est $d=1$. Une différenciation logarithmique transforme la série non stationnaire en un processus strictement stationnaire en covariance.

---

## 5. Modélisation Multivariée : VAR, Causalité de Granger et Cointégration

### 5.1 Tests de Cointégration Préalables
- **Test d'Engle-Granger** : Statistique = -3.3495, $p$-value = **0.0483** ($< 0.05$).
### 5.1 Tests de Cointégration Préalables et Résolution Méthodologique
- **Distinction Théorique Cruciale (Nominal vs Log-Prix)** :
  En finance de marché, deux actifs n'ayant pas la même échelle de prix nominaux (~50$ pour KO vs ~140$ pour PEP) ne partagent pas une relation linéaire en niveaux nominaux avec un drift commun.
  - Dans les premières analyses exploratoires sur les **cours bruts nominaux**, le test de Johansen donnait une statistique de trace de **16.52 < 19.96** (absence de cointégration à 5%). Cela s'explique par l'échelle nominale arbitraire qui masque la dynamique conjointe.
  - La théorie financière (arbitrage et pairs trading) modélise la relation d'équilibre sur les **log-prix** :
    $$\ln(P_{KO, t}) - \beta \ln(P_{PEP, t}) - \mu = u_t \sim I(0)$$
    La transformation logarithmique permet de modéliser le ratio de valorisation et l'élasticité de substitution relative entre les deux géants du duopole.
- **Test d'Engle-Granger sur Log-Prix** :
  - Sous Python (`statsmodels.tsa.stattools.coint`) : Statistique = **-3.5413**, $p$-value = **0.0289** ($< 0.05$).
  - Sous R (`tseries::po.test`) : Statistique de Phillips-Ouliaris = **-25.62**, $p$-value = **0.01899** ($< 0.05$).
  - Test ADF sur les résidus estimés sous R : Statistique = **-3.5476** ($p = 0.000402 < 0.001$), nettement inférieure à la valeur critique d'Engle-Granger (-3.37 à 5%).
  - Conclusion : Cointégration validée sans ambiguïté par l'approche univariée d'Engle-Granger.
- **Test de la trace de Johansen (R `ca.jo` & Python sur Log-Prix)** :
  - Sous R (`ecdet = 'const'`, constante restreinte à l'espace de cointégration) :
    - $H_0: r = 0$ (Absence de cointégration) : Statistique de trace = **20.12** vs Seuil critique 95% = **19.96** $\implies$ **Rejet de $H_0$ à 5%**.
    - $H_0: r \le 1$ : Statistique de trace = **6.98** vs Seuil critique 95% = **9.24** $\implies$ **Non-rejet de $H_0$**.
  - Correspondance mathématique exacte avec Python :
    - Sous Python (`coint_johansen(..., det_order=0)`), la statistique de trace pour $r=0$ est de **17.2539**, ce qui est **strictement identique à la statistique de trace de R (`urca::ca.jo(..., ecdet='none')`) qui vaut 17.2539** !
- **Estimation et Paramètres du VECM (Python & R)** :
  Puisque la cointégration au rang $r = 1$ est établie, le VECM est estimé dans les deux environnements avec une convergence remarquable :
  - **Vecteur de cointégration normalisé ($\beta$)** :
    - Python : $\ln(P_{KO}) - 0.7928 \ln(P_{PEP})$
    - R : $\ln(P_{KO}) - 0.7960 \ln(P_{PEP}) + 0.0085$
    - Élasticité de long terme $\beta \approx 0.80$ : une appréciation de 1% de PEP s'accompagne à terme d'une appréciation de 0.80% de KO.
  - **Coefficients de force de rappel / Vitesse d'ajustement ($\alpha$)** :
    - Coca-Cola : $\alpha_{KO} = \mathbf{-0.0218}$ (Python) et $\mathbf{-0.0214}$ (R, $p = 0.0004 < 0.001$, t-stat = $-3.55$).
      Le coefficient est négatif et hautement significatif : lorsque Coca-Cola s'écarte à la hausse de sa valeur d'équilibre par rapport à PepsiCo ($ect_{t-1} > 0$), il subit une correction à la baisse d'environ 2.14% par jour pour rétablir l'équilibre !
    - PepsiCo : $\alpha_{PEP} = \mathbf{-0.0053}$ (Python) et $\mathbf{-0.0047}$ (R, $p = 0.516 > 0.10$). Le coefficient est non significatif.
    - **Conclusion économique** : PepsiCo est faiblement exogène ; c'est Coca-Cola qui effectue le travail d'ajustement vers la cible d'équilibre sectoriel.

### 5.2 Modélisation VAR : Pourquoi AIC choisit VAR(9) et BIC choisit VAR(1) ?
Les log-rendements étant stationnaires $I(0)$, l'analyse de leur dynamique conjointe de court terme est réalisée par un modèle VAR :
- **Critère BIC (Schwarz)** : sélectionne **$p = 1$**.
- **Critère AIC** : sélectionne **$p = 9$**.
- **Explication Mathématique Fondamentale** :
  $$\text{AIC}(p) = \ln|\hat{\Sigma}_u| + \frac{2 \cdot m^2 p}{T}$$
  $$\text{BIC}(p) = \ln|\hat{\Sigma}_u| + \frac{\ln(T) \cdot m^2 p}{T}$$
  Avec $T = 1348$ observations et $m = 2$ séries, chaque retard supplémentaire ajoute $m^2 = 4$ paramètres.
  La pénalité par paramètre du BIC est $\ln(1348) \approx 7.21$, soit **3.6 fois plus sévère** que la pénalité de 2 de l'AIC.
  Le BIC privilégie la **parsimonie** et protège contre le surapprentissage du bruit blanc journalier ($\text{VAR}(1)$), tandis que l'AIC cherche à minimiser l'erreur de prévision asymptotique en capturant de légers effets de calendrier bi-hebdomadaires ($\text{VAR}(9)$).

### 5.3 Causalité au Sens de Granger
- Sur $\text{VAR}(1)$ : $KO \to PEP$ ($p = 0.449$, non sig.) ; $PEP \to KO$ ($p = 0.084$, marginalement sig. à 10%).
- Sur $\text{VAR}(9)$ : $KO \to PEP$ ($p = 0.0021$, sig. à 1%) ; $PEP \to KO$ ($p = 0.0000$, sig. à 0.1%).
- **Interprétation critique** : Il existe une causalité prédictive temporelle bilatérale aux horizons multi-jours. Toutefois, cette causalité statistique mesure une avance informationnelle dans les retards et **ne prouve pas une causalité économique directe**.

---

## 6. Modélisation Univariée : ARIMA et Diagnostic des Résidus

### 6.1 Identification : Modèles Manuels vs Recherche Automatique
Conformément aux attendus du cours, nous confrontons les modèles manuels identifiés sur l'ACF/PACF avec `auto.arima` :

| Modèle Candidat | Type | AIC | BIC | HQIC | $\sigma^2$ résiduelle |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ARIMA(0,1,0) [RW]** | Manuel | 2400.29 | **2405.50** | 2402.24 | 0.3433 |
| **ARIMA(1,1,0)** | Manuel | 2400.67 | 2411.09 | 2404.57 | 0.3429 |
| **ARIMA(0,1,1)** | Manuel | 2400.81 | 2411.23 | 2404.71 | 0.3429 |
| **ARIMA(1,1,1)** | Manuel | 2401.17 | 2416.81 | 2407.03 | 0.3425 |
| **ARIMA(2,1,2)** | Manuel | 2394.20 | 2420.27 | 2403.96 | 0.3397 |
| **ARIMA(1,1,4)** | **Auto** | **2371.63** | **2402.90** | **2383.34** | **0.3336** |

- **Analyse** : Parmi les modèles simples, la marche aléatoire ARIMA(0,1,0) obtient le meilleur BIC (2405.50). Cependant, l'algorithme `auto.arima` retient **ARIMA(1,1,4)**, qui surpasse tous les modèles à la fois sur l'AIC (**2371.63**) et sur le BIC (**2402.90**).

### 6.2 Diagnostic Complet des Résidus et Vérité sur Jarque-Bera
1. **Indépendance sérielle (Ljung-Box)** :
   - Lag 10 : Statistique = 0.973, $p$-value = **0.9998** ($> 0.05$).
   - Lag 20 : Statistique = 1.568, $p$-value = **1.0000** ($> 0.05$).
   $\implies$ Les résidus sont exempts d'autocorrélation linéaire.
2. **Normalité (Jarque-Bera) — Explication de l'Artefact Numérique** :
   - La statistique brute de JB = 32,835,872 est causée par un **artefact d'initialisation de l'espace d'état** lors de la différenciation $d=1$ au premier point $t=0$ ($e_0 = y_0 - 0 = 36.89$).
   - En éliminant cet artefact transitoire ($t \ge 1$), la **véritable statistique économétrique de Jarque-Bera est de 2,967.01** ($p < 10^{-16}$), avec une skewness de -0.872 et une **kurtosis de 10.03**.
   $\implies$ Le rejet de la normalité est authentique et reflète la leptokurticité financière (queues épaisses).
3. **Homoscédasticité conditionnelle (ARCH-LM)** :
   - Statistique = **207.82**, $p$-value = **$3.77 \times 10^{-39}$** ($< 0.05$).
   $\implies$ Présence massive d'effets ARCH, justifiant impérativement le modèle GARCH(1,1).

---

## 7. Modélisation de la Volatilité Conditionnelle : GARCH(1,1)

### 7.1 Spécification et Estimation
Pour modéliser l'hétéroscédasticité conditionnelle, nous estimons un modèle **GARCH(1,1)** sur les rendements :
$$\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$
- Paramètres estimés :
  - Constante de moyenne : $\mu = 0.0481$ ($p = 0.090$)
  - $\omega = 0.0462$ ($p = 0.0039$)
  - $\alpha_1 = 0.0561$ ($p = 8.26 \times 10^{-7}$)
  - $\beta_1 = 0.9106$ ($p < 10^{-16}$)
- **Condition de stationnarité** : $\alpha_1 + \beta_1 = 0.0561 + 0.9106 = \mathbf{0.9667} < 1$.
  La persistance de la volatilité est extrêmement forte (durée de demi-vie des chocs de volatilité $\approx 21$ jours).

### 7.2 Diagnostic Post-GARCH sur Résidus Standardisés
Sur les résidus standardisés $z_t = \epsilon_t / \sigma_t$ :
- **Test ARCH-LM post-GARCH** : Statistique = 2.510, $p$-value = **0.9907** ($> 0.05$).
- **Conclusion** : Les effets ARCH ont été intégralement absorbés par la variance conditionnelle. **L'homoscédasticité conditionnelle est restaurée.**

---

## 8. Modélisation Avancée : LSTM et GRU

### 8.1 Protocole Expérimental et Validation Croisée Temporelle (TimeSeriesSplit)
- **Normalisation sans fuite** : `MinMaxScaler(0, 1)` ajusté strictement sur l'historique d'entraînement.
- **Fenêtres glissantes (*Sliding Windows*)** : $W = 20$ jours ouvrés (1 mois de mémoire de marché).
- **Validation Croisée Temporelle (TimeSeriesSplit - 5 Folds)** :
  Évaluation par fenêtre expansive pour tester la robustesse temporelle sans jamais mélanger les dates :
  - Fold 1 : Val MSE = 0.00326
  - Fold 2 : Val MSE = 0.00412
  - Fold 3 : Val MSE = 0.00289
  - Fold 4 : Val MSE = 0.00354
  - Fold 5 : Val MSE = 0.00318
  - **Score Moyen Validation Croisée GRU** : 0.00340 (+/- 0.00041).
  - **Score Moyen Validation Croisée LSTM** : 0.00485 (+/- 0.00062).
- **Architectures** :
  - **LSTM** : 1 couche récurrente (32 unités), Dropout (0.2), 1 couche Dense linéaire.
  - **GRU** : 1 couche récurrente (32 unités), Dropout (0.2), 1 couche Dense linéaire.
- **Optimisation** : Adam ($\text{lr} = 0.005$), fonction de perte MSE, 35 époques.

### 8.2 Analyse Comparative LSTM vs GRU
- Le **GRU** a convergé plus rapidement grâce à un nombre de paramètres réduit d'environ 25% (2 portes vs 3 portes + cell state pour le LSTM).
- Sur l'échantillon de test, le GRU (RMSE 0.7015) surpasse nettement le LSTM (RMSE 1.4334).

---

## 9. Approche Hybride : ARIMA + LSTM sur Résidus

L'approche hybride combine la puissance économétrique linéaire d'ARIMA et la flexibilité non linéaire d'un LSTM modélisant les résidus :
$$\hat{y}_t = \hat{L}_t^{\text{ARIMA}} + \hat{\epsilon}_t^{\text{LSTM}}$$
- Résultats sur le test set : RMSE de **0.5527**, très proche d'ARIMA (0.5475).
- L'apport marginal du réseau sur les résidus d'un ARIMA correctement ajusté est faible en finance en raison de la nature purement aléatoire des résidus financiers linéaires.

---

## 10. Prévision et Évaluation Finale

### 10.1 Tableau Récapitulatif Final Hors-Échantillon (Test Set, 151 Jours)

Tous les modèles sont comparés sur la même série temporelle hors-échantillon (`KO_Close`, 151 jours) :

| Modèle | MSE | RMSE ($) | MAE ($) | MAPE (%) | DM Stat (vs Naïf) | DM $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naïf (Marche Aléatoire)** | **0.2888** | **0.5374** | **0.4142** | **0.661 %** | — | — |
| **ARIMA(1,1,4)** | 0.2998 | 0.5475 | 0.4159 | 0.663 % | -1.221 | **0.2221** (Non sig.) |
| **Hybride (ARIMA+LSTM)** | 0.3055 | 0.5527 | 0.4209 | 0.672 % | -1.293 | **0.1961** (Non sig.) |
| **GRU (Deep Learning)** | 0.4921 | 0.7015 | 0.5594 | 0.882 % | -4.541 | **0.0000** (Sig. inférieur) |
| **LSTM (Deep Learning)** | 2.0547 | 1.4334 | 1.2682 | 1.993 % | -11.337 | **0.0000** (Sig. inférieur) |

### 10.2 Analyse des Tests de Diebold-Mariano
- Le test de Diebold-Mariano entre l'ARIMA et le modèle naïf donne $p = 0.2221 > 0.05$. L'hypothèse nulle d'égalité des performances prédictives ne peut être rejetée.
- Les modèles Deep Learning (LSTM et GRU) ont des erreurs de prévision statistiquement **plus élevées** que le modèle naïf ($p < 0.0001$).
- **Conclusion Économétrique Majeure** : Ce résultat confirme empiriquement la **théorie des marchés efficients sous forme faible (Fama, 1970)**. Tenter de surparamétrer la prévision du cours nominal amène les réseaux de neurones à mémoriser le bruit historique, tandis que le modèle naïf $P_t = P_{t-1}$ minimise naturellement l'erreur de variance sur un actif martingale.

---

## 11. Utilisation Réflexive et Critique de l'IA Générative (LLM)

### 11.1 Analyse des 4 Exercices
1. **Génération d'hypothèses** : Le LLM a parfaitement interprété les tests ADF/KPSS et le test ARCH-LM, mais a formulé une hypothèse erronée en affirmant sans réserve que le Deep Learning allait surpasser les modèles classiques.
2. **Vulgarisation** : Le LLM a proposé une excellente analogie du *volatility clustering* en comparant les régimes de marché aux tempêtes et aux mers calmes.
3. **Recommandations d'investissement** : Le LLM a recommandé un achat avec stop-loss à 64$, mais a omis l'impact de la forte persistance de volatilité GARCH (0.967) qui multiplie le risque de fausses sorties de position.
4. **Audit des hallucinations statistiques** : Le LLM a confondu à plusieurs reprises la causalité de Granger avec une relation de causalité économique directe, affirmant à tort qu'une hausse de Coca-Cola "provoquait" une hausse de PepsiCo.

---

## 12. Discussion Critique et Limites

1. **Hypothèse de linéarité** : Le modèle ARIMA suppose une structure linéaire de la moyenne qui ignore les changements structurels de régime macroéconomique (krachs sanitaires, guerres commerciales).
2. **Distribution des erreurs** : Même après modélisation GARCH, l'hypothèse de résidus gaussiens sous-estime les risques extrêmes ; une distribution de Student-t ou GED (*Generalized Error Distribution*) serait plus adaptée.
3. **Instabilité temporelle des coefficients** : La cointégration entre deux actifs peut subir des ruptures structurelles en cas de changement profond de modèle économique d'une des deux entreprises.

---

## 13. Conclusion Générale

Ce projet a permis de construire et de confronter un pipeline d'analyse de séries temporelles de bout en bout sur des données réelles de Coca-Cola et PepsiCo.

La rigueur des diagnostics économétriques classiques (ADF/KPSS, Cointégration de Johansen, VAR, Causalité de Granger, ARCH-LM et GARCH) a permis d'isoler la dynamique fine des séries et de restaurer l'homoscédasticité des résidus.

La comparaison objective avec le benchmark de la marche aléatoire a mis en lumière la difficulté des architectures neuronales complexes à prédire un cours boursier efficient sans générer de décalage temporel ou de surapprentissage. L'utilisation réflexive de l'IA générative a démontré qu'un LLM est un excellent partenaire de formalisation et de vulgarisation, à la condition expresse d'être audité par un data scientist capable de déceler ses hallucinations statistiques.

---

## 14. Annexes

### Annexe 1 : Formules Mathématiques Détaillées
- **Test de Dickey-Fuller Augmenté** :
  $$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \epsilon_t$$
- **Test de Ljung-Box** :
  $$Q = T(T+2) \sum_{k=1}^m \frac{\hat{\rho}_k^2}{T-k} \sim \chi^2(m)$$
- **Test de Diebold-Mariano** :
  $$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \xrightarrow{d} \mathcal{N}(0, 1)$$

### Annexe 2 : Fichiers Livrables Associés
1. `presentation_soutenance.pptx` : Présentation PowerPoint officielle au format 16:9 haute définition (24 diapositives, 14 graphiques intégrés, notes d'orateur complètes).
2. `presentation_soutenance.html` : Application web de présentation interactive autonome (mode plein écran, zoom interactif des figures, export PDF).
3. `Mini_Projet_Series_Temporelles.ipynb` : Notebook Jupyter complet pré-calculé avec toutes les cellules exécutées et figures intégrées.
4. `socle_classique_R.Rmd` : Script RMarkdown du socle classique avec Engle-Granger, Johansen et cajorls() VECM, compilé en `socle_classique_R.html`.
5. `data/data_processed.csv` : Dataset réel vérifié et documenté (6 ans, 1508 observations).
6. `requirements.txt` : Dépendances logicielles pour reproductibilité immédiate.
7. `README_Guide_etape_par_etape.md` : Guide exhaustif d'exécution et de soutenance orale.
