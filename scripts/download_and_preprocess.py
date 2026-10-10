"""
scripts/download_and_preprocess.py
----------------------------------
Téléchargement et préparation rigoureuse des données financières réelles pour Coca-Cola (KO) et PepsiCo (PEP).
Conforme au cahier des charges du Mini-Projet Séries Temporelles (TEK-UP).

Caractéristiques :
- Source : Yahoo Finance (finance.yahoo.com via yfinance)
- Période : 2019-01-01 à 2024-12-31 (6 années pleines, > 5 ans)
- Fréquence : Journalière (cours de clôture des jours ouvrés de bourse NYSE/NASDAQ)
- Tickers : KO (Coca-Cola Co), PEP (PepsiCo, Inc.)
- Variables brutes : Open, High, Low, Close, Adj Close, Volume
- Indicateurs dérivés : Log-rendements (%), Volatilité réalisée glissante (20j annualisée),
  Moyennes Mobiles (SMA 20, SMA 50), RSI (14j), Ratio de prix KO/PEP
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

def compute_rsi(series, period=14):
    """Calcul du RSI (Relative Strength Index) sur 14 périodes sans fuite d'information future."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def main():
    print("=" * 70)
    print("1. TÉLÉCHARGEMENT DES DONNÉES RÉELLES (YAHOO FINANCE)")
    print("=" * 70)
    
    tickers = ["KO", "PEP"]
    start_date = "2019-01-01"
    end_date = "2024-12-31"
    
    print(f"-> Téléchargement de {tickers} de {start_date} à {end_date}...")
    df_raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    if df_raw.empty:
        raise RuntimeError("Échec du téléchargement Yahoo Finance : dataset vide.")
    
    print(f"-> Données brutes reçues : {df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes.")
    
    # Sauvegarde des données brutes avec MultiIndex conservé
    raw_path = os.path.join("data", "data_ko_pep_daily.csv")
    df_raw.to_csv(raw_path)
    print(f"-> Fichier brut sauvegardé dans '{raw_path}'.")
    
    print("\n" + "=" * 70)
    print("2. PRÉPARATION ET VÉRIFICATION DE L'ALIGNEMENT TEMPOREL")
    print("=" * 70)
    
    # Extraction des cours de clôture ajustée (Adj Close) et Volume
    # Utilisation de 'Close' ou 'Adj Close'
    # Dans Yahoo Finance récent, 'Close' peut être déjà ajusté ou distinct de 'Adj Close'
    has_adj = "Adj Close" in df_raw.columns.levels[0]
    price_col = "Adj Close" if has_adj else "Close"
    
    ko_price = df_raw[(price_col, "KO")].dropna()
    pep_price = df_raw[(price_col, "PEP")].dropna()
    ko_volume = df_raw[("Volume", "KO")].dropna()
    pep_volume = df_raw[("Volume", "PEP")].dropna()
    
    # Alignement sur les dates communes (intersection des jours de cotation effectifs)
    common_idx = ko_price.index.intersection(pep_price.index)
    ko_price = ko_price.loc[common_idx]
    pep_price = pep_price.loc[common_idx]
    ko_volume = ko_volume.loc[common_idx]
    pep_volume = pep_volume.loc[common_idx]
    
    df_processed = pd.DataFrame(index=common_idx)
    df_processed.index.name = "Date"
    
    # Prix de clôture ajustés
    df_processed["KO_Close"] = ko_price
    df_processed["PEP_Close"] = pep_price
    
    # Log-prix
    df_processed["KO_LogPrice"] = np.log(ko_price)
    df_processed["PEP_LogPrice"] = np.log(pep_price)
    
    # Log-rendements en pourcentage : r_t = 100 * ln(P_t / P_{t-1})
    df_processed["KO_Return"] = (np.log(ko_price) - np.log(ko_price.shift(1))) * 100.0
    df_processed["PEP_Return"] = (np.log(pep_price) - np.log(pep_price.shift(1))) * 100.0
    
    # Volumes échangés
    df_processed["KO_Volume"] = ko_volume
    df_processed["PEP_Volume"] = pep_volume
    
    # Moyennes Mobiles Simples (SMA 20 et SMA 50)
    df_processed["KO_SMA20"] = ko_price.rolling(window=20).mean()
    df_processed["KO_SMA50"] = ko_price.rolling(window=50).mean()
    df_processed["PEP_SMA20"] = pep_price.rolling(window=20).mean()
    df_processed["PEP_SMA50"] = pep_price.rolling(window=50).mean()
    
    # Volatilité réalisée glissante annualisée (fenêtre 20 jours ouvrés, racine(252))
    df_processed["KO_Vol20"] = df_processed["KO_Return"].rolling(window=20).std() * np.sqrt(252)
    df_processed["PEP_Vol20"] = df_processed["PEP_Return"].rolling(window=20).std() * np.sqrt(252)
    
    # RSI (14 jours)
    df_processed["KO_RSI14"] = compute_rsi(ko_price, period=14)
    df_processed["PEP_RSI14"] = compute_rsi(pep_price, period=14)
    
    # Ratio de prix et Spread de log-prix
    df_processed["Price_Ratio"] = ko_price / pep_price
    df_processed["Log_Spread"] = df_processed["KO_LogPrice"] - df_processed["PEP_LogPrice"]
    
    # Suppression de la première ligne (NaN due au diff pour les rendements)
    df_processed_clean = df_processed.dropna(subset=["KO_Return", "PEP_Return"]).copy()
    
    # Sauvegarde du dataset traité
    proc_path = os.path.join("data", "data_processed.csv")
    df_processed_clean.to_csv(proc_path)
    print(f"-> Dataset traité sauvegardé dans '{proc_path}' ({len(df_processed_clean)} observations).")
    
    print("\n" + "=" * 70)
    print("3. STATISTIQUES DESCRIPTIVES RÉELLES")
    print("=" * 70)
    stats_cols = ["KO_Close", "PEP_Close", "KO_Return", "PEP_Return"]
    desc = df_processed_clean[stats_cols].describe().T
    desc["skewness"] = df_processed_clean[stats_cols].skew()
    desc["kurtosis"] = df_processed_clean[stats_cols].kurtosis()
    print(desc[["count", "mean", "std", "min", "50%", "max", "skewness", "kurtosis"]])
    
    # Découpage 90% train / 10% test
    n_total = len(df_processed_clean)
    n_train = int(n_total * 0.90)
    n_test = n_total - n_train
    train_end = df_processed_clean.index[n_train - 1]
    test_start = df_processed_clean.index[n_train]
    
    print("\n" + "=" * 70)
    print("4. DÉCOUPAGE TEMPOREL CHRONOLOGIQUE (90% / 10%)")
    print("=" * 70)
    print(f"Total observations : {n_total}")
    print(f"Train (90%)        : {n_train} obs | Du {df_processed_clean.index[0].date()} au {train_end.date()}")
    print(f"Test  (10%)        : {n_test} obs | Du {test_start.date()} au {df_processed_clean.index[-1].date()}")
    print("-> Aucune fuite d'information (chronologie strictement préservée).")

if __name__ == "__main__":
    main()
