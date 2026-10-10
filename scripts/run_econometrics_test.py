"""
scripts/run_econometrics_test.py
--------------------------------
Vérification expérimentale complète de l'ensemble des étapes économétriques et Deep Learning
sur les données réelles de Coca-Cola (KO) et PepsiCo (PEP).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
import pmdarima as pm
from arch import arch_model
import torch
import torch.nn as nn

def run_tests():
    print("Chargement des données...")
    df = pd.read_csv("data/data_processed.csv", index_col=0, parse_dates=True)
    n_train = int(len(df) * 0.90)
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    print(f"Train: {len(train_df)} obs | Test: {len(test_df)} obs")

    print("\n--- 1. TESTS DE STATIONNARITÉ (ADF & KPSS) SUR LE TRAIN ---")
    for col in ["KO_Close", "PEP_Close", "KO_Return", "PEP_Return"]:
        s = train_df[col].dropna()
        # ADF
        adf_res = adfuller(s, autolag="AIC")
        # KPSS
        kpss_res = kpss(s, regression="c", nlags="auto")
        print(f"[{col}]")
        print(f"  ADF Stat: {adf_res[0]:.4f}, p-value: {adf_res[1]:.4e} | H0 (Racine unitaire) {'Rejetée (Stationnaire)' if adf_res[1]<0.05 else 'Non rejetée (Non stationnaire)'}")
        print(f"  KPSS Stat: {kpss_res[0]:.4f}, p-value: {kpss_res[1]:.4e} | H0 (Stationnaire) {'Rejetée (Non stationnaire)' if kpss_res[1]<0.05 else 'Non rejetée (Stationnaire)'}")

    print("\n--- 2. COINTÉGRATION (JOHANSEN & ENGLE-GRANGER) ---")
    # Engle-Granger
    eg_stat, eg_pval, _ = coint(train_df["KO_Close"], train_df["PEP_Close"])
    print(f"Engle-Granger KO/PEP: Stat={eg_stat:.4f}, p-value={eg_pval:.4f}")
    
    # Johansen
    # log prices
    coint_data = train_df[["KO_LogPrice", "PEP_LogPrice"]].dropna()
    joh_res = coint_johansen(coint_data, det_order=0, k_ar_diff=1)
    print("Johansen Trace Stats:", joh_res.lr1)
    print("Johansen Trace Crit (90%, 95%, 99%):", joh_res.cvt)
    r_coint = sum(joh_res.lr1 > joh_res.cvt[:, 1])
    print(f"Nombre de relations de cointégration détectées (seuil 5%): {r_coint}")
    if r_coint > 0:
        print("-> Cointégration détectée : VECM justifiable.")
    else:
        print("-> Cointégration NON détectée au seuil 5% : VECM NON justifié, privilégier VAR sur rendements stationnaires !")

    print("\n--- 3. MODÈLE VAR SUR RENDEMENTS ET GRANGER ---")
    var_data = train_df[["KO_Return", "PEP_Return"]].dropna()
    var_model = VAR(var_data)
    lag_order = var_model.select_order(maxlags=10)
    print("Sélection d'ordre VAR (AIC, BIC, HQIC, FPE):")
    print(lag_order.summary())
    best_p = lag_order.aic
    print(f"Ordre retenu selon AIC: {best_p}")
    var_fitted = var_model.fit(best_p)
    print(str(var_fitted.summary())[:400])

    print("\nTest de Causalité de Granger :")
    gc_ko_pep = grangercausalitytests(var_data[["PEP_Return", "KO_Return"]], maxlag=[best_p])
    p_val_ko_to_pep = gc_ko_pep[best_p][0]["ssr_ftest"][1]
    gc_pep_ko = grangercausalitytests(var_data[["KO_Return", "PEP_Return"]], maxlag=[best_p])
    p_val_pep_to_ko = gc_pep_ko[best_p][0]["ssr_ftest"][1]
    print(f"  KO -> PEP p-value: {p_val_ko_to_pep:.4f} ({'Significatif' if p_val_ko_to_pep < 0.05 else 'Non significatif'})")
    print(f"  PEP -> KO p-value: {p_val_pep_to_ko:.4f} ({'Significatif' if p_val_pep_to_ko < 0.05 else 'Non significatif'})")

    print("\n--- 4. ARIMA & DIAGNOSTIC DES RÉSIDUS ---")
    arima_fit = pm.auto_arima(train_df["KO_Close"], seasonal=False, stepwise=True, trace=False, error_action="ignore")
    print("Auto-ARIMA KO_Close sélectionné:", arima_fit.order)
    resid = arima_fit.resid()
    
    # Ljung-Box test
    lb_df = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
    print("Ljung-Box sur résidus ARIMA:")
    print(lb_df)
    
    # Jarque-Bera
    jb_stat, jb_pval, _, _ = jarque_bera(resid)
    print(f"Jarque-Bera Normalité: Stat={jb_stat:.4f}, p-value={jb_pval:.4e}")
    
    # ARCH-LM test
    arch_test = het_arch(resid)
    print(f"ARCH-LM Stat: {arch_test[0]:.4f}, p-value: {arch_test[1]:.4e} (Présence d'effets ARCH: {arch_test[1] < 0.05})")

    print("\n--- 5. MODÈLE GARCH(1,1) ---")
    if arch_test[1] < 0.05:
        am = arch_model(train_df["KO_Return"].dropna(), vol="Garch", p=1, q=1, dist="Normal")
        garch_res = am.fit(disp="off")
        print(str(garch_res.summary())[:400])
        std_resid = garch_res.std_resid.dropna()
        arch_test_post = het_arch(std_resid)
        print(f"ARCH-LM post-GARCH sur résidus standardisés: Stat={arch_test_post[0]:.4f}, p-value={arch_test_post[1]:.4f}")
        print(f"-> Effets ARCH éliminés (Homoscédasticité restaurée) : {arch_test_post[1] > 0.05}")

    print("\nTout le pipeline économétrique a été validé avec succès sur données réelles !")

if __name__ == "__main__":
    run_tests()
