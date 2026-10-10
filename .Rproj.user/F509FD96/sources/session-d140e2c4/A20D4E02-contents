"""
scripts/test_dl_and_evaluation.py
---------------------------------
Entraînement de LSTM, GRU, Modèle Hybride (ARIMA+LSTM) et Benchmarking rigoureux.
Évaluation hors-échantillon sur le test set (151 jours).
Génération du tableau comparatif unifié et du test de Diebold-Mariano.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
import scipy.stats as stats

# Fixer la graine aléatoire pour une reproductibilité absolue
torch.manual_seed(42)
np.random.seed(42)

def diebold_mariano_test(e1, e2, h=1):
    """
    Test de Diebold-Mariano pour comparer la précision prédictive de deux modèles.
    H0: Les deux modèles ont la même précision prédictive (E[d_t] = 0)
    H1: Les deux modèles ont une précision prédictive différente.
    """
    d = (e1 ** 2) - (e2 ** 2)
    mean_d = np.mean(d)
    n = len(d)
    
    # Variance autocorrélée (Newey-West / Bartlett pour horizon h)
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0
    for lag in range(1, h):
        weight = 1 - (lag / h)
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        gamma_sum += 2 * weight * cov
    var_d = (gamma0 + gamma_sum) / n
    if var_d <= 0:
        var_d = 1e-12
        
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

class TimeSeriesRNN(nn.Module):
    def __init__(self, rnn_type="LSTM", input_size=1, hidden_size=32, num_layers=1, dropout=0.2):
        super(TimeSeriesRNN, self).__init__()
        self.rnn_type = rnn_type
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def create_sliding_windows(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def main():
    print("Chargement des données...")
    df = pd.read_csv("data/data_processed.csv", index_col=0, parse_dates=True)
    target_col = "KO_Close"
    
    n_total = len(df)
    n_train = int(n_total * 0.90)
    train_series = df[target_col].iloc[:n_train].values
    test_series = df[target_col].iloc[n_train:].values
    
    window_size = 20
    print(f"Total: {n_total}, Train: {n_train}, Test: {len(test_series)}, Window Size: {window_size}")
    
    # 1. Benchmark Naif (Random Walk: y_hat_t = y_{t-1})
    # Pour le test set, prévision = valeur de la veille
    # Pour le premier point du test, on utilise le dernier point du train
    full_series = df[target_col].values
    naive_preds = full_series[n_train - 1 : -1]
    y_test_true = test_series
    
    # 2. Modèle ARIMA(1,1,4)
    print("Ajustement ARIMA(1,1,4)...")
    arima_model = ARIMA(train_series, order=(1, 1, 4))
    arima_res = arima_model.fit()
    
    # Prévisions 1-step rolling sur le test set (ou prévision out-of-sample)
    # Pour être rigoureux et comparable au deep learning sur 1-step rolling :
    history = list(train_series)
    arima_preds = []
    for t in range(len(test_series)):
        # prévision 1 pas en avant
        model_t = ARIMA(history, order=(1, 1, 4))
        # Pour accélérer sans refit complet de l'espace d'état : filter
        res_t = model_t.filter(arima_res.params)
        y_hat = res_t.forecast(steps=1)[0]
        arima_preds.append(y_hat)
        history.append(test_series[t])
    arima_preds = np.array(arima_preds)
    
    # 3. Préparation Deep Learning (LSTM & GRU)
    # Scaling fitted ONLY on train !
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()
    
    # Pour prédire test_series[0], on a besoin des `window_size` points précédents du train
    # On concatène les `window_size` derniers points de train avec test
    full_scaled = scaler.transform(full_series.reshape(-1, 1)).flatten()
    
    # Split train en train et val temporels (walk-forward/validation temporelle : 85% train, 15% val au sein du train)
    val_split = int(len(train_scaled) * 0.85)
    train_sub = train_scaled[:val_split]
    val_sub = train_scaled[val_split - window_size:]
    
    X_train_sub, y_train_sub = create_sliding_windows(train_sub, window_size)
    X_val_sub, y_val_sub = create_sliding_windows(val_sub, window_size)
    
    # Train complet pour le test final
    X_train_full, y_train_full = create_sliding_windows(train_scaled, window_size)
    
    # Test windows : de n_train - window_size à la fin
    test_slice = full_scaled[n_train - window_size:]
    X_test, y_test_scaled = create_sliding_windows(test_slice, window_size)
    
    print(f"X_train shape: {X_train_full.shape}, X_test shape: {X_test.shape}")
    
    def train_rnn(rnn_type="LSTM", epochs=40, lr=0.005):
        model = TimeSeriesRNN(rnn_type=rnn_type, input_size=1, hidden_size=32, num_layers=1, dropout=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_full, dtype=torch.float32).unsqueeze(-1),
                          torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(-1)),
            batch_size=32, shuffle=False # pas de shuffle pour respecter l'ordre temporel
        )
        
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)).squeeze().numpy()
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return model, preds
    
    print("Entraînement LSTM...")
    lstm_model, lstm_preds = train_rnn("LSTM", epochs=35, lr=0.005)
    
    print("Entraînement GRU...")
    gru_model, gru_preds = train_rnn("GRU", epochs=35, lr=0.005)
    
    # 4. Modèle Hybride (ARIMA + LSTM sur les résidus)
    print("Entraînement Modèle Hybride (ARIMA + LSTM sur résidus)...")
    # Résidus de l'ARIMA sur le train
    arima_train_preds = arima_res.fittedvalues
    # Ajuster la taille (car premier lag diff)
    residuals_train = train_series - arima_train_preds
    
    res_scaler = MinMaxScaler(feature_range=(-1, 1))
    res_scaled = res_scaler.fit_transform(residuals_train.reshape(-1, 1)).flatten()
    X_res, y_res = create_sliding_windows(res_scaled, window_size)
    
    res_model = TimeSeriesRNN(rnn_type="LSTM", input_size=1, hidden_size=16, num_layers=1, dropout=0.1)
    res_opt = torch.optim.Adam(res_model.parameters(), lr=0.005)
    res_crit = nn.MSELoss()
    res_loader = DataLoader(
        TensorDataset(torch.tensor(X_res, dtype=torch.float32).unsqueeze(-1),
                      torch.tensor(y_res, dtype=torch.float32).unsqueeze(-1)),
        batch_size=32, shuffle=False
    )
    for _ in range(25):
        for bx, by in res_loader:
            res_opt.zero_grad()
            l = res_crit(res_model(bx), by)
            l.backward()
            res_opt.step()
            
    # Résidus sur l'horizon test
    arima_resids_test = y_test_true - arima_preds
    full_resids = np.concatenate([residuals_train, arima_resids_test])
    full_res_scaled = res_scaler.transform(full_resids.reshape(-1, 1)).flatten()
    test_res_slice = full_res_scaled[len(residuals_train) - window_size:]
    X_test_res, _ = create_sliding_windows(test_res_slice, window_size)
    
    res_model.eval()
    with torch.no_grad():
        pred_res_scaled = res_model(torch.tensor(X_test_res, dtype=torch.float32).unsqueeze(-1)).squeeze().numpy()
    pred_res = res_scaler.inverse_transform(pred_res_scaled.reshape(-1, 1)).flatten()
    hybrid_preds = arima_preds + pred_res

    # 5. Table comparative et métriques
    models_dict = {
        "Naïf (Marche Aléatoire)": naive_preds,
        "ARIMA(1,1,4)": arima_preds,
        "LSTM": lstm_preds,
        "GRU": gru_preds,
        "Hybride (ARIMA+LSTM)": hybrid_preds
    }
    
    res_rows = []
    e_naive = y_test_true - naive_preds
    
    for name, pred in models_dict.items():
        err = y_test_true - pred
        mse = np.mean(err ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(err))
        mape = np.mean(np.abs(err / y_test_true)) * 100.0
        
        # Test Diebold-Mariano vs Naive
        if name == "Naïf (Marche Aléatoire)":
            dm_stat, dm_p = np.nan, np.nan
        else:
            dm_stat, dm_p = diebold_mariano_test(e_naive, err)
            
        res_rows.append({
            "Modèle": name,
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MAPE (%)": round(mape, 3),
            "DM Stat (vs Naïf)": round(dm_stat, 3) if not np.isnan(dm_stat) else "-",
            "DM p-value": round(dm_p, 4) if not np.isnan(dm_p) else "-"
        })
        
    df_metrics = pd.DataFrame(res_rows)
    print("\n" + "=" * 80)
    print("TABLEAU RÉCAPITULATIF FINAL DES PERFORMANCES HORS-ÉCHANTILLON (TEST SET)")
    print("=" * 80)
    print(df_metrics.to_string(index=False))
    
    # Sauvegarde des prévisions du test
    df_preds = pd.DataFrame({
        "Date": df.index[n_train:],
        "True": y_test_true,
        "Naive": naive_preds,
        "ARIMA": arima_preds,
        "LSTM": lstm_preds,
        "GRU": gru_preds,
        "Hybrid": hybrid_preds
    }).set_index("Date")
    df_preds.to_csv("data/test_forecast_comparison.csv")
    print("\n-> Prévisions sauvegardées dans 'data/test_forecast_comparison.csv'.")

if __name__ == "__main__":
    main()
