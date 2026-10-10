import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank, VECM

df = pd.read_csv("data/data_processed.csv", index_col=0, parse_dates=True)
n_train = int(len(df) * 0.90)
train_df = df.iloc[:n_train]

ko_p = train_df['KO_Close']
pep_p = train_df['PEP_Close']
ko_log = train_df['KO_LogPrice']
pep_log = train_df['PEP_LogPrice']

print("=== 1. ENGLE-GRANGER TEST ===")
# trend='c' (constant, default), trend='ct' (constant + trend)
eg_c_nom = coint(ko_p, pep_p, trend='c')
eg_ct_nom = coint(ko_p, pep_p, trend='ct')
eg_c_log = coint(ko_log, pep_log, trend='c')
eg_ct_log = coint(ko_log, pep_log, trend='ct')

print(f"Nominal trend='c'  : Stat={eg_c_nom[0]:.4f}, p={eg_c_nom[1]:.4f}, crit={eg_c_nom[2]}")
print(f"Nominal trend='ct' : Stat={eg_ct_nom[0]:.4f}, p={eg_ct_nom[1]:.4f}, crit={eg_ct_nom[2]}")
print(f"Log     trend='c'  : Stat={eg_c_log[0]:.4f}, p={eg_c_log[1]:.4f}, crit={eg_c_log[2]}")
print(f"Log     trend='ct' : Stat={eg_ct_log[0]:.4f}, p={eg_ct_log[1]:.4f}, crit={eg_ct_log[2]}")

print("\n=== 2. STATSMODELS SELECT_COINT_RANK ===")
for det in [-1, 0, 1]:
    for k in [1, 2]:
        r_trace = select_coint_rank(train_df[['KO_LogPrice', 'PEP_LogPrice']], det_order=det, k_ar_diff=k, method='trace', signif=0.05)
        r_maxeig = select_coint_rank(train_df[['KO_LogPrice', 'PEP_LogPrice']], det_order=det, k_ar_diff=k, method='maxeig', signif=0.05)
        print(f"det_order={det:2d}, k_ar_diff={k}: rank_trace={r_trace.rank}, rank_maxeig={r_maxeig.rank}")
