suppressPackageStartupMessages({
  library(readr)
  library(urca)
  library(vars)
  library(tseries)
})

df <- read_csv("data/data_processed.csv", col_types = cols())
train_df <- df[1:floor(nrow(df) * 0.9), ]

cat("=== 1. ENGLE-GRANGER TEST IN R ===\n")
# Regression of KO_Log on PEP_Log with constant
eg_model_log <- lm(KO_LogPrice ~ PEP_LogPrice, data = train_df)
eg_resids_log <- residuals(eg_model_log)
# Phillips-Ouliaris test (standard Engle-Granger coint test in tseries / urca)
po_res <- po.test(cbind(train_df$KO_LogPrice, train_df$PEP_LogPrice))
print(po_res)

# Direct ADF on residuals without constant (Engle-Granger 2nd step)
adf_eg <- ur.df(eg_resids_log, type = "none", lags = 1)
print(summary(adf_eg))

cat("\n=== 2. JOHANSEN ON LOG PRICES IN R ===\n")
log_mat <- cbind(train_df$KO_LogPrice, train_df$PEP_LogPrice)
colnames(log_mat) <- c("KO_log", "PEP_log")

# ecdet = "const" (constant restricted to cointegration space)
jo_const <- ca.jo(log_mat, type = "trace", ecdet = "const", K = 2)
cat("--- Trace test (ecdet = 'const') ---\n")
print(jo_const@teststat)
print(jo_const@cval)

# ecdet = "none" (no constant)
jo_none <- ca.jo(log_mat, type = "trace", ecdet = "none", K = 2)
cat("\n--- Trace test (ecdet = 'none') ---\n")
print(jo_none@teststat)
print(jo_none@cval)

# ecdet = "trend" (linear trend in cointegration relation)
jo_trend <- ca.jo(log_mat, type = "trace", ecdet = "trend", K = 2)
cat("\n--- Trace test (ecdet = 'trend') ---\n")
print(jo_trend@teststat)
print(jo_trend@cval)

cat("\n=== 3. VECM ESTIMATION VIA cajorls() IN R ===\n")
vecm_r <- cajorls(jo_const, r = 1)
print(vecm_r)

cat("\nCointegrating vector beta:\n")
print(jo_const@V[, 1])
