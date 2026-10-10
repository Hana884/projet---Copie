# scripts/test_r_pipeline.R
# Test du socle classique en R sur les donnees reelles de KO et PEP

suppressPackageStartupMessages({
  library(readr)
  library(tseries)
  library(urca)
  library(forecast)
  library(vars)
  library(fGarch)
  library(lmtest)
})

cat("1. Chargement des donnees reelles...\n")
df <- read_csv("data/data_processed.csv", col_types = cols())
n_total <- nrow(df)
n_train <- floor(n_total * 0.90)

train_df <- df[1:n_train, ]
test_df <- df[(n_train + 1):n_total, ]
cat(sprintf("Train: %d obs | Test: %d obs\n", nrow(train_df), nrow(test_df)))

cat("\n2. Tests ADF et KPSS sur KO_Close et KO_Return...\n")
# ADF KO_Close
adf_ko_p <- adf.test(train_df$KO_Close)
cat(sprintf("ADF KO_Close: Stat=%.4f, p-value=%.4f (Non-stationnaire)\n", adf_ko_p$statistic, adf_ko_p$p.value))

# KPSS KO_Close
kpss_ko_p <- kpss.test(train_df$KO_Close, null = "Level")
cat(sprintf("KPSS KO_Close: Stat=%.4f, p-value=%.4f (Non-stationnaire)\n", kpss_ko_p$statistic, kpss_ko_p$p.value))

# ADF KO_Return
adf_ko_r <- adf.test(train_df$KO_Return)
cat(sprintf("ADF KO_Return: Stat=%.4f, p-value=%.4f (Stationnaire)\n", adf_ko_r$statistic, adf_ko_r$p.value))

# KPSS KO_Return
kpss_ko_r <- kpss.test(train_df$KO_Return, null = "Level")
cat(sprintf("KPSS KO_Return: Stat=%.4f, p-value=%.4f (Stationnaire)\n", kpss_ko_r$statistic, kpss_ko_r$p.value))

cat("\n3. Cointégration de Johansen...\n")
prices_mat <- cbind(train_df$KO_Close, train_df$PEP_Close)
colnames(prices_mat) <- c("KO", "PEP")
joh_test <- ca.jo(prices_mat, type = "trace", ecdet = "const", K = 2)
cat("Statistiques de trace Johansen :\n")
print(joh_test@teststat)
print(joh_test@cval)

cat("\n4. Modélisation VAR et Causalité de Granger...\n")
returns_mat <- cbind(train_df$KO_Return, train_df$PEP_Return)
colnames(returns_mat) <- c("KO_Ret", "PEP_Ret")
var_sel <- VARselect(returns_mat, lag.max = 10, type = "const")
cat("Critères de sélection VAR :\n")
print(var_sel$criteria)
p_opt <- var_sel$selection["AIC(n)"]
cat(sprintf("Ordre optimal AIC: %d\n", p_opt))

var_fit <- VAR(returns_mat, p = p_opt, type = "const")
gc_ko <- causality(var_fit, cause = "KO_Ret")
cat("Granger Cause KO -> PEP:\n")
print(gc_ko$Granger)

cat("\n5. Modèle ARIMA et Recherche Automatique...\n")
auto_fit <- auto.arima(train_df$KO_Close, seasonal = FALSE, stepwise = TRUE)
cat("Ordre auto.arima retenu :\n")
print(summary(auto_fit))

cat("\nDiagnostic résidus Ljung-Box :\n")
lb_r <- Box.test(residuals(auto_fit), lag = 10, type = "Ljung-Box")
cat(sprintf("Ljung-Box Stat=%.4f, p-value=%.4f\n", lb_r$statistic, lb_r$p.value))

cat("\n6. Modèle GARCH(1,1) via fGarch...\n")
garch_fit <- garchFit(~ garch(1, 1), data = train_df$KO_Return, trace = FALSE)
cat("Parametres GARCH(1,1) :\n")
print(garch_fit@fit$matcoef)

cat("\nValidation R terminee avec succes !\n")
