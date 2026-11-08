# =============================================================================
# POGIT = Zero-Inflated Poisson (Logit + Poisson)
# Versión limpia: sin variables con multicolinealidad alta
# =============================================================================

library(pscl)
library(openxlsx)
library(dplyr)
library(car)

# -----------------------------------------------------------------------------
# 1. Cargar matrices
# >>> Debe coincidir con MIN_MONETARY_T1 usado en code2 (0 -> "_min0", 30 -> "_min30") <<<
TAG <- "_min30"

# -----------------------------------------------------------------------------
# A) Cargar matrices V, W
# -----------------------------------------------------------------------------
wb <- paste0("data/matrices/matrices_data", TAG, ".xlsx")

V_train <- read.xlsx(wb, sheet = "V_train")
V_test  <- read.xlsx(wb, sheet = "V_test")
W_train <- read.xlsx(wb, sheet = "W_train")
W_test  <- read.xlsx(wb, sheet = "W_test")
Y_train <- as.numeric(read.xlsx(wb, sheet = "Y_train")[[1]])
Y_test  <- as.numeric(read.xlsx(wb, sheet = "Y_test")[[1]])

# Unir V + W
X_train <- cbind(V_train, W_train)
X_test  <- cbind(V_test,  W_test)

# Eliminar columnas Intercept
intercept_cols <- grep("^Intercept", names(X_train), value = TRUE)
if (length(intercept_cols) > 0) {
  X_train <- X_train[, !names(X_train) %in% intercept_cols, drop = FALSE]
  X_test  <- X_test[,  !names(X_test)  %in% intercept_cols, drop = FALSE]
}

# -----------------------------------------------------------------------------
# 2. Limpiar nombres de columnas
# -----------------------------------------------------------------------------
names(X_train) <- make.names(names(X_train), unique = TRUE)
names(X_test)  <- make.names(names(X_test),  unique = TRUE)

common_cols <- intersect(names(X_train), names(X_test))
X_train <- X_train[, common_cols, drop = FALSE]
X_test  <- X_test[,  common_cols, drop = FALSE]

X_train <- as.data.frame(lapply(X_train, function(x) {
  x <- as.numeric(x)
  x[is.na(x)] <- 0
  x
}))
X_test <- as.data.frame(lapply(X_test, function(x) {
  x <- as.numeric(x)
  x[is.na(x)] <- 0
  x
}))

train_df <- data.frame(Y = Y_train, X_train, check.names = FALSE)
test_df  <- data.frame(Y = Y_test,  X_test,  check.names = FALSE)

# -----------------------------------------------------------------------------
# 3. Eliminar variables con multicolinealidad alta
# -----------------------------------------------------------------------------
# dollar_avg_12m y dollar_top1_12m tenían correlación 0.98 y VIF > 35
vars_to_drop <- c("dollar_avg_12m", "dollar_top1_12m")

train_df <- train_df[, !names(train_df) %in% vars_to_drop, drop = FALSE]
test_df  <- test_df[,  !names(test_df)  %in% vars_to_drop, drop = FALSE]

# -----------------------------------------------------------------------------
# 4. Construir la fórmula limpia
# -----------------------------------------------------------------------------
preds <- setdiff(names(train_df), "Y")

# Parte logit: solo RFM + n_states
logit_candidates <- c(
  "recency_days_2018_Samsung",
  "frequency_2018_Samsung",
  "monetary_2018_Samsung",
  "n_states"
)
logit_vars <- intersect(make.names(logit_candidates), preds)

if (length(logit_vars) == 0) {
  logit_vars <- preds
  message("No se encontraron las variables RFM/n_states. Se usarán todas en la parte logit.")
}

form_pogit <- as.formula(
  paste0(
    "Y ~ ", paste(preds, collapse = " + "),
    " | ", paste(logit_vars, collapse = " + ")
  )
)

print(form_pogit)

# -----------------------------------------------------------------------------
# 5. Verificar multicolinealidad después de la limpieza
# -----------------------------------------------------------------------------
vars_cor <- c(
  "recency_days_2018_Samsung",
  "frequency_2018_Samsung",
  "monetary_2018_Samsung",
  "dollar_avg12_top1_ratio",
  "n_states"
)
vars_cor <- intersect(vars_cor, names(train_df))

cor_mat <- cor(train_df[, vars_cor], use = "complete.obs")
print(round(cor_mat, 2))

glm_aux <- glm(Y ~ ., data = train_df[, c("Y", vars_cor)], family = poisson)
print(vif(glm_aux))

# -----------------------------------------------------------------------------
# 6. Ajustar el Pogit (Poisson)
# -----------------------------------------------------------------------------
set.seed(123)

pogit_model <- zeroinfl(
  formula = form_pogit,
  data    = train_df,
  dist    = "poisson",
  link    = "logit",
  method  = "BFGS"
)

summary(pogit_model)

# -----------------------------------------------------------------------------
# 7. Predicciones y métricas
# -----------------------------------------------------------------------------
pred_train <- predict(pogit_model, newdata = train_df, type = "response")
pred_test  <- predict(pogit_model, newdata = test_df,  type = "response")

rmse <- function(a, b) sqrt(mean((a - b)^2, na.rm = TRUE))
mae  <- function(a, b) mean(abs(a - b), na.rm = TRUE)

cat("\n=== Métricas TRAIN ===\n")
cat("RMSE:", round(rmse(Y_train, pred_train), 4), "\n")
cat("MAE :", round(mae(Y_train, pred_train), 4), "\n")

cat("\n=== Métricas TEST ===\n")
cat("RMSE:", round(rmse(Y_test, pred_test), 4), "\n")
cat("MAE :", round(mae(Y_test, pred_test), 4), "\n")

# -----------------------------------------------------------------------------
# 8. Guardar modelo
# -----------------------------------------------------------------------------
dir.create("data/matrices", showWarnings = FALSE, recursive = TRUE)
saveRDS(pogit_model, "data/matrices/pogit_model_samsung_clean.rds")
cat("\nModelo guardado en data/matrices/pogit_model_samsung_clean.rds\n")