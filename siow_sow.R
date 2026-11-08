# =============================================================================
# Predecir SoW (y Siow) con V + W  — esquema en R
# =============================================================================
library(dplyr)
library(openxlsx)
library(pscl)      # opcional, si luego quieres ZIP sobre Siow
library(caret)     # métricas opcionales
library(tidyr)     # replace_na() (lo usa build_sow_siow)

# >>> Debe coincidir con MIN_MONETARY_T1 usado en code2 (0 -> "_min0", 30 -> "_min30") <<<
TAG <- "_min20"

# -----------------------------------------------------------------------------
# A) Cargar matrices V, W
# -----------------------------------------------------------------------------
wb <- paste0("data/matrices/matrices_data", TAG, ".xlsx")

V_train <- read.xlsx(wb, sheet = "V_train")
V_test  <- read.xlsx(wb, sheet = "V_test")
W_train <- read.xlsx(wb, sheet = "W_train")
W_test  <- read.xlsx(wb, sheet = "W_test")

drop_int <- function(df) df[, !grepl("^Intercept", names(df), ignore.case = TRUE), drop = FALSE]
V_train <- drop_int(V_train); V_test <- drop_int(V_test)
W_train <- drop_int(W_train); W_test <- drop_int(W_test)

X_train <- cbind(V_train, W_train)
X_test  <- cbind(V_test,  W_test)
names(X_train) <- make.names(names(X_train), unique = TRUE)
names(X_test)  <- make.names(names(X_test),  unique = TRUE)
common <- intersect(names(X_train), names(X_test))
X_train <- X_train[, common, drop = FALSE]
X_test  <- X_test[,  common, drop = FALSE]

to_num <- function(df) {
  as.data.frame(lapply(df, function(z) {
    z <- as.numeric(z); z[is.na(z)] <- 0; z
  }))
}
X_train <- to_num(X_train)
X_test  <- to_num(X_test)

# -----------------------------------------------------------------------------
# B) IDs train/test (DEBES tenerlos del split de code2)
# -----------------------------------------------------------------------------
# Opción recomendada: guardarlos al hacer el split
# write.csv(data.frame(response_id = train_ids), "data/matrices/ids_train.csv", row.names = FALSE)
# write.csv(data.frame(response_id = test_ids),  "data/matrices/ids_test.csv",  row.names = FALSE)

ids_train <- as.character(read.csv(paste0("data/matrices/ids_train", TAG, ".csv"))$response_id)
ids_test  <- as.character(read.csv(paste0("data/matrices/ids_test",  TAG, ".csv"))$response_id)

cat("TRAIN: ids =", length(ids_train), "| filas X_train =", nrow(X_train), "\n")
cat("TEST : ids =", length(ids_test),  "| filas X_test  =", nrow(X_test),  "\n")
stopifnot(
  "ids_train y X_train no coinciden: matrices y IDs vienen de corridas distintas" =
    length(ids_train) == nrow(X_train),
  "ids_test y X_test no coinciden: matrices y IDs vienen de corridas distintas" =
    length(ids_test) == nrow(X_test)
)

# -----------------------------------------------------------------------------
# C) Calcular SoW y Siow en T2 para TRAIN y TEST
#    (misma lógica que code2; ajusta nombres de objetos a tu sesión)
# -----------------------------------------------------------------------------
# Necesitas en memoria (o recargar):
# purchases_tech2, ref_date3, ref_date4, c_train  (c_train del code2)

# Si no tienes c_train guardado, estímalo solo con train (gasto Samsung T2):
# o léelo de un archivo si lo guardaste.

build_sow_siow <- function(ids, purchases_tech2, ref_date3, ref_date4,
                           brand = "Samsung", c_train) {
  ids <- as.character(ids)
  period <- purchases_tech2 %>%
    filter(
      between(order_date, ref_date3, ref_date4),
      response_id %in% ids
    )
  
  out <- period %>%
    group_by(response_id) %>%
    summarise(
      amount_brand  = sum(if_else(tech_brand == brand, total_price, 0), na.rm = TRUE),
      amount_other  = sum(if_else(tech_brand != brand, total_price, 0), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    right_join(tibble(response_id = ids), by = "response_id") %>%
    mutate(
      amount_brand = replace_na(amount_brand, 0),
      amount_other = replace_na(amount_other, 0),
      amount_total = amount_brand + amount_other,
      siow = pmax(0, as.integer(round(amount_total / c_train))),
      sow  = if_else(amount_total > 0, amount_brand / amount_total, 0)
    ) %>%
    select(response_id, sow, siow)
  
  # mismo orden que ids
  out[match(ids, out$response_id), ]
}

# c_train guardado por code2 (misma corrida / mismo TAG)
c_train <- readRDS(paste0("data/matrices/c_train", TAG, ".rds"))

# Estos objetos vienen de code2: deben estar cargados en la sesion de R
stopifnot(exists("purchases_tech2"), exists("ref_date3"), exists("ref_date4"))

sow_train_df <- build_sow_siow(ids_train, purchases_tech2, ref_date3, ref_date4, "Samsung", c_train)
sow_test_df  <- build_sow_siow(ids_test,  purchases_tech2, ref_date3, ref_date4, "Samsung", c_train)

# -----------------------------------------------------------------------------
# Guardar SoW y Siow de TRAIN -> data/matrices/sow_siow_train_min<MIN_MONETARY_T1>.csv
# (columnas: response_id, sow, siow; es el archivo que lee tune_xgb_sow_siow.py)
# -----------------------------------------------------------------------------
dir.create("data/matrices", showWarnings = FALSE, recursive = TRUE)
stopifnot(identical(as.character(sow_train_df$response_id), ids_train))

sow_train_file <- paste0("data/matrices/sow_siow_train", TAG, ".csv")
write.csv(sow_train_df, sow_train_file, row.names = FALSE)
cat("Guardado:", sow_train_file, "| n =", nrow(sow_train_df), "\n")

SoW_train <- sow_train_df$sow
SoW_test  <- sow_test_df$sow
Siow_train <- sow_train_df$siow
Siow_test  <- sow_test_df$siow

cat("% ceros SoW train:", mean(SoW_train == 0) * 100, "\n")
cat("% ceros SoW test :", mean(SoW_test  == 0) * 100, "\n")

# -----------------------------------------------------------------------------
# D1) Modelo 1: Fractional logit sobre SoW  (Papke-Wooldridge)
#     SoW en [0,1] directamente
# -----------------------------------------------------------------------------
train_sow <- data.frame(SoW = SoW_train, X_train)
test_sow  <- data.frame(SoW = SoW_test,  X_test)

# Fórmula (puedes restringir columnas como en el Pogit)
form_sow <- SoW ~ .

m_frac <- glm(form_sow, data = train_sow, family = quasibinomial(link = "logit"))
print(summary(m_frac))   # coeficientes SoW (fractional logit)

pred_sow_train <- predict(m_frac, newdata = train_sow, type = "response")
pred_sow_test  <- predict(m_frac, newdata = test_sow,  type = "response")

rmse <- function(a, b) sqrt(mean((a - b)^2, na.rm = TRUE))
mae  <- function(a, b) mean(abs(a - b), na.rm = TRUE)

cat("\n=== Fractional logit SoW ===\n")
cat("TRAIN RMSE:", round(rmse(SoW_train, pred_sow_train), 4),
    " MAE:", round(mae(SoW_train, pred_sow_train), 4), "\n")
cat("TEST  RMSE:", round(rmse(SoW_test,  pred_sow_test), 4),
    " MAE:", round(mae(SoW_test,  pred_sow_test), 4), "\n")
cat("TEST  cor :", round(cor(SoW_test, pred_sow_test, use = "complete.obs"), 4), "\n")

# -----------------------------------------------------------------------------
# D2) Modelo 2: Two-part (recomendado con muchos ceros en SoW)
#     (1) P(SoW > 0)  (2) E[SoW | SoW > 0]
# -----------------------------------------------------------------------------
train_sow$pos <- as.integer(train_sow$SoW > 0)
test_sow$pos  <- as.integer(test_sow$SoW  > 0)

# Parte 1: probabilidad de SoW positivo
m_bin <- glm(pos ~ . - SoW, data = train_sow, family = binomial)
cat("\n=== Coeficientes SoW, parte 1: P(SoW > 0) ===\n")
print(summary(m_bin))
p_pos_test <- predict(m_bin, newdata = test_sow, type = "response")

# Parte 2: solo observaciones con SoW > 0
train_pos <- train_sow %>% filter(SoW > 0)
m_pos <- glm(SoW ~ . - pos, data = train_pos, family = quasibinomial(link = "logit"))
cat("\n=== Coeficientes SoW, parte 2: E[SoW | SoW > 0] ===\n")
print(summary(m_pos))
mu_pos_test <- predict(m_pos, newdata = test_sow, type = "response")

# Predicción combinada
pred_sow_2p_test <- p_pos_test * mu_pos_test

cat("\n=== Two-part SoW ===\n")
cat("TEST RMSE:", round(rmse(SoW_test, pred_sow_2p_test), 4), "\n")
cat("TEST MAE :", round(mae(SoW_test,  pred_sow_2p_test), 4), "\n")
cat("TEST cor :", round(cor(SoW_test, pred_sow_2p_test, use = "complete.obs"), 4), "\n")

# -----------------------------------------------------------------------------
# D3) Opcional: Siow como conteo (similar espíritu al Pogit)
# -----------------------------------------------------------------------------
train_siow <- data.frame(Siow = Siow_train, X_train)
test_siow  <- data.frame(Siow = Siow_test,  X_test)

m_siow <- zeroinfl(Siow ~ . | ., data = train_siow, dist = "poisson", link = "logit")
cat("\n=== Coeficientes Siow (ZIP): bloque count = Poisson (log); bloque zero = logit de cero estructural ===\n")
print(summary(m_siow))
pred_siow_test <- predict(m_siow, newdata = test_siow, type = "response")

cat("\n=== ZIP sobre Siow ===\n")
cat("TEST RMSE:", round(rmse(Siow_test, pred_siow_test), 4), "\n")
cat("TEST MAE :", round(mae(Siow_test,  pred_siow_test), 4), "\n")
cat("TEST cor :", round(cor(Siow_test, pred_siow_test, use = "complete.obs"), 4), "\n")

# -----------------------------------------------------------------------------
# E) Guardar predicciones test
# -----------------------------------------------------------------------------
out <- data.frame(
  response_id   = ids_test,
  SoW_real      = SoW_test,
  SoW_pred_frac = pred_sow_test,
  SoW_pred_2p   = pred_sow_2p_test,
  Siow_real     = Siow_test,
  Siow_pred     = pred_siow_test
)
out_file <- paste0("data/matrices/pred_sow_siow_test", TAG, ".csv")
write.csv(out, out_file, row.names = FALSE)
cat("\nGuardado:", out_file, "\n")

# -----------------------------------------------------------------------------
# F) Guardar TODOS los coeficientes (SoW y Siow) en un Excel
# -----------------------------------------------------------------------------
tidy_coef <- function(tab) {
  x <- as.data.frame(tab)
  data.frame(term = rownames(x), x, row.names = NULL, check.names = FALSE)
}

sm_siow <- summary(m_siow)

wb_coef <- openxlsx::createWorkbook()
add_sheet <- function(name, tab) {
  openxlsx::addWorksheet(wb_coef, name)
  openxlsx::writeData(wb_coef, name, tidy_coef(tab))
}
add_sheet("SoW_fraclogit",     summary(m_frac)$coefficients)
add_sheet("SoW_2p_binario",    summary(m_bin)$coefficients)
add_sheet("SoW_2p_positivo",   summary(m_pos)$coefficients)
add_sheet("Siow_ZIP_conteo",   sm_siow$coefficients$count)
add_sheet("Siow_ZIP_ceros",    sm_siow$coefficients$zero)

coef_file <- paste0("data/matrices/coeficientes_sow_siow", TAG, ".xlsx")
openxlsx::saveWorkbook(wb_coef, coef_file, overwrite = TRUE)
cat("Coeficientes guardados:", coef_file, "\n")