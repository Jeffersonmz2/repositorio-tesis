
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

#variables

count_vars <- c(
  "frequency_2018_Samsung",
  "monetary_2018_Samsung",
  "dollar_avg12_top1_ratio",
  "recency_days_2018_Samsung",
  "n_states",
  "q_amazon_use_howmany_4.",  
  "q_amazon_use_howmany_3",
  "q_demos_income_.50.000....74.999",
  "q_demos_income_.150.000.or.more",
  "q_demos_age_25...34.years",
  "q_demos_age_35...44.years",
  "q_demos_age_45...54.years",
  "life_lost_a_job",
  "life_moved_place_of_residence"
)

zero_vars <- c(
  "recency_days_2018_Samsung",
  "frequency_2018_Samsung",
  "monetary_2018_Samsung",
  "n_states"
)

# Asegurar que existen en los datos
count_vars <- intersect(count_vars, names(train_df))
zero_vars  <- intersect(zero_vars,  names(train_df))

form_final <- as.formula(
  paste("Y ~", paste(count_vars, collapse = " + "),
        "|", paste(zero_vars, collapse = " + "))
)

print(form_final)

set.seed(123)
pogit_final <- zeroinfl(
  formula = form_final,
  data    = train_df,
  dist    = "poisson",
  link    = "logit",
  method  = "BFGS"
)

summary(pogit_final)

# Métricas en test
pred_test_final <- predict(pogit_final, newdata = test_df, type = "response")
sse_test <- sum((Y_test - pred_test_final)^2)
sst_test <- sum((Y_test - mean(Y_test))^2)
r2_test <- 1 - (sse_test / sst_test)
cat("TEST RMSE:", round(sqrt(mean((Y_test - pred_test_final)^2)), 4), "\n")
cat("TEST MAE :", round(mean(abs(Y_test - pred_test_final)), 4), "\n")
cat("TEST R2  :", round(r2_test, 4), "\n")