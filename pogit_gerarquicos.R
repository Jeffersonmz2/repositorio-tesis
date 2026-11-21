# =============================================================================
# MÉTRICAS TRAIN / TEST: Y, ShoW y SioW
# Solo lectura de referencias e impresión en consola.
# =============================================================================

# Carpeta del archivo de matrices utilizado para ajustar el modelo.
DIR_IDS <- dirname(wb)
DIR_REFERENCIAS <- dirname(wb)

# Si los CSV empíricos están en otra carpeta, cambia únicamente:
# DIR_REFERENCIAS <- "ruta/a/esa/carpeta"

# -----------------------------------------------------------------------------
# 1. Lectura de CSV
# -----------------------------------------------------------------------------
leer_csv <- function(ruta) {
  if (!file.exists(ruta)) {
    stop("No se encontró el archivo: ", ruta)
  }
  
  read.csv(
    ruta,
    colClasses = "character",
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

# -----------------------------------------------------------------------------
# 2. Referencias empíricas alineadas con el orden de las matrices
# -----------------------------------------------------------------------------
leer_referencia <- function(conjunto, n_predicciones) {
  
  sufijo <- tolower(conjunto)
  
  archivo_ids <- file.path(
    DIR_IDS,
    paste0("ids_", sufijo, TAG, ".csv")
  )
  
  ids_df <- leer_csv(archivo_ids)
  
  if (!"response_id" %in% names(ids_df)) {
    stop("El archivo de IDs no contiene response_id: ", archivo_ids)
  }
  
  ids <- as.character(ids_df$response_id)
  
  if (length(ids) != n_predicciones ||
      anyNA(ids) ||
      any(ids == "") ||
      anyDuplicated(ids)) {
    stop("Los IDs de ", conjunto,
         " son inválidos o no coinciden con el número de predicciones.")
  }
  
  archivo_ref <- file.path(
    DIR_REFERENCIAS,
    paste0("sow_siow_", sufijo, TAG, ".csv")
  )
  
  if (file.exists(archivo_ref)) {
    
    ref <- leer_csv(archivo_ref)
    
  } else if (conjunto == "TEST") {
    
    # Alternativa: Excel empírico generado por code2.
    archivo_excel <- file.path(
      DIR_REFERENCIAS,
      paste0("empirical_sow", TAG, ".xlsx")
    )
    
    if (!file.exists(archivo_excel)) {
      stop(
        "No se encontró la referencia TEST. Se necesita ",
        archivo_ref, " o ", archivo_excel
      )
    }
    
    id_excel <- openxlsx::read.xlsx(archivo_excel, sheet = "id")
    sow_excel <- openxlsx::read.xlsx(archivo_excel, sheet = "sow")
    siow_excel <- openxlsx::read.xlsx(archivo_excel, sheet = "siow")
    
    if (nrow(id_excel) != nrow(sow_excel) ||
        nrow(id_excel) != nrow(siow_excel) ||
        !"response_id" %in% names(id_excel)) {
      stop("Las hojas del Excel empírico TEST no son consistentes.")
    }
    
    ref <- data.frame(
      response_id = as.character(id_excel$response_id),
      sow = sow_excel[[1]],
      siow = siow_excel[[1]]
    )
    
  } else {
    stop("No se encontró la referencia TRAIN: ", archivo_ref)
  }
  
  requeridas <- c("response_id", "sow", "siow")
  
  if (!all(requeridas %in% names(ref))) {
    stop("La referencia ", conjunto,
         " debe contener response_id, sow y siow.")
  }
  
  ref$response_id <- as.character(ref$response_id)
  
  if (anyNA(ref$response_id) ||
      any(ref$response_id == "") ||
      anyDuplicated(ref$response_id)) {
    stop("La referencia ", conjunto, " contiene IDs inválidos o repetidos.")
  }
  
  if (!setequal(ids, ref$response_id)) {
    stop("Los clientes de la referencia ", conjunto,
         " no coinciden con los IDs de las matrices.")
  }
  
  # Reordenar según los IDs guardados al construir las matrices.
  ref <- ref[match(ids, ref$response_id), requeridas, drop = FALSE]
  
  ref$sow <- suppressWarnings(as.numeric(ref$sow))
  ref$siow <- suppressWarnings(as.numeric(ref$siow))
  
  if (any(!is.finite(ref$sow)) ||
      any(!is.finite(ref$siow)) ||
      any(ref$sow < 0 | ref$sow > 1) ||
      any(ref$siow < 0)) {stop("Las referencias empíricas de ", conjunto, " no son válidas.")
  }
  
  rownames(ref) <- NULL
  ref
}

ref_train <- leer_referencia("TRAIN", nrow(pred_train))
ref_test  <- leer_referencia("TEST", nrow(pred_test))

if (length(intersect(ref_train$response_id, ref_test$response_id)) > 0L) {
  stop("Hay clientes compartidos entre TRAIN y TEST.")
}

# -----------------------------------------------------------------------------
# 3. Función de evaluación
# -----------------------------------------------------------------------------
evaluar_respuesta <- function(observado, estimado, conjunto, variable) {
  
  observado <- as.numeric(observado)
  estimado  <- as.numeric(estimado)
  
  if (length(observado) == 0L ||
      length(observado) != length(estimado) ||
      any(!is.finite(observado)) ||
      any(!is.finite(estimado))) {
    stop("Valores inválidos al evaluar ", variable, " en ", conjunto)
  }
  
  error <- estimado - observado
  
  data.frame(
    Conjunto = conjunto,
    Variable = variable,
    N = length(observado),
    MAE = mean(abs(error)),
    RMSE = sqrt(mean(error^2)),
    row.names = NULL
  )
}

# -----------------------------------------------------------------------------
# 4. MAE y RMSE de las tres respuestas en ambos conjuntos
# -----------------------------------------------------------------------------
metricas_pogit <- rbind(
  
  evaluar_respuesta(
    Y_train, pred_train$mu_Y,
    "TRAIN", "Y"
  ),
  
  evaluar_respuesta(
    Y_test, pred_test$mu_Y,
    "TEST", "Y"
  ),
  
  evaluar_respuesta(
    ref_train$sow, pred_train$theta_ShoW,
    "TRAIN", "ShoW"
  ),
  
  evaluar_respuesta(
    ref_test$sow, pred_test$theta_ShoW,
    "TEST", "ShoW"
  ),
  
  evaluar_respuesta(
    ref_train$siow, pred_train$SioW_hat,
    "TRAIN", "SioW condicionado a Y"
  ),
  
  evaluar_respuesta(
    ref_test$siow, pred_test$SioW_hat,
    "TEST", "SioW condicionado a Y"
  )
)

cat("\n============================================================\n")
cat("POGIT JERÁRQUICO: MÉTRICAS TRAIN / TEST\n")
cat("============================================================\n")

print(metricas_pogit, digits = 6, row.names = FALSE)

# -----------------------------------------------------------------------------
# 5. Diferencias TEST - TRAIN
# -----------------------------------------------------------------------------
m_train <- metricas_pogit[
  metricas_pogit$Conjunto == "TRAIN", , drop = FALSE
]

m_test <- metricas_pogit[
  metricas_pogit$Conjunto == "TEST", , drop = FALSE
]

m_test <- m_test[
  match(m_train$Variable, m_test$Variable), , drop = FALSE
]

brechas <- data.frame(
  Variable = m_train$Variable,
  MAE_train = m_train$MAE,
  MAE_test = m_test$MAE,
  Diferencia_MAE = m_test$MAE - m_train$MAE,
  RMSE_train = m_train$RMSE,
  RMSE_test = m_test$RMSE,
  Diferencia_RMSE = m_test$RMSE - m_train$RMSE
)

cat("\n=== DIFERENCIAS TEST - TRAIN ===\n")
cat("Una diferencia positiva indica mayor error en TEST.\n\n")
# -----------------------------------------------------------------------------
# 6. Resumen de todas las salidas del modelo
# -----------------------------------------------------------------------------
cat("\n=== RESUMEN DE SALIDAS TRAIN ===\n")
print(summary(pred_train))

cat("\n=== RESUMEN DE SALIDAS TEST ===\n")
print(summary(pred_test))

cat("\n=== DIAGNÓSTICO DEL AJUSTE ===\n")
cat("Código de convergencia optim:", fit$convergence, "\n")
cat("Negativa de la log-verosimilitud:", fit$value, "\n")

print(brechas, digits = 6, row.names = FALSE)