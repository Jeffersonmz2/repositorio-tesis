# =============================================================================
# 0. LIBRERÍAS
# =============================================================================
library(tidyverse)   # readr, dplyr, tidyr, ggplot2 (dplyr >= 1.1.0)
library(scales)      # label_percent
library(patchwork)   # plot_layout, plot_annotation

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
TAG <- "_min80"

BASE <- "cruce_cluster_siow_sow"   # carpeta al mismo nivel que este script
BASE1 <- "data/matrices"
# Carpeta (no archivo) donde están el CSV de train y donde se guardan resultados.
SALIDA <- file.path(
  "..", "data", "matrices", "resultados_integracion_clusters_min80"
)

# Predicciones XGBoost por debajo de este valor se tratan como cero
# (XGBoost casi nunca predice 0 exacto; el empírico sí tiene ceros exactos).
UMBRAL_CERO <- 0.005

RUTA_CRUCE <- file.path(
  BASE,
  paste0("cruce_clientes_clusters_sow_siow", TAG, ".csv")
)

RUTA_TEST <- file.path(
  BASE1, paste0("xgb_optuna_sow_siow_test", TAG, ".csv")
)

RUTA_TRAIN <- file.path(
  SALIDA, paste0("xgb_optuna_sow_siow_train", TAG, ".csv")
)

# =============================================================================
# 2. LECTURA Y COMPROBACIONES
# =============================================================================
leer <- function(ruta) {
  if (!file.exists(ruta)) stop("No se encontró: ", ruta)
  
  read_csv(
    ruta,
    col_types = cols(
      response_id = col_character(),
      .default = col_guess()
    ),
    show_col_types = FALSE
  )
}

validar <- function(d, columnas) {
  if (!all(columnas %in% names(d))) {
    stop("Faltan columnas: ",
         paste(setdiff(columnas, names(d)), collapse = ", "))
  }
  
  if (anyNA(d$response_id) ||
      any(d$response_id == "") ||
      anyDuplicated(d$response_id)) {
    stop("Hay IDs vacíos o duplicados.")
  }
  
  numericas <- setdiff(columnas, c("response_id", "cluster"))
  
  for (v in numericas) {
    if (!is.numeric(d[[v]]) || any(!is.finite(d[[v]]))) {
      stop("Valores inválidos en: ", v)
    }
  }
}

base <- leer(RUTA_CRUCE)
validar(base, c("response_id", "cluster", "sow", "siow"))

if (!setequal(as.character(base$cluster), as.character(1:4))) {
  stop("Se esperaban los cuatro clústeres identificados como 1, 2, 3 y 4.")
}

if (any(base$sow < 0 | base$sow > 1) || any(base$siow < 0)) {
  stop("ShoW o SioW empíricos fuera de su dominio.")
}

base <- base %>%
  mutate(cluster = factor(
    paste0("C", cluster),
    levels = paste0("C", 1:4)
  ))

xgb_test <- leer(RUTA_TEST)
columnas_xgb <- c(
  "response_id", "SoW_real", "SoW_pred", "Siow_real", "Siow_pred"
)
validar(xgb_test, columnas_xgb)

if (length(intersect(base$response_id, xgb_test$response_id)) > 0L) {
  stop("El cruce TRAIN y el archivo TEST comparten clientes.")
}

# =============================================================================
# 3. INTEGRACIÓN DE LAS PREDICCIONES TRAIN
# =============================================================================
conjunta <- base %>%
  select(response_id, cluster, sow, siow) %>%
  mutate(fuente = "Empírico TRAIN")

comparacion_train <- NULL

if (file.exists(RUTA_TRAIN)) {
  
  xgb_train <- leer(RUTA_TRAIN)
  validar(xgb_train, columnas_xgb)
  
  if (!setequal(base$response_id, xgb_train$response_id)) {
    stop("El CSV XGBoost TRAIN no contiene los mismos clientes del cruce.")
  }
  
  comparacion_train <- base %>%
    left_join(
      xgb_train,
      by = "response_id",
      relationship = "one-to-one"
    )
  
  if (any(abs(comparacion_train$sow -
              comparacion_train$SoW_real) > 1e-6) ||
      any(abs(comparacion_train$siow -
              comparacion_train$Siow_real) > 1e-6)) {
    stop("Las referencias empíricas TRAIN no coinciden por cliente.")
  }
  
  if (any(comparacion_train$SoW_pred < 0 |
          comparacion_train$SoW_pred > 1) ||
      any(comparacion_train$Siow_pred < 0)) {
    stop("Las predicciones están fuera de su dominio.")
  }
  
  conjunta <- bind_rows(
    conjunta,
    comparacion_train %>%
      transmute(
        response_id, cluster,
        sow = SoW_pred,
        siow = Siow_pred,
        fuente = "XGBoost TRAIN"
      )
  )
  
} else {
  message(
    "Falta el CSV de predicciones TRAIN. ",
    "Las figuras mostrarán únicamente los valores empíricos."
  )
}

conjunta <- conjunta %>%
  mutate(fuente = factor(
    fuente,
    levels = c("Empírico TRAIN", "XGBoost TRAIN")
  ))

# =============================================================================
# 4. CARACTERIZACIÓN POR CLÚSTER
# =============================================================================
tamanos <- base %>%
  count(cluster, name = "clientes") %>%
  mutate(porcentaje_clientes = 100 * clientes / sum(clientes))

resumen <- conjunta %>%
  pivot_longer(
    cols = c(sow, siow),
    names_to = "variable",
    values_to = "valor"
  ) %>%
  mutate(variable = if_else(variable == "sow", "ShoW", "SioW")) %>%
  group_by(cluster, fuente, variable) %>%
  summarise(
    n = n(),
    media = mean(valor),
    mediana = median(valor),
    q25 = quantile(valor, 0.25),
    q75 = quantile(valor, 0.75),
    desviacion = sd(valor),
    maximo = max(valor),
    porcentaje_ceros = 100 * mean(valor <= UMBRAL_CERO),
    .groups = "drop"
  ) %>%
  left_join(
    tamanos %>% select(cluster, porcentaje_clientes),
    by = "cluster"
  )

cat("\n=== TAMAÑO DE LOS CLÚSTERES: TRAIN ===\n")
print(tamanos, n = Inf)

cat("\n=== CARACTERIZACIÓN EMPÍRICA Y XGBOOST TRAIN ===\n")
print(resumen, n = Inf, width = Inf)

# =============================================================================
# 5. MAE, RMSE Y SESGO
# =============================================================================
metricas <- function(d, conjunto, por_cluster = FALSE) {
  
  if (!por_cluster) d$cluster <- "Total"
  
  bind_rows(
    d %>% transmute(
      cluster, variable = "ShoW",
      observado = SoW_real, predicho = SoW_pred
    ),
    d %>% transmute(
      cluster, variable = "SioW",
      observado = Siow_real, predicho = Siow_pred
    )
  ) %>%
    group_by(cluster, variable) %>%
    summarise(
      n = n(),
      MAE = mean(abs(predicho - observado)),
      RMSE = sqrt(mean((predicho - observado)^2)),
      sesgo = mean(predicho - observado),
      media_observada = mean(observado),
      media_predicha = mean(predicho),
      .groups = "drop"
    ) %>%
    mutate(conjunto = conjunto, .before = 1)
}

metricas_test <- metricas(xgb_test, "TEST")

cat("\n=== XGBOOST: TEST GLOBAL ===\n")
print(metricas_test, n = Inf, width = Inf)

metricas_train <- NULL

if (!is.null(comparacion_train)) {
  
  metricas_train <- metricas(
    comparacion_train,
    "TRAIN: ajuste dentro de muestra",
    por_cluster = TRUE
  )
  
  cat("\n=== XGBOOST: AJUSTE TRAIN POR CLÚSTER ===\n")
  print(metricas_train, n = Inf, width = Inf)
  
  cat("\n=== CONTRASTE DESCRIPTIVO POR CLÚSTER ===\n")
  
  for (cl in levels(base$cluster)) {
    
    d <- comparacion_train %>% filter(cluster == cl)
    
    cat(sprintf(
      paste0(
        "\n%s (n=%d): ShoW medio empírico %.2f%% y predicho %.2f%%; ",
        "SioW mediano empírico %.2f y predicho %.2f.\n"
      ),
      cl, nrow(d),
      100 * mean(d$sow),
      100 * mean(d$SoW_pred),
      median(d$siow),
      median(d$Siow_pred)
    ))
  }
}

# =============================================================================
# 6. ESTILO GRÁFICO
# =============================================================================
colores <- c(
  "Empírico TRAIN" = "#24566F",
  "XGBoost TRAIN" = "#C47735"
)

tema <- theme_minimal(base_size = 12, base_family = "sans") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "#555555"),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.text = element_text(face = "bold")
  )

grafico_perfil <- function(variable_elegida) {
  
  d <- resumen %>% filter(variable == variable_elegida)
  
  p <- ggplot(
    d,
    aes(x = cluster, y = mediana, color = fuente, group = fuente)
  ) +
    geom_linerange(
      aes(ymin = q25, ymax = q75),
      position = position_dodge(width = 0.45),
      linewidth = 1.2
    ) +
    geom_point(
      position = position_dodge(width = 0.45),
      size = 3.2
    ) +
    scale_color_manual(values = colores) +
    labs(x = "Clúster", color = NULL) +
    tema
  
  if (variable_elegida == "ShoW") {
    p <- p +
      scale_y_continuous(labels = label_percent(accuracy = 1)) +
      labs(title = "Participación de Samsung", y = "ShoW")
  } else {
    p <- p +
      scale_y_continuous(
        trans = "log1p",
        breaks = c(0, 1, 2, 5, 10, 20, 50, 100, 200, 400)
      ) +
      labs(
        title = "Tamaño de billetera",
        y = "SioW discretizado; eje log(1 + x)"
      )
  }
  
  p
}

figura_perfiles <- (
  grafico_perfil("ShoW") + grafico_perfil("SioW") +
    plot_layout(guides = "collect") +
    plot_annotation(
      title = "Perfil de billetera de los cuatro clústeres",
      subtitle = "TRAIN: puntos = mediana; barras = intervalo intercuartílico"
    )
) & theme(legend.position = "bottom")

print(figura_perfiles)

# =============================================================================
# 7. MATRIZ DE BURBUJAS: ShoW × SioW
# =============================================================================
# Cortes de SioW calculados UNA VEZ con el TRAIN empírico.
# Los empates pueden producir menos de cinco intervalos.
cortes <- sort(unique(as.numeric(
  quantile(base$siow, probs = c(0.2, 0.4, 0.6, 0.8))
)))

limites <- c(-Inf, cortes, Inf)

fmt <- function(x) {
  format(round(x, 2), trim = TRUE, scientific = FALSE)
}

etiquetas_siow <- vapply(
  seq_len(length(limites) - 1L),
  function(i) {
    inferior <- limites[i]
    superior <- limites[i + 1L]
    
    if (!is.finite(inferior)) {
      paste0("\u2264 ", fmt(superior))
    } else if (!is.finite(superior)) {
      paste0("> ", fmt(inferior))
    } else {
      paste0("(", fmt(inferior), "; ", fmt(superior), "]")
    }
  },
  character(1)
)

etiquetas_sow <- c(
  "0", "(0,20]", "(20,40]", "(40,60]", "(60,80]", "(80,100]"
)

burbujas <- conjunta %>%
  mutate(
    sow_intervalo = if_else(
      sow <= UMBRAL_CERO,
      "0",
      as.character(cut(
        sow,
        breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1),
        labels = etiquetas_sow[-1], include.lowest = TRUE,
        right = TRUE
      ))
    ),
    sow_intervalo = factor(
      sow_intervalo,
      levels = etiquetas_sow
    ),
    siow_intervalo = cut(
      siow,
      breaks = limites,
      labels = etiquetas_siow,
      include.lowest = TRUE,
      right = TRUE
    )
  ) %>%
  count(fuente, cluster, sow_intervalo, siow_intervalo, name = "n") %>%
  group_by(fuente, cluster) %>%
  mutate(porcentaje = 100 * n / sum(n)) %>%
  ungroup()

figura_burbujas <- ggplot(
  burbujas,
  aes(x = sow_intervalo, y = siow_intervalo)
) +
  geom_point(
    aes(size = porcentaje),
    shape = 21,
    fill = "#D9E6ED",
    color = "#24566F",
    stroke = 0.6
  ) +
  geom_text(
    aes(label = if_else(
      porcentaje >= 5,
      sprintf("%.1f%%", porcentaje),
      ""
    )),
    size = 2.7,
    color = "#162F3C"
  ) +
  facet_grid(fuente ~ cluster) +
  scale_size_area(
    max_size = 15,
    limits = c(0, 100),
    breaks = c(5, 20, 40, 60, 80),
    name = "% dentro del clúster"
  ) +
  scale_x_discrete(drop = FALSE) +
  scale_y_discrete(drop = FALSE) +
  labs(
    title = "Distribución conjunta de ShoW y SioW por clúster",
    subtitle = "El área de cada burbuja representa el porcentaje de clientes",
    x = "ShoW (%)",
    y = "Intervalos de SioW discretizado",
    caption = paste(
      "Porcentajes calculados dentro de cada clúster y fuente.",
      "Cortes de SioW comunes, derivados del TRAIN empírico.",
      "Se rotulan las celdas con al menos 5%."
    )
  ) +
  tema +
  theme(
    panel.grid.major = element_line(color = "#E5E7EB"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.title = element_text(face = "bold"),
    strip.text = element_text(size = 11)
  )

print(figura_burbujas)

# =============================================================================
# 8. EXPORTAR RESULTADOS EN UNA CARPETA NUEVA
# =============================================================================
dir.create(SALIDA, recursive = TRUE, showWarnings = FALSE)

write_csv(tamanos, file.path(SALIDA, "tamanos_clusters.csv"))
write_csv(resumen, file.path(SALIDA, "caracterizacion_clusters.csv"))
write_csv(metricas_test, file.path(SALIDA, "metricas_xgb_test_global.csv"))
write_csv(burbujas, file.path(SALIDA, "matriz_burbujas.csv"))

if (!is.null(metricas_train)) {
  write_csv(
    metricas_train,
    file.path(SALIDA, "metricas_xgb_train_por_cluster.csv")
  )
}

exportar_figura <- function(grafico, nombre, ancho, alto) {
  ggsave(
    file.path(SALIDA, paste0(nombre, ".png")),
    plot = grafico, width = ancho, height = alto,
    units = "in", dpi = 400, bg = "white"
  )
  ggsave(
    file.path(SALIDA, paste0(nombre, ".pdf")),
    plot = grafico, width = ancho, height = alto,
    units = "in", device = grDevices::cairo_pdf
  )
}

exportar_figura(
  figura_perfiles, "01_perfiles_mediana_iqr", 11, 5.5
)

exportar_figura(
  figura_burbujas, "02_burbujas_show_siow",
  15, if (is.null(comparacion_train)) 5.5 else 9
)

cat("\nResultados guardados en:\n", SALIDA, "\n")