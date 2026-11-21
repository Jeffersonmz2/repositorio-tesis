library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

# ------------------------------------------------------------------
# 1. CARGAR Y UNIR MATRICES (N = 2,190)
# ------------------------------------------------------------------
ruta_excel <- "data/Matrices/matrices_data_min80.xlsx"

V_total <- bind_rows(
  read_excel(ruta_excel, sheet = "V_train"),
  read_excel(ruta_excel, sheet = "V_test")
)

W_total <- bind_rows(
  read_excel(ruta_excel, sheet = "W_train"),
  read_excel(ruta_excel, sheet = "W_test")
)

# ------------------------------------------------------------------
# 2. FUNCIÓN DE ESTILO GRÁFICO (IDÉNTICO A LAS FIGURAS 6.2 - 6.4)
# ------------------------------------------------------------------
graficar_barras <- function(df_resumen, titulo, eje_x, y_max,
                            rotar_x = FALSE, archivo_salida = NULL) {
  p <- ggplot(df_resumen, aes(x = categoria, y = pct)) +
    geom_col(fill = "#1F77B4", width = 0.72) +
    geom_text(
      aes(label = sprintf("%.1f%%", pct)),
      vjust = -0.45,
      size = 3.5,
      color = "#111111"
    ) +
    scale_y_continuous(
      limits = c(0, y_max),
      expand = expansion(mult = c(0, 0.04))
    ) +
    labs(
      title = titulo,
      x = eje_x,
      y = "Porcentaje de clientes (%)"
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12, color = "#111111"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "#E5E5E5", linewidth = 0.4),
      panel.border = element_rect(color = "#333333", fill = NA, linewidth = 0.6),
      axis.text = element_text(color = "#111111", size = 9.5),
      axis.title = element_text(color = "#111111", size = 10)
    )
  
  if (rotar_x) {
    p <- p + theme(axis.text.x = element_text(angle = 35, hjust = 1))
  }
  
  if (!is.null(archivo_salida)) {
    ggsave(archivo_salida, plot = p, width = 7.5, height = 4.3, dpi = 300, bg = "white")
  }
  return(p)
}

# ------------------------------------------------------------------
# 3. GRÁFICA 1: FRECUENCIA DE COMPRA SAMSUNG EN W (Reconstruida desde z)
# ------------------------------------------------------------------
z_vals <- sort(unique(round(W_total$frequency_2018_Samsung, 6)))
paso_z <- z_vals[2] - z_vals[1]

df_freq_w <- W_total %>%
  mutate(
    n_compras_num = round((frequency_2018_Samsung - min(z_vals)) / paso_z) + 1,
    categoria = ifelse(n_compras_num >= 9, "9 o más", as.character(n_compras_num))
  ) %>%
  count(n_compras_num, categoria, name = "n") %>%
  arrange(n_compras_num) %>%
  mutate(
    pct = 100 * n / sum(n),
    categoria = factor(categoria, levels = categoria)
  )

p_freq_w <- graficar_barras(
  df_freq_w,
  titulo = "Distribución de clientes según frecuencia de compras tecnológicas Samsung",
  eje_x = "Número de transacciones tecnológicas Samsung (Ventana 1)",
  y_max = 46,
  archivo_salida = "fig_frecuencia_compras_W.png"
)
print(p_freq_w)

# (Opcional) Agrupada en 3 tramos de frecuencia
df_freq_tramos <- W_total %>%
  mutate(
    n_compras = round((frequency_2018_Samsung - min(z_vals)) / paso_z) + 1,
    categoria = case_when(
      n_compras == 1 ~ "1 compra",
      n_compras %in% 2:3 ~ "Entre 2 y 3 compras",
      TRUE ~ "4 o más compras"
    ),
    categoria = factor(categoria, levels = c("1 compra", "Entre 2 y 3 compras", "4 o más compras"))
  ) %>%
  count(categoria, name = "n") %>%
  mutate(pct = 100 * n / sum(n))

p_freq_tramos <- graficar_barras(
  df_freq_tramos,
  titulo = "Distribución de clientes según nivel de frecuencia de compra",
  eje_x = "Frecuencia de compras tecnológicas Samsung",
  y_max = 48,
  archivo_salida = "fig_frecuencia_tramos_W.png"
)
print(p_freq_tramos)

# ------------------------------------------------------------------
# 4. GRÁFICA 2: PERSONAS QUE COMPARTEN LA CUENTA (V)
# ------------------------------------------------------------------
df_howmany <- V_total %>%
  mutate(
    categoria = case_when(
      q_amazon_use_howmany_2 == 1 ~ "2",
      q_amazon_use_howmany_3 == 1 ~ "3",
      `q_amazon_use_howmany_4+` == 1 ~ "4 o más",
      TRUE ~ "1\n(Solo el participante)"
    ),
    categoria = factor(
      categoria,
      levels = c("1\n(Solo el participante)", "2", "3", "4 o más")
    )
  ) %>%
  count(categoria, name = "n") %>%
  mutate(pct = 100 * n / sum(n))

p_howmany <- graficar_barras(
  df_howmany,
  titulo = "Distribución de clientes según el número de personas que comparten la cuenta de Amazon",
  eje_x = "Número de personas que comparten la cuenta",
  y_max = 70,
  archivo_salida = "fig_comparten_cuenta_V.png"
)
print(p_howmany)

# ------------------------------------------------------------------
# 5. GRÁFICA 3: RANGO DE EDAD (V)
# ------------------------------------------------------------------
df_edad <- V_total %>%
  mutate(
    categoria = case_when(
      `q_demos_age_25 - 34 years` == 1 ~ "25–34",
      `q_demos_age_35 - 44 years` == 1 ~ "35–44",
      `q_demos_age_45 - 54 years` == 1 ~ "45–54",
      `q_demos_age_55 - 64 years` == 1 ~ "55–64",
      `q_demos_age_65 and older` == 1  ~ "65 o más",
      TRUE ~ "18–24"
    ),
    categoria = factor(
      categoria,
      levels = c("18–24", "25–34", "35–44", "45–54", "55–64", "65 o más")
    )
  ) %>%
  count(categoria, name = "n") %>%
  mutate(pct = 100 * n / sum(n))

p_edad <- graficar_barras(
  df_edad,
  titulo = "Distribución de clientes por rango de edad",
  eje_x = "Rango de edad",
  y_max = 42,
  archivo_salida = "fig_rango_edad_V.png"
)
print(p_edad)

# ------------------------------------------------------------------
# 6. GRÁFICA 4: RANGO DE INGRESOS DEL HOGAR (V)
# ------------------------------------------------------------------
df_ingreso <- V_total %>%
  mutate(
    categoria = case_when(
      `q_demos_income_Less than $25,000` == 1   ~ "Menos de $25k",
      `q_demos_income_$50,000 - $74,999` == 1   ~ "$50k–$74.9k",
      `q_demos_income_$75,000 - $99,999` == 1   ~ "$75k–$99.9k",
      `q_demos_income_$100,000 - $149,999` == 1 ~ "$100k–$149.9k",
      `q_demos_income_$150,000 or more` == 1    ~ "$150k o más",
      `q_demos_income_Prefer not to say` == 1   ~ "Prefiere no decir",
      TRUE ~ "$25k–$49.9k"
    ),
    categoria = factor(
      categoria,
      levels = c("Menos de $25k", "$25k–$49.9k", "$50k–$74.9k",
                 "$75k–$99.9k", "$100k–$149.9k", "$150k o más", "Prefiere no decir")
    )
  ) %>%
  count(categoria, name = "n") %>%
  mutate(pct = 100 * n / sum(n))

p_ingreso <- graficar_barras(
  df_ingreso,
  titulo = "",
  eje_x = "Ingreso anual del hogar (USD)",
  y_max = 28,
  rotar_x = TRUE,
  archivo_salida = "fig_ingresos_V.png"
)
print(p_ingreso)