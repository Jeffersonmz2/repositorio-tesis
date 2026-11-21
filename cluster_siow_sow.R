library(dplyr)
library(readr)
library(openxlsx)

# 1) Clústeres con response_id REAL (ajusta la ruta)
clust <- read_csv("data/matrices/clusters_train_min80_REAL_IDS.csv")
# columnas esperadas: response_id, cluster

# 2) SoW / Siow de train
sow_tr <- read_csv("data/matrices/sow_siow_train.csv")
# response_id, sow, siow

df <- clust %>%
  mutate(response_id = as.character(response_id)) %>%
  inner_join(
    sow_tr %>% mutate(response_id = as.character(response_id)),
    by = "response_id"
  )

cat("N cruzados:", nrow(df), "\n")

# --- Resumen por clúster ---
resumen <- df %>%
  group_by(cluster) %>%
  summarise(
    n              = n(),
    pct            = 100 * n() / nrow(df),
    sow_media      = mean(sow, na.rm = TRUE),
    sow_mediana    = median(sow, na.rm = TRUE),
    pct_sow_cero   = 100 * mean(sow == 0, na.rm = TRUE),
    pct_sow_alto   = 100 * mean(sow >= 0.2, na.rm = TRUE),  # cuota >= 20%
    siow_media     = mean(siow, na.rm = TRUE),
    siow_mediana   = median(siow, na.rm = TRUE),
    potencial_medio = mean(siow * (1 - sow), na.rm = TRUE),  # hueco de cuota
    .groups = "drop"
  ) %>%
  arrange(cluster)

print(resumen)

# --- Tabla clúster × tramo de SoW ---
df <- df %>%
  mutate(
    sow_bin = cut(sow, breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1),
                  include.lowest = TRUE, right = TRUE),
    siow_q  = ntile(siow, 5)
  )

tabla_sow <- df %>%
  count(cluster, sow_bin) %>%
  group_by(cluster) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ungroup()

print(tabla_sow)

write_csv(resumen, "data/matrices/cruce_cluster_sow_siow_train.csv")