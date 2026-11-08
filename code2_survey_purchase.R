#-----------------------------------------------------------------------------
# code2.R 
#-----------------------------------------------------------------------------


# =============================================================================
# AMAZON PURCHASES (Samsung) — MATRICES V/W/Y + VALIDACIÓN SoW EMPÍRICO
# (VERSIÓN SIN FUGA EN WINSORIZACIÓN: caps SOLO con TRAIN, después del split)
#
# Reglas (acordadas):
#  1) ELIMINAR globalmente “severos” (regla fija, no Tukey):
#        amount_2023_Samsung > 5000  -> se eliminan response_id en compras, V, W, Y.
#  2) Tukey EXTREMO (k=3) NO elimina: WINSORIZA (cap superior):
#        - amount_2023_Samsung (para construir Y)  [cap estimado SOLO con TRAIN]
#        - variables numéricas de W              [cap estimado SOLO con TRAIN]
#        - SOLO n_states en V                    [cap estimado SOLO con TRAIN]
#  3) Split train/test sobre IDs (sin fuga).
#  4) c estimada SOLO con TRAIN; usarla también para TEST.
#  5) Full column rank sin QR para seleccionar: (a) V escala solo última col,
#     (b) W fija 1..4 y selecciona desde 5+ por Spearman,
#     (c) W se estandariza SOLO con TRAIN -> aplica a TEST.
# =============================================================================

#-----------------------------------------------------------------------------
# Packages
#-----------------------------------------------------------------------------
packages <- c(
  "dplyr","tidyr","stringr","tibble",
  "readr","janitor","lubridate",
  "ggplot2","scales","patchwork",
  "fastDummies","openxlsx","caret",
  "tidymodels"
)
invisible(lapply(packages, require, character.only = TRUE))

#-----------------------------------------------------------------------------
# Helpers: Tukey cap (solo arriba) + aplicar cap fijo + winsor TRAIN-only
#-----------------------------------------------------------------------------
tukey_cap_upper <- function(x, k = 3) {
  # Calcula cap = Q3 + k*IQR y devuelve vector winsorizado (solo arriba) + cap
  x_num <- suppressWarnings(as.numeric(x))
  if (all(is.na(x_num))) return(list(x = x_num, cap = NA_real_, q3 = NA_real_, iqr = NA_real_))
  
  q <- stats::quantile(x_num, probs = c(0.25, 0.75), na.rm = TRUE, type = 7)
  iqr <- as.numeric(q[2] - q[1])
  q3  <- as.numeric(q[2])
  cap <- q3 + k * iqr
  
  x_w <- x_num
  x_w[!is.na(x_w) & x_w > cap] <- cap
  list(x = x_w, cap = cap, q3 = q3, iqr = iqr)
}

apply_upper_cap <- function(x, cap) {
  # Aplica un cap ya estimado (por ejemplo desde TRAIN) a cualquier vector
  x_num <- suppressWarnings(as.numeric(x))
  if (!is.finite(cap)) return(x_num)
  x_num[!is.na(x_num) & x_num > cap] <- cap
  x_num
}

winsorize_df_numeric_upper_trainonly <- function(df_train, df_test,
                                                 id_col = "response_id",
                                                 k = 3,
                                                 cols = NULL) {
  # Estima caps SOLO con TRAIN y los aplica a TRAIN y TEST.
  # Si cols=NULL: todas las numéricas excepto id_col.
  stopifnot(id_col %in% names(df_train), id_col %in% names(df_test))
  
  if (is.null(cols)) {
    cols <- names(df_train)[sapply(df_train, is.numeric)]
    cols <- setdiff(cols, id_col)
  } else {
    cols <- intersect(cols, names(df_train))
    cols <- setdiff(cols, id_col)
  }
  
  out_tr <- df_train
  out_te <- df_test
  caps <- list()
  
  for (cc in cols) {
    tk <- tukey_cap_upper(out_tr[[cc]], k = k)
    caps[[cc]] <- tk$cap
    out_tr[[cc]] <- tk$x
    out_te[[cc]] <- apply_upper_cap(out_te[[cc]], tk$cap)
  }
  
  list(train = out_tr, test = out_te, caps = caps)
}

#-----------------------------------------------------------------------------
# Helpers: alineación IDs (V/W/var_2023) con universo maestro
#-----------------------------------------------------------------------------
align_master_ids <- function(V, W, var_2023, id_col = "response_id", master = c("W","V","var")) {
  master <- match.arg(master)
  
  V <- V %>% distinct(.data[[id_col]], .keep_all = TRUE) %>% mutate(!!id_col := as.character(.data[[id_col]]))
  W <- W %>% distinct(.data[[id_col]], .keep_all = TRUE) %>% mutate(!!id_col := as.character(.data[[id_col]]))
  var_2023 <- var_2023 %>% distinct(.data[[id_col]], .keep_all = TRUE) %>% mutate(!!id_col := as.character(.data[[id_col]]))
  
  ids_common <- Reduce(intersect, list(V[[id_col]], W[[id_col]], var_2023[[id_col]]))
  stopifnot(length(ids_common) > 0)
  
  ids_master <- switch(
    master,
    W   = W[[id_col]][W[[id_col]] %in% ids_common],
    V   = V[[id_col]][V[[id_col]] %in% ids_common],
    var = var_2023[[id_col]][var_2023[[id_col]] %in% ids_common]
  )
  
  V_a   <- V[match(ids_master, V[[id_col]]), , drop = FALSE]
  W_a   <- W[match(ids_master, W[[id_col]]), , drop = FALSE]
  var_a <- var_2023[match(ids_master, var_2023[[id_col]]), , drop = FALSE]
  
  stopifnot(identical(V_a[[id_col]], W_a[[id_col]]))
  stopifnot(identical(V_a[[id_col]], var_a[[id_col]]))
  
  list(V = V_a, W = W_a, var_2023 = var_a, ids = ids_master)
}

#-----------------------------------------------------------------------------
# Amazon Purchases Data
#-----------------------------------------------------------------------------
purchases <- readr::read_csv("data/survey-data/amazon-purchases.csv", show_col_types = FALSE) |>
  janitor::clean_names()

resp_col <- names(purchases)[stringr::str_detect(names(purchases), "response")]
stopifnot(length(resp_col) == 1)

# Semántica coherente:
# - purchase_price_per_unit = precio unitario
# - quantity = unidades
# - amount = gasto total = unitario * unidades
purchases0 <- purchases %>%
  dplyr::rename(response_id = !!resp_col) %>%
  dplyr::mutate(
    response_id = as.character(response_id),
    order_date  = lubridate::as_date(order_date),
    dplyr::across(c(purchase_price_per_unit, quantity), as.numeric),
    amount      = purchase_price_per_unit * quantity
  )

#-----------------------------------------------------------------------------
# Macro categories + Technology pattern
#-----------------------------------------------------------------------------
electronics_keywords <- c(
  "electronics?",
  "computer", "laptop", "notebook", "desktop",
  "tablet", "smartphone", "\\bphone\\b",
  "\\btv\\b", "television", "monitor", "projector",
  "printer", "scanner", "camera", "webcam",
  "speaker", "audio", "soundbar", "headphones?", "headset", "earbuds?",
  "microphone",
  "charger", "charging", "battery", "power( bank| supply)?", "adapter", "cable",
  "dock(ing)?( station)?",
  "\\bmouse\\b", "keyboard", "trackpad", "controller", "stylus",
  "\\bssd\\b", "(hard ?drive|hdd)", "(flash ?drive|thumb ?drive)",
  "\\b(memory|ram)\\b", "\\bgpu\\b", "graphics card",
  "\\busb\\b", "bluetooth", "hdmi", "\\bethernet\\b", "\\bwi[- ]?fi\\b|wifi",
  "smartwatch", "wearable",
  "router", "modem", "mesh( wifi)?",
  "toner", "\\bink\\b",
  "(vr headset|virtual reality)"
)
electronics_pattern <- paste(electronics_keywords, collapse = "|")

classify_category <- function(cat) {
  dplyr::case_when(
    stringr::str_detect(cat, "(?i)book|ebook|dvd|blu[-_ ]?ray|video_games?")           ~ "Media",
    stringr::str_detect(cat, "(?i)apparel|clothing|hosiery|bra|coat|shoe|fashion")     ~ "Apparel",
    stringr::str_detect(cat, "(?i)beauty|cosmetic|hair_|skin|lotion|fragrance")        ~ "Beauty_PersonalCare",
    stringr::str_detect(cat, "(?i)grocery|food|snack|beverage|coffee|tea|candy")       ~ "Grocery",
    stringr::str_detect(cat, "(?i)kitchen|cook|bake|dinnerware|utensil|cookware")      ~ "Kitchen_Dining",
    stringr::str_detect(cat, "(?i)home_|furniture|decor|bedding|lamp|curtain|rug")     ~ "Home_Living",
    stringr::str_detect(cat, stringr::regex(electronics_pattern, ignore_case = TRUE)) ~ "Electronics",
    stringr::str_detect(cat, "(?i)sport|fitness|exercise|camping|cycling|golf|ball")   ~ "Sports_Outdoors",
    stringr::str_detect(cat, "(?i)toy|game|lego|puzzle|doll|costume")                  ~ "Toys_Games",
    stringr::str_detect(cat, "(?i)pet_|animal")                                        ~ "Pet_Supplies",
    stringr::str_detect(cat, "(?i)baby_")                                              ~ "Baby",
    stringr::str_detect(cat, "(?i)auto|car|vehicle|engine|tire|battery")               ~ "Automotive",
    TRUE                                                                               ~ "Other"
  )
}

purchases_cat <- purchases0 %>%
  dplyr::mutate(
    broad_cat = classify_category(category) |> factor(),
    title_lc  = stringr::str_squish(stringr::str_to_lower(dplyr::coalesce(title, "")))
  )

#-----------------------------------------------------------------------------
# Brand dictionary + functions
#-----------------------------------------------------------------------------
brand_dict <- tibble::tibble(
  pattern = c(
    "\\bsamsung\\b","\\bapple\\b","\\bdell\\b","hewlett[ -]?packard|\\bhp\\b",
    "\\basus\\b","\\bacer\\b","\\blenovo\\b","\\bhuawei\\b","\\bxiaomi\\b",
    "\\bsony\\b","\\blg\\b","\\bmotorola\\b","\\bnvidia\\b","\\bamd\\b",
    "\\bintel\\b","\\bcanon\\b","\\bepson\\b","\\blogitech\\b","\\banker\\b",
    "\\bseagate\\b","\\bsandisk\\b","\\bkingston\\b","tp[- ]?link","\\bnetgear\\b",
    "\\bbose\\b","\\bjbl\\b"
  ),
  brand = c(
    "Samsung","Apple","Dell","HP","Asus","Acer","Lenovo","Huawei","Xiaomi",
    "Sony","LG","Motorola","Nvidia","AMD","Intel","Canon","Epson",
    "Logitech","Anker","Seagate","Sandisk","Kingston","TP-Link","Netgear",
    "Bose","JBL"
  )
)

is_tech_product <- function(title_lc, broad_cat) {
  by_title <- stringr::str_detect(title_lc, stringr::regex(electronics_pattern, ignore_case = TRUE))
  (by_title | broad_cat == "Electronics") %in% TRUE
}

extract_brand <- function(title_lc) {
  vapply(title_lc, function(t) {
    idx <- which(stringr::str_detect(t, stringr::regex(brand_dict$pattern, ignore_case = TRUE)))
    if (length(idx) == 0) NA_character_ else brand_dict$brand[idx[1]]
  }, character(1))
}

purchases_cat <- purchases_cat %>%
  dplyr::mutate(
    is_tech         = is_tech_product(title_lc, broad_cat),
    tech_brand_raw  = dplyr::if_else(is_tech, extract_brand(title_lc), NA_character_),
    tech_brand      = dplyr::if_else(is_tech & is.na(tech_brand_raw), "Generic", tech_brand_raw)
  ) %>%
  dplyr::select(-tech_brand_raw)

stopifnot(!any(is.na(purchases_cat$is_tech)))
stopifnot(with(purchases_cat, all(!is.na(tech_brand[is_tech]))))

# Solo tech
purchases_tech <- purchases_cat %>% dplyr::filter(is_tech)

#-----------------------------------------------------------------------------
# Technology Sub-Macrocategories
#-----------------------------------------------------------------------------
computers_pattern <- paste(c("computer", "desktop", "laptop", "notebook"), collapse = "|")
mobiles_pattern   <- paste(c("smartphone", "\\bphone\\b", "tablet"), collapse = "|")
input_pattern     <- paste(c("\\bmouse\\b", "keyboard", "trackpad", "stylus", "controller"), collapse = "|")

audio_video_pattern <- paste(
  c("speaker","headphones?","headset","earbuds?","soundbar","microphone",
    "camera","webcam","projector","monitor","\\btv\\b","television"),
  collapse = "|"
)

storage_pattern <- paste(
  c("ssd","hard ?drive","hdd","flash ?drive","thumb ?drive","memory","\\bram\\b","\\bgpu\\b","graphics card"),
  collapse = "|"
)

accessories_pattern <- paste(
  c("charger","charging","battery","adapter","cable","dock(ing)?( station)?",
    "power bank","power supply","toner","\\bink\\b"),
  collapse = "|"
)

connectivity_pattern <- paste(
  c("\\busb\\b","bluetooth","hdmi","ethernet","\\bwi[- ]?fi\\b|wifi","router","modem","mesh"),
  collapse = "|"
)

wearables_pattern <- paste(c("smartwatch","wearable","vr headset","virtual reality"), collapse = "|")

classify_tech <- function(cat) {
  dplyr::case_when(
    stringr::str_detect(cat, stringr::regex(computers_pattern,    ignore_case = TRUE)) ~ "Computers",
    stringr::str_detect(cat, stringr::regex(mobiles_pattern,      ignore_case = TRUE)) ~ "Mobiles",
    stringr::str_detect(cat, stringr::regex(input_pattern,        ignore_case = TRUE)) ~ "Input_Peripherals",
    stringr::str_detect(cat, stringr::regex(audio_video_pattern,  ignore_case = TRUE)) ~ "Audio_Video",
    stringr::str_detect(cat, stringr::regex(storage_pattern,      ignore_case = TRUE)) ~ "Storage_Components",
    stringr::str_detect(cat, stringr::regex(accessories_pattern,  ignore_case = TRUE)) ~ "Accessories_Power",
    stringr::str_detect(cat, stringr::regex(connectivity_pattern, ignore_case = TRUE)) ~ "Connectivity",
    stringr::str_detect(cat, stringr::regex(wearables_pattern,    ignore_case = TRUE)) ~ "Wearables_VR",
    TRUE ~ "Other_Tech"
  )
}

purchases_tech2 <- purchases_tech %>%
  dplyr::filter(
    stringr::str_detect(
      category,
      stringr::regex(
        paste(
          c(computers_pattern, mobiles_pattern, input_pattern,
            audio_video_pattern, storage_pattern, accessories_pattern,
            connectivity_pattern, wearables_pattern),
          collapse = "|"
        ),
        ignore_case = TRUE
      )
    )
  ) %>%
  dplyr::mutate(
    response_id = as.character(response_id),
    tech_macro  = classify_tech(category) |> factor(),
    unit_price  = purchase_price_per_unit,  # unitario
    total_price = amount                    # total
  )

#-----------------------------------------------------------------------------
# Ventanas T1/T2 (Samsung)
#-----------------------------------------------------------------------------
ref_date1a <- lubridate::ymd("2018-01-01")
ref_date2a <- lubridate::ymd("2024-12-31")

date_range <- purchases_tech2 %>%
  dplyr::filter(
    tech_brand == "Samsung",
    dplyr::between(order_date, ref_date1a, ref_date2a),
    total_price > 0
  ) %>%
  dplyr::summarise(
    min_date = min(order_date, na.rm = TRUE),
    max_date = max(order_date, na.rm = TRUE),
    .groups = "drop"
  )

min_date <- date_range$min_date[[1]]
max_date <- date_range$max_date[[1]]
stopifnot(!is.na(min_date), !is.na(max_date))

ref_date1 <- min_date
ref_date2 <- lubridate::ymd("2021-10-31")
ref_date3 <- lubridate::ymd("2021-11-01")
ref_date4 <- max_date

has_order_id <- "order_id" %in% names(purchases_tech2)

#-----------------------------------------------------------------------------
# RFM Samsung en T1
#-----------------------------------------------------------------------------
# >>> FILTRO T1: gasto TOTAL en tecnologia (todas las marcas) en T1 >= umbral en USD <<<
# (cambia SOLO este numero: 0 = sin filtro, 10, 30, 50, ...)
MIN_MONETARY_T1 <- 30
TAG <- paste0("_min", MIN_MONETARY_T1)   # se agrega al nombre de los archivos exportados

# 1) RFM de TODOS los clientes Samsung en T1 (sin filtro)
rfm_raw_2018_all <- purchases_tech2 %>%
  dplyr::filter(
    tech_brand == "Samsung",
    dplyr::between(order_date, ref_date1, ref_date2),
    total_price > 0
  ) %>%
  dplyr::group_by(response_id) %>%
  dplyr::summarise(
    recency_days_2018_Samsung = as.numeric(ref_date2 - max(order_date, na.rm = TRUE)),
    frequency_2018_Samsung    = if (has_order_id) dplyr::n_distinct(order_id) else dplyr::n(),
    monetary_2018_Samsung     = sum(total_price, na.rm = TRUE),
    .groups = "drop"
  )

# 2) Gasto TOTAL en tecnologia en T1 (todas las marcas, incluida Samsung), por cliente
tech_spend_T1 <- purchases_tech2 %>%
  dplyr::filter(
    dplyr::between(order_date, ref_date1, ref_date2),
    total_price > 0
  ) %>%
  dplyr::group_by(response_id) %>%
  dplyr::summarise(tech_spend_T1 = sum(total_price, na.rm = TRUE), .groups = "drop")

rfm_check <- rfm_raw_2018_all %>%
  dplyr::left_join(tech_spend_T1, by = "response_id") %>%
  dplyr::mutate(tech_spend_T1 = tidyr::replace_na(tech_spend_T1, 0))

# Tabla de sensibilidad (cuantos clientes quedan con cada umbral)
sens <- sapply(c(0, 100, 200, 300, 500, 1000),
               function(th) sum(rfm_check$tech_spend_T1 >= th))
names(sens) <- paste0(">=", c(0, 100, 200, 300, 500, 1000))
print(sens)

# 3) Aplicar el filtro: se queda el cliente Samsung cuyo gasto tech total en T1 >= umbral
ids_keep_T1 <- rfm_check %>%
  dplyr::filter(tech_spend_T1 >= MIN_MONETARY_T1) %>%
  dplyr::pull(response_id)

rfm_raw_2018 <- rfm_raw_2018_all %>%
  dplyr::filter(response_id %in% ids_keep_T1)

message("Clientes Samsung T1: ", nrow(rfm_raw_2018_all),
        " -> ", nrow(rfm_raw_2018),
        " (gasto tech total T1 >= ", MIN_MONETARY_T1, ")")
stopifnot(nrow(rfm_raw_2018) > 0)

# Tech-macro flags (Samsung) alineadas al universo RFM
tech_flags <- purchases_tech2 %>%
  dplyr::filter(
    tech_brand == "Samsung",
    dplyr::between(order_date, ref_date1, ref_date2)
  ) %>%
  dplyr::select(response_id, tech_macro) %>%
  dplyr::distinct() %>%
  dplyr::mutate(flag = 1L) %>%
  tidyr::pivot_wider(
    names_from   = tech_macro,
    values_from  = flag,
    values_fill  = list(flag = 0L),
    names_prefix = "2018_Samsung_tech_",
    names_expand = TRUE
  ) %>%
  dplyr::right_join(dplyr::select(rfm_raw_2018, response_id), by = "response_id") %>%
  dplyr::mutate(dplyr::across(dplyr::starts_with("2018_Samsung_tech_"), \(x) dplyr::coalesce(x, 0L)))

#-----------------------------------------------------------------------------
# Y cruda: amount_2023_Samsung (Samsung) en T2  (AÚN SIN WINSOR: se hace tras split)
#-----------------------------------------------------------------------------
var_2023 <- purchases_tech2 %>%
  dplyr::filter(
    tech_brand == "Samsung",
    dplyr::between(order_date, ref_date3, ref_date4)
  ) %>%
  dplyr::group_by(response_id) %>%
  dplyr::summarise(
    amount_2023_Samsung = sum(total_price, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::mutate(amount_2023_Samsung = tidyr::replace_na(amount_2023_Samsung, 0))

#-----------------------------------------------------------------------------
# (1) ELIMINACIÓN SEVERA global: amount_2023_Samsung > 5000
#-----------------------------------------------------------------------------
SEVERE_THRESHOLD <- 5000

anomalous_ids_severe <- var_2023 %>%
  dplyr::filter(amount_2023_Samsung > SEVERE_THRESHOLD) %>%
  dplyr::pull(response_id) %>%
  unique()

message("Severos (> ", SEVERE_THRESHOLD, "): ", length(anomalous_ids_severe))

if (length(anomalous_ids_severe) > 0) {
  var_2023       <- var_2023 %>% dplyr::filter(!response_id %in% anomalous_ids_severe)
  purchases_tech2<- purchases_tech2 %>% dplyr::filter(!response_id %in% anomalous_ids_severe)
  purchases_tech <- purchases_tech  %>% dplyr::filter(!response_id %in% anomalous_ids_severe)
  purchases_cat  <- purchases_cat   %>% dplyr::filter(!response_id %in% anomalous_ids_severe)
  purchases0     <- purchases0      %>% dplyr::filter(!response_id %in% anomalous_ids_severe)
}

# Asegurar universo consistente con RFM
var_2023 <- var_2023 %>% dplyr::semi_join(rfm_raw_2018, by = "response_id")

#-----------------------------------------------------------------------------
# Merge RFM + Y (sin NA)
#-----------------------------------------------------------------------------
rfm_amount <- rfm_raw_2018 %>%
  dplyr::left_join(var_2023, by = "response_id") %>%
  tidyr::replace_na(list(amount_2023_Samsung = 0))

#-----------------------------------------------------------------------------
# Purchases tech en T2 (para conteos por marca y validación)
#-----------------------------------------------------------------------------
purchases_tech_period <- purchases_tech2 %>%
  dplyr::filter(dplyr::between(order_date, ref_date3, ref_date4))

brand_count <- purchases_tech_period %>%
  dplyr::filter(!is.na(tech_brand), tech_brand != "") %>%
  dplyr::group_by(response_id) %>%
  dplyr::summarise(`2023_n_brand` = dplyr::n_distinct(tech_brand), .groups = "drop")

brand_items <- purchases_tech_period %>%
  dplyr::filter(!is.na(tech_brand), tech_brand != "") %>%
  dplyr::group_by(response_id, tech_brand) %>%
  dplyr::summarise(n_items = sum(quantity, na.rm = TRUE), .groups = "drop") %>%
  tidyr::pivot_wider(
    names_from   = tech_brand,
    values_from  = n_items,
    values_fill  = list(n_items = 0),
    names_prefix = "2023_brand_"
  )

rfm_amount_2018_2023 <- rfm_amount %>%
  dplyr::left_join(tech_flags, by = "response_id") %>%
  dplyr::mutate(dplyr::across(starts_with("2018_Samsung_tech_"), ~ tidyr::replace_na(.x, 0)))

rfm_amount_2018_2023b <- rfm_amount_2018_2023 %>%
  dplyr::left_join(brand_count, by = "response_id") %>%
  dplyr::left_join(brand_items, by = "response_id") %>%
  dplyr::mutate(
    `2023_n_brand` = dplyr::coalesce(`2023_n_brand`, 0),
    dplyr::across(starts_with("2023_brand_"), ~ dplyr::coalesce(.x, 0))
  )

# Broad category summary (T1)
broadcat_summary <- purchases_tech %>%
  dplyr::filter(dplyr::between(order_date, ref_date1, ref_date2)) %>%
  dplyr::group_by(response_id, broad_cat) %>%
  dplyr::summarise(
    units_cat  = sum(quantity, na.rm = TRUE),
    amount_cat = sum(amount,   na.rm = TRUE),
    .groups = "drop"
  ) %>%
  tidyr::pivot_wider(
    names_from  = broad_cat,
    values_from = c(units_cat, amount_cat),
    values_fill = list(units_cat = 0, amount_cat = 0),
    names_glue  = "2018_{.value}_{broad_cat}"
  )

rfm_amount_2018_2023b <- rfm_amount_2018_2023b %>%
  dplyr::left_join(broadcat_summary, by = "response_id") %>%
  dplyr::mutate(
    dplyr::across(starts_with("2018_units_cat_"),  ~ dplyr::coalesce(.x, 0)),
    dplyr::across(starts_with("2018_amount_cat_"), ~ dplyr::coalesce(.x, 0))
  )

# Top shipping state per user (T1)
mode_safe2 <- function(x) {
  x <- x[!is.na(x) & x != ""]
  if (length(x) == 0) return(NA_character_)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

state_summ <- purchases0 %>%
  dplyr::filter(dplyr::between(order_date, ref_date1, ref_date2)) %>%
  dplyr::mutate(
    shipping_address_state = shipping_address_state %>%
      stringr::str_trim() %>%
      stringr::str_to_upper(),
    shipping_address_state = dplyr::if_else(
      is.na(shipping_address_state) | shipping_address_state == "",
      "DIGITAL/LOCKER",
      shipping_address_state
    )
  ) %>%
  dplyr::group_by(response_id) %>%
  dplyr::summarise(
    n_states  = dplyr::n_distinct(shipping_address_state[shipping_address_state != "DIGITAL/LOCKER"]),
    top_state = mode_safe2(shipping_address_state),
    .groups   = "drop"
  )

rfm_amount_2018_2023b <- rfm_amount_2018_2023b %>%
  dplyr::left_join(state_summ, by = "response_id")

#-----------------------------------------------------------------------------
# Merge survey_enriched (se asume ya existe en tu flujo)
#-----------------------------------------------------------------------------
cols_drop <- c(
  "duration_in_seconds","q_prolific_mturk","q_control","q_altruism",
  "q_bonus_05","q_bonus_20","q_bonus_50",
  "q_data_value_100","q_data_value_any","q_data_value_any_1_text",
  "q_sell_your_data","q_sell_consumer_data","q_small_biz_use","q_census_use",
  "q_research_society","q_attn_check","showdata","incentive","connect",
  "q_demos_race","q_life_changes","q_demos_state","recorded_date"
)

valid_ids <- rfm_amount_2018_2023b %>%
  dplyr::select(response_id) %>%
  dplyr::distinct()

survey_enriched2 <- survey_enriched %>%
  dplyr::distinct(response_id, .keep_all = TRUE) %>%
  dplyr::semi_join(valid_ids, by = "response_id") %>%
  dplyr::left_join(rfm_amount_2018_2023b, by = "response_id", relationship = "one-to-one") %>%
  dplyr::select(-dplyr::all_of(cols_drop))


#-----------------------------------------------------------------------------
# Dummies (V/W base) — y NAs->0 en numéricos
#-----------------------------------------------------------------------------

# 1) Edad: referencia = "18 - 24 years"
survey_enriched2$q_demos_age <- factor(
  as.character(survey_enriched2$q_demos_age),
  levels = c(
    "18 - 24 years",
    "25 - 34 years",
    "35 - 44 years",
    "45 - 54 years",
    "55 - 64 years",
    "65 and older"
  )
)

# 2) Personas que comparten cuenta: referencia = "1 (just me!)"
survey_enriched2$q_amazon_use_howmany <- factor(
  as.character(survey_enriched2$q_amazon_use_howmany),
  levels = c("1 (just me!)", "2", "3", "4+")
)

data_dum <- survey_enriched2 %>%
  fastDummies::dummy_cols(
    select_columns = c(
      "q_amazon_use_how_oft",
      "q_demos_age",
      "q_demos_hispanic",
      "q_demos_education",
      "q_demos_income",
      "q_demos_gender",
      "q_sexual_orientation",
      "q_substance_use_1",
      "q_substance_use_2",
      "q_substance_use_3",
      "q_personal_1",
      "q_personal_2",
      "q_amazon_use_howmany",
      "q_amazon_use_hh_size"
    ),
    remove_first_dummy      = TRUE,
    remove_selected_columns = TRUE,
    ignore_na               = TRUE
  ) %>%
  dplyr::mutate(dplyr::across(where(is.numeric), ~ tidyr::replace_na(.x, 0)))

#-----------------------------------------------------------------------------
# W_pre (transaccional) + 12m Samsung features (T1)
#-----------------------------------------------------------------------------
W_pre <- data_dum %>%
  dplyr::select(
    response_id,
    recency_days_2018_Samsung,
    frequency_2018_Samsung,
    monetary_2018_Samsung
  ) %>%
  dplyr::distinct(response_id, .keep_all = TRUE) %>%
  dplyr::mutate(response_id = as.character(response_id))

if (length(anomalous_ids_severe) > 0) {
  W_pre <- W_pre %>% dplyr::filter(!response_id %in% anomalous_ids_severe)
}

ids_w <- unique(W_pre$response_id)

anchor_month <- lubridate::floor_date(ref_date2, "month")
months_12 <- seq(anchor_month %m-% months(11), anchor_month, by = "month")

Samsung_monthly <- purchases_tech2 %>%
  dplyr::filter(
    tech_brand == "Samsung",
    dplyr::between(order_date, ref_date1, ref_date2),
    response_id %in% ids_w
  ) %>%
  dplyr::mutate(ym = lubridate::floor_date(order_date, "month")) %>%
  dplyr::group_by(response_id, ym) %>%
  dplyr::summarise(
    dollar_m   = sum(total_price, na.rm = TRUE),
    products_m = sum(quantity, na.rm = TRUE),
    .groups = "drop"
  )

grid_12 <- tidyr::expand_grid(response_id = ids_w, ym = months_12) %>%
  dplyr::left_join(Samsung_monthly, by = c("response_id", "ym")) %>%
  dplyr::mutate(
    dollar_m   = tidyr::replace_na(dollar_m, 0),
    products_m = tidyr::replace_na(products_m, 0),
    mnum       = lubridate::month(ym)
  )

safe_div <- function(num, den) ifelse(den == 0, 0, num / den)
topk_avg <- function(x, k) mean(utils::head(sort(x, decreasing = TRUE), k))

feat_12 <- grid_12 %>%
  dplyr::group_by(response_id) %>%
  dplyr::summarise(
    dollar_avg_12m   = mean(dollar_m),
    products_avg_12m = mean(products_m),
    dollar_std_12m   = stats::sd(dollar_m),
    products_std_12m = stats::sd(products_m),
    
    dollar_top1_12m     = max(dollar_m),
    dollar_top3_avg_12m = topk_avg(dollar_m, 3),
    dollar_top6_avg_12m = topk_avg(dollar_m, 6),
    
    products_top1_12m     = max(products_m),
    products_top3_avg_12m = topk_avg(products_m, 3),
    products_top6_avg_12m = topk_avg(products_m, 6),
    
    dollar_spring_avg_12m   = mean(dollar_m[mnum %in% c(3,4,5)]),
    dollar_summer_avg_12m   = mean(dollar_m[mnum %in% c(6,7,8)]),
    dollar_autumn_avg_12m   = mean(dollar_m[mnum %in% c(9,10,11)]),
    dollar_winter_avg_12m   = mean(dollar_m[mnum %in% c(12,1,2)]),
    
    products_spring_avg_12m = mean(products_m[mnum %in% c(3,4,5)]),
    products_summer_avg_12m = mean(products_m[mnum %in% c(6,7,8)]),
    products_autumn_avg_12m = mean(products_m[mnum %in% c(9,10,11)]),
    products_winter_avg_12m = mean(products_m[mnum %in% c(12,1,2)]),
    
    months_with_purchase_12m = sum(products_m > 0),
    .groups = "drop"
  ) %>%
  dplyr::mutate(
    dollar_std_12m   = tidyr::replace_na(dollar_std_12m, 0),
    products_std_12m = tidyr::replace_na(products_std_12m, 0)
  )

feat_ratios_12 <- feat_12 %>%
  dplyr::transmute(
    response_id,
    dollar_avg12_top1_ratio    = safe_div(dollar_avg_12m,    dollar_top1_12m),
    dollar_avg12_top3_ratio    = safe_div(dollar_avg_12m,    dollar_top3_avg_12m),
    dollar_avg12_top6_ratio    = safe_div(dollar_avg_12m,    dollar_top6_avg_12m),
    products_avg12_top1_ratio  = safe_div(products_avg_12m,  products_top1_12m),
    products_avg12_top3_ratio  = safe_div(products_avg_12m,  products_top3_avg_12m),
    products_avg12_top6_ratio  = safe_div(products_avg_12m,  products_top6_avg_12m)
  )

Samsung_feats_12 <- feat_12 %>% dplyr::left_join(feat_ratios_12, by = "response_id")

# Evitar duplicados si re-ejecutas
cols_new <- setdiff(names(Samsung_feats_12), "response_id")
W_pre <- W_pre %>% dplyr::select(-dplyr::any_of(cols_new))
W_pre <- W_pre %>% dplyr::left_join(Samsung_feats_12, by = "response_id")

#-----------------------------------------------------------------------------
# V_pre (perfilado): dummies + n_states (NO winsor aquí; se hace tras split)
#-----------------------------------------------------------------------------
V_pre <- data_dum %>%
  dplyr::select(
    response_id,
    dplyr::starts_with("race_"),
    dplyr::starts_with("life_"),
    dplyr::starts_with("q_demos_age_"),
    dplyr::starts_with("q_demos_income_"),
    dplyr::starts_with("q_amazon_use_howmany_"),
    n_states
  ) %>%
  dplyr::distinct(response_id, .keep_all = TRUE) %>%
  dplyr::mutate(response_id = as.character(response_id))

if (length(anomalous_ids_severe) > 0) {
  V_pre <- V_pre %>% dplyr::filter(!response_id %in% anomalous_ids_severe)
}


#-----------------------------------------------------------------------------
# Universo maestro y alineación (V/W/var_2023) — SIN WINSOR aquí
#-----------------------------------------------------------------------------
# Completar var_2023 con 0 para IDs faltantes según intersección V/W
ids_master <- intersect(V_pre$response_id, W_pre$response_id)
stopifnot(length(ids_master) > 0)

# Orden maestro: el orden de W_pre (como venías haciendo)
ids_master <- W_pre$response_id[W_pre$response_id %in% ids_master]

var_2023_full <- tibble(response_id = ids_master) %>%
  left_join(var_2023, by = "response_id") %>%
  mutate(amount_2023_Samsung = tidyr::replace_na(amount_2023_Samsung, 0))

V_full <- V_pre[match(ids_master, V_pre$response_id), , drop = FALSE]
W_full <- W_pre[match(ids_master, W_pre$response_id), , drop = FALSE]

stopifnot(identical(V_full$response_id, ids_master))
stopifnot(identical(W_full$response_id, ids_master))
stopifnot(identical(var_2023_full$response_id, ids_master))
stopifnot(!any(is.na(var_2023_full$amount_2023_Samsung)))

V_aligned <- V_full
W_aligned <- W_full
var_2023_aligned <- var_2023_full

#-----------------------------------------------------------------------------
# Split train/test (sin fuga): primero IDs, luego subset por índices
#-----------------------------------------------------------------------------
set.seed(123)
ids <- V_aligned$response_id
n <- length(ids)

train_ids <- sample(ids, size = floor(0.8*n), replace = FALSE)
test_ids  <- setdiff(ids, train_ids)

train_idx <- match(train_ids, ids)
test_idx  <- match(test_ids,  ids)

V_train <- V_aligned[train_idx, , drop = FALSE]
V_test  <- V_aligned[test_idx,  , drop = FALSE]
W_train <- W_aligned[train_idx, , drop = FALSE]
W_test  <- W_aligned[test_idx,  , drop = FALSE]

var_train <- var_2023_aligned[train_idx, , drop = FALSE]
var_test  <- var_2023_aligned[test_idx,  , drop = FALSE]

#-----------------------------------------------------------------------------
# (2) WINSORIZACIÓN POST-SPLIT, caps SOLO con TRAIN (sin fuga)
#   - amount_2023_Samsung (para Y)
#   - W numéricas
#   - SOLO n_states en V
#-----------------------------------------------------------------------------
# ---- (a) cap de Y (amount_2023_Samsung) estimado en TRAIN ----
tk_y_tr <- tukey_cap_upper(var_train$amount_2023_Samsung, k = 3)
cap_y_train <- tk_y_tr$cap
message("Cap Tukey (k=3) TRAIN para amount_2023_Samsung: ", round(cap_y_train, 4))

var_train$amount_2023_Samsung <- tk_y_tr$x
var_test$amount_2023_Samsung  <- apply_upper_cap(var_test$amount_2023_Samsung, cap_y_train)

# ---- (b) cap en W numéricas estimado en TRAIN y aplicado a TEST ----
W_wins <- winsorize_df_numeric_upper_trainonly(
  df_train = W_train,
  df_test  = W_test,
  id_col = "response_id",
  k = 3,
  cols = NULL   # todas las numéricas excepto response_id
)
W_train <- W_wins$train
W_test  <- W_wins$test

# Cuántas features había en W antes de la selección (sin response_id)
n_before <- ncol(W_train) - 1 

# W_wins$caps queda disponible si quieres auditar

# ---- (c) cap SOLO para n_states en V (si existe) estimado en TRAIN ----
if ("n_states" %in% names(V_train) && is.numeric(V_train$n_states)) {
  tk_ns_tr <- tukey_cap_upper(V_train$n_states, k = 3)
  cap_ns_train <- tk_ns_tr$cap
  message("Cap Tukey (k=3) TRAIN para n_states: ", round(cap_ns_train, 4))
  
  V_train$n_states <- tk_ns_tr$x
  V_test$n_states  <- apply_upper_cap(V_test$n_states, cap_ns_train)
}

#-----------------------------------------------------------------------------
# Construir Y desde amount_2023_Samsung (c SOLO con TRAIN)
#  - OJO: amount_2023_Samsung ya está:
#      (i) sin severos >5000
#      (ii) winsorizada con cap TRAIN (k=3) aplicada también a test
#-----------------------------------------------------------------------------
amt_train <- var_train$amount_2023_Samsung
amt_test  <- var_test$amount_2023_Samsung

mean_train <- mean(amt_train, na.rm = TRUE)
var_trainv <- stats::var(amt_train, na.rm = TRUE)

c_train <- round(var_trainv / mean_train, 0)
if (!is.finite(c_train) || c_train <= 0) stop("c_train no es válido. Revisa mean/var en TRAIN.")

Y_train <- pmax(0L, as.integer(round(amt_train / c_train, 0)))
Y_test  <- pmax(0L, as.integer(round(amt_test  / c_train, 0)))

message("Split sizes: train=", length(Y_train), " | test=", length(Y_test),
        " | c_train=", c_train)

# -----------------------------------------------------------------------------
# Guardar IDs train/test y c_train (ya calculada)
# -----------------------------------------------------------------------------
dir.create("data/matrices", showWarnings = FALSE, recursive = TRUE)
write.csv(data.frame(response_id = as.character(train_ids)),
          file = paste0("data/matrices/ids_train", TAG, ".csv"), row.names = FALSE)
write.csv(data.frame(response_id = as.character(test_ids)),
          file = paste0("data/matrices/ids_test", TAG, ".csv"), row.names = FALSE)
saveRDS(c_train, paste0("data/matrices/c_train", TAG, ".rds"))
message("IDs guardados: train=", length(train_ids), " | test=", length(test_ids))

# -----------------------------------------------------------------------------
# SoW y Siow de TRAIN en T2 -> data/matrices/sow_siow_train_min<MIN_MONETARY_T1>.csv
# (mismo calculo que el SoW/Siow empirico de TEST que se hace mas abajo;
#  lo usa tune_xgb_sow_siow.py)
# -----------------------------------------------------------------------------
build_sow_siow <- function(ids, purchases_tech2, ref_date3, ref_date4,
                           brand = "Samsung", c_train) {
  ids <- as.character(ids)
  
  out <- purchases_tech2 %>%
    dplyr::filter(
      dplyr::between(order_date, ref_date3, ref_date4),
      response_id %in% ids
    ) %>%
    dplyr::group_by(response_id) %>%
    dplyr::summarise(
      amount_brand = sum(dplyr::if_else(tech_brand == brand, total_price, 0), na.rm = TRUE),
      amount_other = sum(dplyr::if_else(tech_brand != brand, total_price, 0), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::right_join(tibble::tibble(response_id = ids), by = "response_id") %>%
    dplyr::mutate(
      amount_brand = tidyr::replace_na(amount_brand, 0),
      amount_other = tidyr::replace_na(amount_other, 0),
      amount_total = amount_brand + amount_other,
      siow = pmax(0L, as.integer(round(amount_total / c_train, 0))),
      sow  = dplyr::if_else(amount_total > 0, amount_brand / amount_total, 0)
    ) %>%
    dplyr::select(response_id, sow, siow)
  
  # mismo orden que ids
  out[match(ids, out$response_id), ]
}

sow_train_df <- build_sow_siow(train_ids, purchases_tech2, ref_date3, ref_date4,
                               "Samsung", c_train)
stopifnot(
  nrow(sow_train_df) == length(train_ids),
  identical(as.character(sow_train_df$response_id), as.character(train_ids))
)

sow_train_file <- paste0("data/matrices/sow_siow_train", TAG, ".csv")
write.csv(sow_train_df, sow_train_file, row.names = FALSE)
message("Guardado: ", sow_train_file,
        " | n=", nrow(sow_train_df),
        " | % ceros SoW train=", round(100 * mean(sow_train_df$sow == 0), 1))

# Entradas de T2 para siow_sow.R (no dependen del umbral; sirven para min0, min30, ...)
saveRDS(list(purchases_tech2 = purchases_tech2,
             ref_date3 = ref_date3,
             ref_date4 = ref_date4),
        "data/matrices/t2_inputs.rds")

dim(V_train); dim(W_train); length(Y_train)
dim(V_test);  dim(W_test);  length(Y_test)


# ------------------------------------------------------------
# Plot response variable distributions (IGUAL que tu bloque)
# ------------------------------------------------------------
fill_grey   <- "grey85"
border_grey <- "grey60"
line_grey   <- "grey30"
text_grey   <- "grey30"

plot_bar_with_labels <- function(vec, main, col,
                                 xlab = "Y",
                                 ylab = "Frequency",
                                 border_col = border_grey,
                                 text_col   = text_grey) {
  counts <- table(vec)
  
  op <- par(font      = par("font"),
            font.main = par("font.main"),
            font.lab  = par("font.lab"),
            font.axis = par("font.axis"),
            col.axis  = par("col.axis"),
            fg        = par("fg"),
            family    = par("family"))
  on.exit(par(op))
  
  par(font = 1, font.main = 1, font.lab = 1, font.axis = 1,
      col.axis = text_col,
      fg       = border_col)
  
  bp <- barplot(
    counts,
    main   = "", xlab = "", ylab = "",
    col    = col,
    border = border_col,
    ylim   = c(0, max(counts) * 1.2)
  )
  
  text(bp, counts, labels = counts, pos = 3, cex = 0.7, col = text_col)
  
  mtext(main, side = 3, line = 1.2, font = 1, col = text_col)
  mtext(xlab, side = 1, line = 3,   font = 1, col = text_col)
  mtext(ylab, side = 2, line = 3,   font = 1, col = text_col)
}

graphics.off()
grDevices::dev.new()
plot_bar_with_labels(Y_train, "Training set", "grey85", xlab = "")
plot_bar_with_labels(Y_test,  "Test set",     "grey85")

#-----------------------------------------------------------------------------
# (G) SOW empírico en T2 (validación)  +  GRÁFICOS + EXPORT
#-----------------------------------------------------------------------------
# 1) Universo SoW empírico = IDs del TEST (intersección consistente)
ids_sow_test <- Reduce(intersect, list(W_test$response_id, V_test$response_id))
ids_sow_test <- sort(as.character(ids_sow_test))
stopifnot(length(ids_sow_test) > 0)

# Helper: reordenar DF a un vector de ids (manteniendo todos los ids si se pide)
reorder_by_ids <- function(df, ids, id_col = "response_id", keep_all_ids = TRUE) {
  stopifnot(id_col %in% names(df))
  ids <- as.character(ids)
  df[[id_col]] <- as.character(df[[id_col]])
  
  if (keep_all_ids) {
    miss <- setdiff(ids, df[[id_col]])
    if (length(miss) > 0) {
      df <- dplyr::bind_rows(df, tibble::tibble(!!id_col := miss))
    }
  }
  
  out <- df[match(ids, df[[id_col]]), , drop = FALSE]
  rownames(out) <- NULL
  out
}

# 2) Filtrar transacciones T2 SOLO para esos ids del TEST
period_df <- purchases_tech2 %>%
  dplyr::filter(
    dplyr::between(order_date, ref_date3, ref_date4),
    response_id %in% ids_sow_test
  )

# 3) Calcular SoW empírico y mantener universo COMPLETO del TEST
Samsung_vs_others <- period_df %>%
  dplyr::group_by(response_id) %>%
  dplyr::summarise(
    amount_Samsung       = sum(dplyr::if_else(tech_brand == "Samsung", total_price, 0), na.rm = TRUE),
    amount_otherbrands = sum(dplyr::if_else(tech_brand != "Samsung", total_price, 0), na.rm = TRUE),
    n_total            = sum(quantity, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::right_join(tibble::tibble(response_id = ids_sow_test), by = "response_id") %>%
  dplyr::mutate(
    amount_Samsung       = tidyr::replace_na(amount_Samsung, 0),
    amount_otherbrands = tidyr::replace_na(amount_otherbrands, 0),
    n_total            = tidyr::replace_na(n_total, 0),
    amount_total=amount_Samsung + amount_otherbrands,
    aux=(amount_Samsung + amount_otherbrands)/c_train,
    siow       = round(aux,0),
    sow = dplyr::if_else(amount_total == 0, 0, amount_Samsung / amount_total),
    # Nota: si querías "avg_Samsung" como ticket promedio, debe ser amount_total/n_total
    avg_Samsung          = dplyr::if_else(n_total > 0, siow / n_total, 0)
  ) %>%
  dplyr::select(
    response_id,
    amount_Samsung, 
    amount_otherbrands,
    siow, 
    sow
  )

# 4) Reordenar exactamente en el mismo orden de ids_sow_test
Samsung_vs_others2 <- reorder_by_ids(Samsung_vs_others, ids_sow_test, keep_all_ids = TRUE)
stopifnot(nrow(Samsung_vs_others2) == length(ids_sow_test))

# (Opcional) Guardar vectores como antes
T2.siow      <- Samsung_vs_others2$siow
T2.sow <- Samsung_vs_others2$sow

#-----------------------------------------------------------------------------
# GRÁFICOS SoW empírico (3): hist completo, 0 vs >0, hist positivos
#-----------------------------------------------------------------------------
theme_clean_grey <- ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(
    panel.grid       = ggplot2::element_blank(),
    panel.background = ggplot2::element_blank(),
    plot.background  = ggplot2::element_blank(),
    axis.line        = ggplot2::element_line(color = "grey60"),
    axis.ticks       = ggplot2::element_line(color = "grey60"),
    axis.text        = ggplot2::element_text(color = "grey20"),
    axis.title       = ggplot2::element_text(color = "grey20"),
    plot.title       = ggplot2::element_text(color = "grey20")
  )

fill_grey   <- "grey85"
border_grey <- "grey60"
line_grey   <- "grey30"
text_grey   <- "grey20"

df <- Samsung_vs_others2 %>%
  dplyr::transmute(share = pmin(pmax(sow, 0), 1)) %>%
  dplyr::filter(!is.na(share))

m_all  <- mean(df$share, na.rm = TRUE)
md_all <- median(df$share, na.rm = TRUE)
dx <- 0.02

p_all <- ggplot2::ggplot(df, ggplot2::aes(x = share)) +
  ggplot2::geom_histogram(
    ggplot2::aes(y = ggplot2::after_stat(density)),
    binwidth = 0.01, boundary = 0, closed = "right",
    fill = fill_grey, color = border_grey
  ) +
  ggplot2::geom_vline(xintercept = m_all,  linetype = "dashed", color = line_grey) +
  ggplot2::geom_vline(xintercept = md_all, linetype = "dotted", color = line_grey) +
  ggplot2::annotate(
    "text", x = min(1, m_all + dx), y = Inf, vjust = 1.5, color = text_grey,
    label = paste0("Mean = ", scales::percent(m_all, accuracy = 0.1))
  ) +
  ggplot2::annotate(
    "text", x = min(1, md_all + dx), y = Inf, vjust = 3.0, color = text_grey,
    label = paste0("Median = ", scales::percent(md_all, accuracy = 0.1))
  ) +
  ggplot2::coord_cartesian(xlim = c(0, 1)) +
  ggplot2::scale_x_continuous(labels = scales::percent) +
  ggplot2::labs(
    title = "Samsung SoW (Amz) in T2 — Test (zeros included)",
    x = "Samsung SoW", y = "Density"
  ) +
  theme_clean_grey

tot_n <- nrow(df)
counts_df <- df %>%
  dplyr::summarise(
    zeros = sum(share == 0),
    positives = sum(share > 0),
    .groups = "drop"
  ) %>%
  tidyr::pivot_longer(dplyr::everything(), names_to = "group", values_to = "n") %>%
  dplyr::mutate(
    pct   = n / tot_n,
    group = dplyr::recode(group, zeros = "SoW = 0", positives = "SoW > 0")
  )

y_max <- max(counts_df$n)
counts_df <- counts_df %>% dplyr::mutate(y_label = n + pmax(12, 0.06 * y_max))

p_counts <- ggplot2::ggplot(counts_df, ggplot2::aes(x = group, y = n)) +
  ggplot2::geom_col(width = 0.6, fill = fill_grey, color = border_grey) +
  ggplot2::geom_text(
    ggplot2::aes(y = y_label,
                 label = paste0(scales::comma(n), " (", scales::percent(pct, accuracy = 0.1), ")")),
    color = text_grey, vjust = 0
  ) +
  ggplot2::scale_y_continuous(
    limits = c(0, max(counts_df$y_label) * 1.08),
    expand = ggplot2::expansion(mult = c(0, 0))
  ) +
  ggplot2::coord_cartesian(clip = "off") +
  ggplot2::labs(
    title = "Samsung SoW (Amz) in T2 — Test: zeros vs positives",
    x = NULL, y = "Count"
  ) +
  theme_clean_grey +
  ggplot2::theme(plot.margin = ggplot2::margin(5, 25, 5, 5))

pos <- df %>% dplyr::filter(share > 0)
m_pos  <- mean(pos$share, na.rm = TRUE)
md_pos <- median(pos$share, na.rm = TRUE)

p_pos <- ggplot2::ggplot(pos, ggplot2::aes(x = share)) +
  ggplot2::geom_histogram(
    ggplot2::aes(y = ggplot2::after_stat(density)),
    binwidth = 0.02, boundary = 0, closed = "right",
    fill = fill_grey, color = border_grey
  ) +
  ggplot2::geom_vline(xintercept = m_pos,  linetype = "dashed", color = line_grey) +
  ggplot2::geom_vline(xintercept = md_pos, linetype = "dotted", color = line_grey) +
  ggplot2::annotate(
    "text", x = min(1, m_pos + dx), y = Inf, vjust = 1.5, color = text_grey,
    label = paste0("Mean = ", scales::percent(m_pos, accuracy = 0.1))
  ) +
  ggplot2::annotate(
    "text", x = min(1, md_pos + dx), y = Inf, vjust = 3.0, color = text_grey,
    label = paste0("Median = ", scales::percent(md_pos, accuracy = 0.1))
  ) +
  ggplot2::coord_cartesian(xlim = c(0, 1)) +
  ggplot2::scale_x_continuous(labels = scales::percent) +
  ggplot2::labs(
    title = "Samsung SoW (Amz) in T2 — Test (positives only)",
    x = "Samsung SoW", y = "Density"
  ) +
  theme_clean_grey

print(p_all)
print(p_counts)
print(p_pos)

(p_all / p_counts / p_pos) + patchwork::plot_layout(heights = c(0.45, 0.25, 0.30))

#-----------------------------------------------------------------------------
# EXPORT empirical_sow.xlsx (tal como tu referencia)
#-----------------------------------------------------------------------------

# Crear workbook
wb <- createWorkbook()

# Hoja ID
addWorksheet(wb, "id")
writeData(
  wb,
  sheet = "id",
  x = data.frame(response_id = Samsung_vs_others2$response_id)
)

# Hoja SOW
addWorksheet(wb, "sow")
writeData(
  wb,
  sheet = "sow",
  x = data.frame(sow = Samsung_vs_others2$sow)
)

# Hoja SIOW
addWorksheet(wb, "siow")
writeData(
  wb,
  sheet = "siow",
  x = data.frame(siow = Samsung_vs_others2$siow)
)

# Guardar archivo
saveWorkbook(
  wb,
  file = paste0("data/matrices/empirical_sow", TAG, ".xlsx"),
  overwrite = TRUE
)


#-----------------------------------------------------------------------------
# Bubble plot: Customers by SoW bins and SioW quintiles (tu versión 2)
#-----------------------------------------------------------------------------
fill_grey   <- "grey85"
border_grey <- "grey60"
line_grey   <- "grey30"
text_grey   <- "grey20"

theme_clean_grey <- theme_minimal(base_size = 14) +
  theme(
    panel.grid       = element_blank(),
    panel.background = element_blank(),
    plot.background  = element_blank(),
    axis.line        = element_line(color = border_grey),
    axis.ticks       = element_line(color = border_grey),
    axis.text        = element_text(color = text_grey),
    axis.title       = element_text(color = text_grey),
    plot.title       = element_text(color = text_grey),
    plot.subtitle    = element_text(color = text_grey),
    legend.text      = element_text(color = text_grey),
    legend.title     = element_text(color = text_grey)
  )

breaks_sow <- c(0, .2, .4, .6, .8, 1)
labs_sow   <- c("[0, .2]", "(.2, .4]", "(.4, .6]", "(.6, .8]", "(.8, 1]")

plot_df <- Samsung_vs_others2 %>%
  transmute(
    sow = pmin(pmax(sow, 0), 1),
    tw  = siow # antes amount_total
  ) %>%
  mutate(
    sow_bin = cut(sow, breaks = breaks_sow, include.lowest = TRUE,
                  right = TRUE, labels = labs_sow),
    tw_q = ntile(tw, 5)
  ) %>%
  filter(!is.na(sow_bin)) %>%
  group_by(tw_q, sow_bin) %>%
  summarise(
    n       = n(),
    avg_usd = mean(tw, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    tw_lab = factor(
      case_when(
        tw_q == 1 ~ "Bottom quintile",
        tw_q == 2 ~ "2d quintile",
        tw_q == 3 ~ "3d quintile",
        tw_q == 4 ~ "4th quintile",
        tw_q == 5 ~ "Top quintile"
      ),
      levels = c("Top quintile","4th quintile","3d quintile","2d quintile","Bottom quintile")
    ),
    sow_lab   = factor(sow_bin, levels = labs_sow),
    label_txt = dollar(avg_usd, accuracy = 0.01)
  ) %>%
  mutate(
    n_bin = cut(
      n,
      breaks = c(-Inf, 5, 10, 20, 70, Inf),
      labels = c("≤5", "6–10", "11–20", "21–70", ">70"),
      right  = TRUE
    )
  )

x_labs <- labs_sow
x_labs[1] <- paste0("Small\n", x_labs[1])
x_labs[length(x_labs)] <- paste0("Large\n", x_labs[length(x_labs)])

p <- ggplot(plot_df, aes(x = sow_lab, y = tw_lab)) +
  geom_point(aes(size = n_bin),
             shape = 21, fill = fill_grey, color = border_grey, stroke = 0.6) +
  geom_text(aes(label = label_txt), size = 3, color = text_grey) +
  scale_size_manual(
    name   = "Count",
    values = c(11, 15, 19, 23, 27),
    breaks = c("≤5","6–10","11–20","21–70",">70"),
    guide  = guide_legend(
      override.aes = list(shape = 21, fill = fill_grey, color = border_grey)
    )
  ) +
  scale_x_discrete(labels = x_labs, expand = ggplot2::expansion(add = c(0.35, 0.15))) +
  scale_y_discrete(limits = rev(levels(plot_df$tw_lab))) +
  labs(
    title = "Customers by SoW (deciles) and SioW (quintiles)",
    x = "SoW, 0.1-width bins",
    y = "SioW, quintiles"
  ) +
  coord_cartesian(clip = "off") +
  theme_clean_grey +
  theme(
    plot.title  = element_text(margin = ggplot2::margin(b = 8)),
    plot.margin = ggplot2::margin(t = 10, r = 10, b = 16, l = 10)
  )

print(p)

plot_df %>%
  dplyr::arrange(sow_lab, tw_lab) %>%
  dplyr::select(sow_lab, tw_lab, n, n_bin) %>%
  print(n = Inf)

table(plot_df$n_bin, useNA = "ifany")

lvl1 <- levels(plot_df$sow_lab)[1]
plot_df %>%
  dplyr::filter(sow_lab == lvl1) %>%
  dplyr::select(tw_lab, n, n_bin)

# -----------------------------------------------------------------------------
# Firmas de marca (mantengo tu bloque; solo cuido ids_w_test definidos)
# -----------------------------------------------------------------------------
ids_w_train <- W_aligned$response_id[train_idx]
ids_w_test  <- W_aligned$response_id[test_idx]

data1_marcas_2023 <- survey_enriched2 %>%
  select(response_id, starts_with("2023_")) %>%
  filter(response_id %in% ids_w_test)

pos1 <- which(data1_marcas_2023$`2023_brand_Samsung` != 0)
pos2 <- which(data1_marcas_2023$`2023_brand_Samsung` == 0)
stopifnot(length(pos1) + length(pos2) == nrow(data1_marcas_2023))

data1_prop <- data1_marcas_2023 %>%
  select(response_id, starts_with("2023_brand_")) %>%
  rowwise() %>%
  mutate(total_compras = sum(c_across(starts_with("2023_brand_")), na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(across(
    starts_with("2023_brand_"),
    ~ if_else(total_compras > 0, .x / total_compras, 0),
    .names = "{.col}_prop"
  ))

ids_interes <- data1_marcas_2023$response_id[pos1]

brand_cols <- grep("^2023_brand_", names(survey_enriched2), value = TRUE)

df_counts <- survey_enriched2 %>%
  select(response_id, all_of(brand_cols)) %>%
  mutate(across(all_of(brand_cols), ~ replace_na(as.numeric(.x), 0))) %>%
  mutate(total_units = rowSums(across(all_of(brand_cols)), na.rm = TRUE))

df_long <- df_counts %>%
  pivot_longer(cols = all_of(brand_cols),
               names_to = "brand", values_to = "units") %>%
  mutate(
    brand      = gsub("^2023_brand_", "", brand),
    proportion = if_else(total_units > 0, units / total_units, 0)
  ) %>%
  filter(response_id %in% ids_interes, units > 0)

sig_tbl <- df_long %>%
  filter(brand != "Samsung") %>%
  group_by(response_id) %>%
  summarise(
    signature = {
      br <- sort(unique(brand[units > 0]))
      if (length(br) == 0) "NONE" else paste(br, collapse = " + ")
    },
    .groups = "drop"
  )

df_long <- df_long %>%
  left_join(sig_tbl, by = "response_id") %>%
  filter(signature != "NONE")

users_by_sig <- df_long %>%
  distinct(response_id, signature) %>%
  arrange(signature, response_id)

panel_groups <- list()
for (sig in unique(users_by_sig$signature)) {
  ids_sig <- users_by_sig %>% filter(signature == sig) %>% pull(response_id)
  if (length(ids_sig) == 0) next
  chunks <- split(ids_sig, ceiling(seq_along(ids_sig) / 3))
  for (ch in chunks) {
    panel_groups <- append(panel_groups, list(list(ids = ch, signature = sig)))
  }
}

plots <- lapply(seq_along(panel_groups), function(i) {
  meta <- panel_groups[[i]]
  df_plot <- df_long %>%
    filter(response_id %in% meta$ids, units > 0)
  
  if (nrow(df_plot) == 0) return(NULL)
  
  ggplot(df_plot,
         aes(x = reorder(brand, -proportion), y = proportion, fill = brand)) +
    geom_col(width = 0.4) +
    geom_text(
      aes(label = paste0(scales::percent(proportion, accuracy = 1),
                         " (", units, ")")),
      hjust = -0.1, size = 3
    ) +
    facet_wrap(~ response_id, scales = "free_x", nrow = 1) +
    coord_flip(clip = "off") +
    labs(
      title = paste0("Users grouped by same non-Samsung brand set: ", meta$signature),
      x = "Brand",
      y = "Purchase proportion",
      fill = "Brand"
    ) +
    theme_minimal() +
    theme(
      panel.spacing    = unit(2, "lines"),
      plot.margin      = margin(5, 40, 5, 5),
      legend.position  = "bottom",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
})

if (dev.cur() != 1) dev.off()
for (i in seq_along(plots)) print(plots[[i]])

pdf("WithSamsung_buyers_plots_testdata.pdf", width = 12, height = 6)
invisible(lapply(plots, function(p) if (!is.null(p)) print(p)))
dev.off()
while (dev.cur() > 1) dev.off()


# -----------------------------------------------------------------------------
# FULL COLUMN RANK + Spearman en W (col>=5) + estandarización W (TRAIN->TEST)
# -----------------------------------------------------------------------------
as_numeric_matrix <- function(df, id_col = "response_id") {
  stopifnot(id_col %in% names(df))
  X <- df %>% dplyr::select(-dplyr::all_of(id_col))
  X <- X %>% dplyr::mutate(dplyr::across(dplyr::everything(), ~ suppressWarnings(as.numeric(.x))))
  as.matrix(X)
}

rank_report <- function(df, id_col = "response_id", tol = 1e-10) {
  X <- as_numeric_matrix(df, id_col)
  X[is.na(X)] <- 0
  q <- qr(X, tol = tol)
  r <- q$rank
  list(
    n = nrow(X), p = ncol(X),
    rank = r,
    full_col_rank = (r == ncol(X)),
    deficiency = ncol(X) - r,
    kappa = tryCatch(kappa(X), error = function(e) NA_real_)
  )
}

drop_allzero_constant_nzv <- function(df, id_col = "response_id", nzv = TRUE) {
  X <- df %>% dplyr::select(-dplyr::all_of(id_col))
  X <- X %>% dplyr::select(where(is.numeric))
  X0 <- X; X0[is.na(X0)] <- 0
  
  all_zero <- names(X0)[vapply(X0, function(z) all(z == 0), logical(1))]
  constant <- names(X0)[vapply(X0, function(z) length(unique(z)) == 1, logical(1))]
  
  keep <- setdiff(names(X0), union(all_zero, constant))
  X1 <- X0[, keep, drop = FALSE]
  
  nzv_cols <- character(0)
  if (nzv && ncol(X1) > 0) {
    if (!requireNamespace("caret", quietly = TRUE)) {
      message("Paquete 'caret' no está disponible; saltando nearZeroVar().")
    } else {
      nzv_idx <- caret::nearZeroVar(X1)
      if (length(nzv_idx) > 0) nzv_cols <- colnames(X1)[nzv_idx]
      keep2 <- setdiff(colnames(X1), nzv_cols)
      X1 <- X1[, keep2, drop = FALSE]
    }
  }
  
  list(
    kept = colnames(X1),
    removed = list(
      all_zero = all_zero,
      constant = setdiff(constant, all_zero),
      nzv = nzv_cols
    )
  )
}

prep_V_lastcol_scale <- function(V_train, V_test,
                                 id_col = "response_id",
                                 last_col_name = NULL,
                                 nzv = TRUE) {
  stopifnot(id_col %in% names(V_train), id_col %in% names(V_test))
  
  feats <- setdiff(names(V_train), id_col)
  if (is.null(last_col_name)) last_col_name <- tail(feats, 1)
  stopifnot(last_col_name %in% names(V_train))
  
  keep_info <- drop_allzero_constant_nzv(
    df = V_train %>% dplyr::select(all_of(id_col), all_of(feats)),
    id_col = id_col,
    nzv = nzv
  )
  # Candidatas que sobrevivieron limpieza + correlación
  kept <- keep_info$kept
  length(kept)
  Vtr <- V_train %>% dplyr::select(all_of(id_col), all_of(kept))
  Vte <- V_test  %>% dplyr::select(all_of(id_col), all_of(kept))
  
  # Estándarizar SOLO última columna, con parámetros TRAIN
  if (last_col_name %in% kept) {
    mu  <- mean(Vtr[[last_col_name]], na.rm = TRUE)
    sdv <- stats::sd(Vtr[[last_col_name]], na.rm = TRUE)
    if (is.finite(sdv) && sdv > 0) {
      Vtr[[last_col_name]] <- (Vtr[[last_col_name]] - mu) / sdv
      Vte[[last_col_name]] <- (Vte[[last_col_name]] - mu) / sdv
    } else {
      warning("La última columna '", last_col_name, "' tiene sd=0 en TRAIN; no se estandariza.")
    }
  } else {
    warning("La última columna '", last_col_name, "' fue removida por all-zero/constant/NZV en TRAIN.")
  }
  
  list(
    V_train2 = Vtr,
    V_test2  = Vte,
    kept = kept,
    removed = keep_info$removed,
    last_col = last_col_name
  )
}

scale_W_cols_train_only <- function(W_train, W_test, cols_to_scale, id_col = "response_id") {
  stopifnot(id_col %in% names(W_train), id_col %in% names(W_test))
  cols_to_scale <- intersect(cols_to_scale, setdiff(names(W_train), id_col))
  
  if (length(cols_to_scale) == 0) {
    return(list(W_train_sc = W_train, W_test_sc = W_test,
                mu = numeric(0), sd = numeric(0), scaled_cols = character(0)))
  }
  
  Wtr <- W_train
  Wte <- W_test
  Wtr[cols_to_scale] <- lapply(Wtr[cols_to_scale], function(x) suppressWarnings(as.numeric(x)))
  Wte[cols_to_scale] <- lapply(Wte[cols_to_scale], function(x) suppressWarnings(as.numeric(x)))
  
  mu  <- vapply(Wtr[cols_to_scale], function(x) mean(x, na.rm = TRUE), numeric(1))
  sdv <- vapply(Wtr[cols_to_scale], function(x) stats::sd(x, na.rm = TRUE), numeric(1))
  
  ok <- is.finite(sdv) & sdv > 0
  scaled_cols <- cols_to_scale[ok]
  
  for (cc in scaled_cols) {
    Wtr[[cc]] <- (Wtr[[cc]] - mu[cc]) / sdv[cc]
    Wte[[cc]] <- (Wte[[cc]] - mu[cc]) / sdv[cc]
  }
  
  list(W_train_sc = Wtr, W_test_sc = Wte, mu = mu, sd = sdv, scaled_cols = scaled_cols)
}

select_W_from5_spearman <- function(W_train, W_test,
                                    id_col = "response_id",
                                    keep_first_k = 4,
                                    cutoff = 0.90,
                                    nzv = TRUE,
                                    scale_selected = TRUE) {
  stopifnot(id_col %in% names(W_train), id_col %in% names(W_test))
  
  feats <- setdiff(names(W_train), id_col)
  stopifnot(setequal(feats, setdiff(names(W_test), id_col)))
  
  # Fijas 1:4 (después del ID), según orden en W_train
  fixed <- feats[seq_len(min(keep_first_k, length(feats)))]
  cand  <- setdiff(feats, fixed)
  
  # Limpieza SOLO en candidatas (train)
  cand_df  <- W_train %>% dplyr::select(all_of(id_col), all_of(cand))
  keep_inf <- drop_allzero_constant_nzv(cand_df, id_col = id_col, nzv = nzv)
  cand_kept <- keep_inf$kept
  
  # Si <=1 candidata, no hay correlación que resolver
  if (length(cand_kept) <= 1) {
    selected <- c(fixed, cand_kept)
    Wtr_sel <- W_train %>% dplyr::select(all_of(id_col), all_of(selected))
    Wte_sel <- W_test  %>% dplyr::select(all_of(id_col), all_of(selected))
    
    sc <- if (scale_selected) scale_W_cols_train_only(Wtr_sel, Wte_sel, selected, id_col) else
      list(W_train_sc=Wtr_sel, W_test_sc=Wte_sel, mu=numeric(0), sd=numeric(0), scaled_cols=character(0))
    
    return(list(
      W_train_sel = sc$W_train_sc,
      W_test_sel  = sc$W_test_sc,
      fixed = fixed,
      kept_from5 = cand_kept,
      dropped_bad = keep_inf$removed,
      dropped_corr = character(0),
      scaled_cols = sc$scaled_cols,
      scale_params = list(mu = sc$mu, sd = sc$sd)
    ))
  }
  
  # Spearman en candidatas (TRAIN)
  X <- W_train %>%
    dplyr::select(all_of(cand_kept)) %>%
    dplyr::mutate(dplyr::across(dplyr::everything(), ~ suppressWarnings(as.numeric(.x)))) %>%
    as.matrix()
  X[is.na(X)] <- 0
  
  cor_mat <- suppressWarnings(stats::cor(X, method = "spearman", use = "pairwise.complete.obs"))
  diag(cor_mat) <- 1
  
  # Remover por correlación alta: caret::findCorrelation
  if (!requireNamespace("caret", quietly = TRUE)) stop("Necesitas el paquete 'caret'.")
  drop_idx <- caret::findCorrelation(cor_mat, cutoff = cutoff, names = FALSE, exact = TRUE)
  dropped_corr <- if (length(drop_idx) > 0) colnames(cor_mat)[drop_idx] else character(0)
  
  kept_from5 <- setdiff(cand_kept, dropped_corr)
  selected <- c(fixed, kept_from5)
  
  Wtr_sel <- W_train %>% dplyr::select(all_of(id_col), all_of(selected))
  Wte_sel <- W_test  %>% dplyr::select(all_of(id_col), all_of(selected))
  
  # Estandarizar W (TRAIN) y aplicar a TEST
  sc <- if (scale_selected) scale_W_cols_train_only(Wtr_sel, Wte_sel, selected, id_col) else
    list(W_train_sc=Wtr_sel, W_test_sc=Wte_sel, mu=numeric(0), sd=numeric(0), scaled_cols=character(0))
  
  list(
    W_train_sel = sc$W_train_sc,
    W_test_sel  = sc$W_test_sc,
    fixed = fixed,
    kept_from5 = kept_from5,
    dropped_bad = keep_inf$removed,
    dropped_corr = dropped_corr,
    cutoff = cutoff,
    scaled_cols = sc$scaled_cols,
    scale_params = list(mu = sc$mu, sd = sc$sd)
  )
}


make_fullrank_VW <- function(V_train, V_test, W_train, W_test,
                             id_col = "response_id",
                             V_last_col = NULL,
                             W_keep_first_k = 4,
                             W_cutoff = 0.90,
                             nzv = TRUE,
                             scale_W = TRUE) {
  
  # ------------------------------------------------------------
  # Chequeos de alineación
  # ------------------------------------------------------------
  stopifnot(identical(as.character(V_train[[id_col]]), as.character(W_train[[id_col]])))
  stopifnot(identical(as.character(V_test[[id_col]]),  as.character(W_test[[id_col]])))
  
  # ============================================================
  # (A) V: NO eliminar columnas
  #     - solo escalar (opcional) una columna específica
  # ============================================================
  Vtr <- V_train
  Vte <- V_test
  
  feats_V <- setdiff(names(Vtr), id_col)
  if (is.null(V_last_col)) V_last_col <- tail(feats_V, 1)
  stopifnot(V_last_col %in% names(Vtr))
  
  # Escalado SOLO de la última columna (TRAIN → TEST)
  mu_v <- NA_real_
  sd_v <- NA_real_
  
  if (is.numeric(Vtr[[V_last_col]])) {
    mu_v <- mean(Vtr[[V_last_col]], na.rm = TRUE)
    sd_v <- stats::sd(Vtr[[V_last_col]], na.rm = TRUE)
    
    if (is.finite(sd_v) && sd_v > 0) {
      Vtr[[V_last_col]] <- (Vtr[[V_last_col]] - mu_v) / sd_v
      Vte[[V_last_col]] <- (Vte[[V_last_col]] - mu_v) / sd_v
    } else {
      warning("V_last_col tiene sd=0 en TRAIN; no se estandariza.")
    }
  } else {
    warning("V_last_col no es numérica; no se estandariza.")
  }
  
  # ============================================================
  # (B) W: selección + limpieza SOLO desde la 5ª columna
  #     - RFM (1:W_keep_first_k) es intocable
  # ============================================================
  Wres <- select_W_from5_spearman(
    W_train, W_test,
    id_col = id_col,
    keep_first_k = W_keep_first_k,
    cutoff = W_cutoff,
    nzv = nzv,
    scale_selected = scale_W
  )
  
  # ============================================================
  # (C) Reporte de rango (diagnóstico, NO forzado)
  # ============================================================
  rep <- list(
    V_train = rank_report(Vtr, id_col),
    V_test  = rank_report(Vte, id_col),
    W_train = rank_report(Wres$W_train_sel, id_col),
    W_test  = rank_report(Wres$W_test_sel,  id_col)
  )
  
  # ============================================================
  # Output
  # ============================================================
  list(
    V_train2 = Vtr,
    V_test2  = Vte,
    W_train2 = Wres$W_train_sel,
    W_test2  = Wres$W_test_sel,
    
    report_rank = rep,
    
    # --- V: no se elimina nada
    removed_V_bad = NULL,
    V_last_col = V_last_col,
    V_scale_params = list(mu = mu_v, sd = sd_v),
    
    # --- W: limpieza controlada
    removed_W_bad   = Wres$dropped_bad,
    dropped_W_corr  = Wres$dropped_corr,
    kept_W_from5    = Wres$kept_from5,
    fixed_W_1to4    = Wres$fixed,
    W_scaled_cols   = Wres$scaled_cols,
    W_scale_params  = Wres$scale_params
  )
}


# ======================
# USO: 
#  - V: NO se eliminan columnas (solo escalado opcional de 1 variable)
#  - W: selección Spearman SOLO desde la 5ª columna
#       (RFM 1..4 intocables) + escalado TRAIN → TEST
# ======================

res <- make_fullrank_VW(
  V_train = V_train,
  V_test  = V_test,
  W_train = W_train,
  W_test  = W_test,
  id_col = "response_id",
  
  # --- V ---
  # Si quieres escalar explícitamente n_states, decláralo;
  # si lo dejas en NULL, se escala la última columna de V
  V_last_col = "n_states",   # recomendado por claridad
  
  # --- W ---
  W_keep_first_k = 4,        # RFM: intocables
  W_cutoff       = 0.90,     # Spearman (solo candidatas)
  nzv            = TRUE,     # NZV solo en W (>= 5)
  scale_W        = TRUE      # escalado TRAIN → TEST
)



V_train2 <- res$V_train2
V_test2  <- res$V_test2
W_train2 <- res$W_train2
W_test2  <- res$W_test2

# Reporte de rango
res$report_rank
cat("V_train full col rank:", res$report_rank$V_train$full_col_rank,
    " def:", res$report_rank$V_train$deficiency, "\n")
cat("W_train full col rank:", res$report_rank$W_train$full_col_rank,
    " def:", res$report_rank$W_train$deficiency, "\n")
cat("W columnas estandarizadas (TRAIN->TEST):", length(res$W_scaled_cols), "\n")
head(res$W_scaled_cols, 20)


#-----------------------------------------------------------------------------
# Export matrices to Excel (con intercepto explícito)
#-----------------------------------------------------------------------------

# 1) Quitar response_id (dejamos solo numéricas)
V_train_mat <- V_train2 %>% dplyr::select(-response_id)
V_test_mat  <- V_test2  %>% dplyr::select(-response_id)
W_train_mat <- W_train2 %>% dplyr::select(-response_id)
W_test_mat  <- W_test2  %>% dplyr::select(-response_id)

# 2) Forzar numéricas (por seguridad) y NA->0 (si algo quedó suelto)
to_num_mat <- function(df) {
  out <- df %>%
    dplyr::mutate(dplyr::across(dplyr::everything(), ~ suppressWarnings(as.numeric(.x))))
  out <- out %>% dplyr::mutate(dplyr::across(dplyr::everything(), ~ tidyr::replace_na(.x, 0)))
  out
}

V_train_mat <- to_num_mat(V_train_mat)
V_test_mat  <- to_num_mat(V_test_mat)
W_train_mat <- to_num_mat(W_train_mat)
W_test_mat  <- to_num_mat(W_test_mat)

# 3) Agregar intercepto (columna de 1s) como PRIMERA columna
add_intercept <- function(df, name = "Intercept") {
  df <- dplyr::relocate(df, dplyr::everything())  # no-op, solo explícito
  cbind(setNames(data.frame(rep(1, nrow(df))), name), df)
}

V_train_mat <- add_intercept(V_train_mat, "Intercept")
V_test_mat  <- add_intercept(V_test_mat,  "Intercept")
W_train_mat <- add_intercept(W_train_mat, "Intercept")
W_test_mat  <- add_intercept(W_test_mat,  "Intercept")

# 4) Checks rápidos
stopifnot(ncol(V_train_mat) == ncol(V_test_mat))
stopifnot(ncol(W_train_mat) == ncol(W_test_mat))
stopifnot(all(V_train_mat$Intercept == 1), all(V_test_mat$Intercept == 1))
stopifnot(all(W_train_mat$Intercept == 1), all(W_test_mat$Intercept == 1))

# 5) Exportar
wb <- openxlsx::createWorkbook()

openxlsx::addWorksheet(wb, "V_train")
openxlsx::writeData(wb, "V_train", as.data.frame(V_train_mat))

openxlsx::addWorksheet(wb, "V_test")
openxlsx::writeData(wb, "V_test", as.data.frame(V_test_mat))

openxlsx::addWorksheet(wb, "W_train")
openxlsx::writeData(wb, "W_train", as.data.frame(W_train_mat))

openxlsx::addWorksheet(wb, "W_test")
openxlsx::writeData(wb, "W_test", as.data.frame(W_test_mat))

openxlsx::addWorksheet(wb, "Y_train")
openxlsx::writeData(wb, "Y_train", as.data.frame(Y_train))

openxlsx::addWorksheet(wb, "Y_test")
openxlsx::writeData(wb, "Y_test", as.data.frame(Y_test))

openxlsx::saveWorkbook(wb, file = paste0("data/matrices/matrices_data", TAG, ".xlsx"), overwrite = TRUE)