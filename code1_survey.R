#-----------------------------------------------------------------------------
# code1.R 
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Packages
#-----------------------------------------------------------------------------

packages <- unique(c(
  "tidyverse", "lubridate", "janitor", "skimr", "snakecase",
  "stringr", "dplyr", "gridExtra", "scales", "viridis",
  "patchwork", "fastDummies", "readxl", "openxlsx", "tidymodels",
  "Matrix", "caret", "pscl", "xgboost", "Metrics",
  "numDeriv", "optimParallel", "parallel", "furrr", "progressr",
  "tibble", "readr", "forcats", "ggplot2", "tidyr"
))

load_pkg <- function(pkgs) {
  ok <- vapply(pkgs, function(p) {
    suppressPackageStartupMessages(require(p, character.only = TRUE))
  }, logical(1))
  if (any(!ok)) {
    stop("Packages not installed/loaded: ", paste(pkgs[!ok], collapse = ", "))
  }
  invisible(ok)
}
load_pkg(packages)

#-----------------------------------------------------------------------------
# Plotting setup (DEFINIR pal ANTES de bar_plot)
#-----------------------------------------------------------------------------

loc_ok <- base::Sys.setlocale("LC_ALL", "es_ES.UTF-8")
if (identical(loc_ok, "")) {
  warning(
    "Could not set locale 'es_ES.UTF-8'; ",
    "separators and month names will use the default locale."
  )
}

pal <- c(
  azul_1   = "#4E79A7",
  naranja  = "#F28E2B",
  rojo     = "#E15759",
  turquesa = "#76B7B2",
  verde    = "#59A14F",
  amarillo = "#EDC948",
  morado   = "#B07AA1",
  rosa     = "#FF9DA7",
  cafe     = "#9C755F",
  gris     = "#BAB0AC"
)

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

# Horizontal bar plot (robusto y consistente con paleta)
bar_plot <- function(df, var, title, x_lab,
                     fill_pal = pal, text_size = 3) {
  
  # Conteos (excluye NA)
  counts <- df %>%
    dplyr::filter(!is.na({{ var }})) %>%
    dplyr::count({{ var }}, name = "n") %>%
    dplyr::mutate(
      pct = n / sum(n) * 100,
      var_chr = as.character({{ var }})
    ) %>%
    dplyr::arrange(dplyr::desc(n)) %>%
    dplyr::mutate(var_chr = forcats::fct_reorder(var_chr, n))
  
  # Paleta reciclada (estable)
  fill_vals <- rep(unname(fill_pal), length.out = nrow(counts))
  names(fill_vals) <- levels(counts$var_chr)
  
  ggplot2::ggplot(counts, ggplot2::aes(x = var_chr, y = n, fill = var_chr)) +
    ggplot2::geom_col() +
    ggplot2::geom_text(
      ggplot2::aes(label = sprintf("%.1f%%", pct), y = n),
      hjust = -0.1, size = text_size, colour = "black"
    ) +
    ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.15))) +
    ggplot2::scale_fill_manual(values = fill_vals, guide = "none") +
    ggplot2::labs(title = title, x = x_lab, y = "Number of participants") +
    ggplot2::theme_minimal()
}

#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------

mode_safe <- function(x) {
  x <- x[!is.na(x) & x != ""]
  if (length(x) == 0) return(NA_character_)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

reorder_drop_id <- function(df, ref_ids) {
  df %>%
    dplyr::filter(response_id %in% ref_ids) %>%
    dplyr::slice(match(ref_ids, response_id))
}

#-----------------------------------------------------------------------------
# Load Survey Data
#-----------------------------------------------------------------------------

survey <- readr::read_csv(
  "data/survey-data/survey.csv",
  show_col_types = FALSE
) %>%
  janitor::clean_names()

#-----------------------------------------------------------------------------
# Process Race Question (multi-response)
#-----------------------------------------------------------------------------

race_levels <- c(
  "White or Caucasian",
  "Black or African American",
  "Asian",
  "American Indian/Native American or Alaska Native",
  "Native Hawaiian or Other Pacific Islander",
  "Other"
)

race_long <- survey %>%
  dplyr::select(response_id, race_raw = q_demos_race) %>%
  dplyr::filter(!is.na(race_raw) & race_raw != "") %>%
  dplyr::mutate(
    race_raw = race_raw %>%
      stringr::str_replace_all("\\s*/\\s*", "/") %>%
      stringr::str_replace_all("\\s*,\\s*", ", ") %>%
      stringr::str_squish()
  ) %>%
  tidyr::separate_rows(race_raw, sep = ",\\s*") %>%
  dplyr::mutate(race_raw = stringr::str_trim(race_raw))

race_dummies <- race_long %>%
  dplyr::mutate(
    race  = factor(race_raw, levels = race_levels),
    value = 1L
  ) %>%
  dplyr::select(-race_raw) %>%
  tidyr::pivot_wider(
    names_from   = race,
    values_from  = value,
    names_prefix = "race_",
    values_fill  = 0L
  ) %>%
  janitor::clean_names()

needed_race <- paste0("race_", janitor::make_clean_names(race_levels))
race_dummies[setdiff(needed_race, names(race_dummies))] <- 0L



#-----------------------------------------------------------------------------
# Process Life Changes Question (multi-response)
#-----------------------------------------------------------------------------

life_levels <- c(
  "Lost a job",
  "Moved place of residence",
  "Divorce",
  "Had a child",
  "Became pregnant"
)

life_long <- survey %>%
  dplyr::select(response_id, life_raw = q_life_changes) %>%
  dplyr::filter(!is.na(life_raw) & life_raw != "") %>%
  dplyr::mutate(
    life_raw = life_raw %>%
      stringr::str_replace_all("\\s*/\\s*", "/") %>%
      stringr::str_replace_all("\\s*,\\s*", ", ") %>%
      stringr::str_squish()
  ) %>%
  tidyr::separate_rows(life_raw, sep = ",\\s*") %>%
  dplyr::mutate(life_raw = stringr::str_trim(life_raw))

life_dummies <- life_long %>%
  dplyr::mutate(
    life  = factor(life_raw, levels = life_levels),
    value = 1L
  ) %>%
  dplyr::select(-life_raw) %>%
  tidyr::pivot_wider(
    names_from   = life,
    values_from  = value,
    names_prefix = "life_",
    values_fill  = 0L
  ) %>%
  janitor::clean_names()

needed_life <- paste0("life_", janitor::make_clean_names(life_levels))
life_dummies[setdiff(needed_life, names(life_dummies))] <- 0L


#-----------------------------------------------------------------------------
# Merge Enriched Variables into Survey
#-----------------------------------------------------------------------------

survey       <- survey       %>% dplyr::distinct(response_id, .keep_all = TRUE)
race_dummies <- race_dummies %>% dplyr::distinct(response_id, .keep_all = TRUE)
life_dummies <- life_dummies %>% dplyr::distinct(response_id, .keep_all = TRUE)

survey_enriched <- survey %>%
  dplyr::left_join(race_dummies, by = "response_id", relationship = "one-to-one") %>%
  dplyr::left_join(life_dummies, by = "response_id", relationship = "one-to-one")

colnames(survey_enriched)

#-----------------------------------------------------------------------------
# Bar Plots (ejecución individual)
#-----------------------------------------------------------------------------

bar_plot(survey, q_prolific_mturk, "Participation in Amazon MTurk", "Response")
bar_plot(survey, q_demos_age, "Age group distribution", "Age group")
bar_plot(survey, q_demos_hispanic, "Hispanic/Latino origin", "Response",
         fill_pal = pal[c("azul_1", "naranja")])
bar_plot(survey, q_demos_race, "Race combinations (declared)", "Combination",
         text_size = 2.8)
bar_plot(race_long, race_raw, "Participants by race (individual count)", "Race")
bar_plot(survey, q_demos_education, "Education level", "Education", text_size = 2.8)
bar_plot(survey, q_demos_gender, "Declared gender identity", "Gender")
bar_plot(survey, q_sexual_orientation, "Declared sexual orientation", "Orientation")
bar_plot(survey, q_demos_state, "State distribution", "State", text_size = 2.4)
bar_plot(survey, q_amazon_use_howmany, "People sharing Amazon account", "Number of people")
bar_plot(survey, q_amazon_use_hh_size, "Household size", "People in household")
bar_plot(survey, q_amazon_use_how_oft, "Amazon order frequency", "Frequency")
bar_plot(survey, q_substance_use_1, "Household cigarette use", "Response")
bar_plot(survey, q_substance_use_2, "Household marijuana use", "Response")
bar_plot(survey, q_substance_use_3, "Household alcohol use", "Response")
bar_plot(survey, q_personal_1, "Diabetes in household", "Response")
bar_plot(survey, q_personal_2, "Wheelchair use in household", "Response")
bar_plot(survey, q_life_changes, "Life changes in 2021 (combinations)", "Combination",
         text_size = 2.8)
bar_plot(life_long, life_raw, "Life changes in 2021 (individual)", "Life change")
bar_plot(survey, q_demos_income, "Income range", "Response")

#------------------------------------------------------------------------------
# Collect all survey plots
#------------------------------------------------------------------------------

plots_survey <- list(
  bar_plot(survey, q_prolific_mturk, "Participation in Amazon MTurk", "Response"),
  bar_plot(survey, q_demos_age, "Age group distribution", "Age group"),
  bar_plot(survey, q_demos_hispanic, "Hispanic/Latino origin", "Response",
           fill_pal = pal[c("azul_1", "naranja")]),
  bar_plot(survey, q_demos_race, "Race combinations (declared)", "Combination",
           text_size = 2.8),
  bar_plot(race_long, race_raw, "Participants by race (individual count)", "Race"),
  bar_plot(survey, q_demos_education, "Education level", "Education", text_size = 2.8),
  bar_plot(survey, q_demos_gender, "Declared gender identity", "Gender"),
  bar_plot(survey, q_sexual_orientation, "Declared sexual orientation", "Orientation"),
  bar_plot(survey, q_demos_state, "State distribution", "State", text_size = 2.4),
  bar_plot(survey, q_amazon_use_howmany, "People sharing Amazon account", "Number of people"),
  bar_plot(survey, q_amazon_use_hh_size, "Household size", "People in household"),
  bar_plot(survey, q_amazon_use_how_oft, "Amazon order frequency", "Frequency"),
  bar_plot(survey, q_substance_use_1, "Household cigarette use", "Response"),
  bar_plot(survey, q_substance_use_2, "Household marijuana use", "Response"),
  bar_plot(survey, q_substance_use_3, "Household alcohol use", "Response"),
  bar_plot(survey, q_personal_1, "Diabetes in household", "Response"),
  bar_plot(survey, q_personal_2, "Wheelchair use in household", "Response"),
  bar_plot(survey, q_life_changes, "Life changes in 2021 (combinations)", "Combination",
           text_size = 2.8),
  bar_plot(life_long, life_raw, "Life changes in 2021 (individual)", "Life change"),
  bar_plot(survey, q_demos_income, "Income range", "Response")
)

#------------------------------------------------------------------------------
# Save all plots into a single PDF
#------------------------------------------------------------------------------

out_pdf <- "survey_barplots.pdf"
grDevices::pdf(out_pdf, width = 10, height = 7)
for (p in plots_survey) print(p)
grDevices::dev.off()
message("Saved: ", out_pdf)

