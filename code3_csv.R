#-----------------------------------------------------------------------------
# code3.R 
#-----------------------------------------------------------------------------

# =============================================================================
# Excel to CSV conversion
# =============================================================================
# This script converts each sheet of an Excel file containing matrices
# into a separate CSV file.
# =============================================================================

library(readxl)

# ----------------------------------------------------------------------------- 
# Configuration
# -----------------------------------------------------------------------------

# Path to the Excel file with the matrices
excel_file <- "data/matrices/matrices_data.xlsx"

# Output directory for the CSV files
output_dir <- "data"

# Create output directory if it does not exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("✓ Output directory created:", output_dir, "\n")
}

cat("=============================================================================\n")
cat("EXCEL-TO-CSV CONVERSION\n")
cat("=============================================================================\n\n")

# ----------------------------------------------------------------------------- 
# List sheets in the Excel file
# -----------------------------------------------------------------------------

sheets <- excel_sheets(excel_file)
cat("Sheets found in the Excel file:\n")
for (i in seq_along(sheets)) {
  cat("  ", i, ".", sheets[i], "\n")
}
cat("\n")

# ----------------------------------------------------------------------------- 
# Convert each sheet to CSV
# -----------------------------------------------------------------------------

cat("Converting sheets to CSV...\n")
for (sheet in sheets) {
  
  # Read current sheet
  data <- read_excel(excel_file, sheet = sheet)
  
  # Build CSV file name and full path
  csv_filename <- paste0(sheet, ".csv")
  csv_path     <- file.path(output_dir, csv_filename)
  
  # Save as CSV (no row names)
  write.csv(data, csv_path, row.names = FALSE)
  
  cat("✓", sheet, "->", csv_filename,
      "(", nrow(data), "rows x", ncol(data), "columns)\n")
}

cat("\n=============================================================================\n")
cat("✓ CONVERSION COMPLETED SUCCESSFULLY!\n")
cat("=============================================================================\n")
cat("CSV files saved in:", output_dir, "\n")
cat("Total number of files:", length(sheets), "\n")
cat("=============================================================================\n")



