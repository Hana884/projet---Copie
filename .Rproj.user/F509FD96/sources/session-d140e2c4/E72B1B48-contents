Sys.setenv(RSTUDIO_PANDOC = "C:/Program Files/RStudio/resources/app/bin/quarto/bin/tools")
tryCatch({
  rmarkdown::render("socle_classique_R.Rmd", output_file = "socle_classique_R.html")
  cat("SUCCESS: socle_classique_R.html rendered successfully!\n")
}, error = function(e) {
  cat("ERROR:\n", conditionMessage(e), "\n")
})
