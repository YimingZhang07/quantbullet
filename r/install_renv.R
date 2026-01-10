# r/install_renv.R

# Ensure user library exists and is first in .libPaths()
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  # Default Windows user lib if not set
  user_lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", paste0(R.version$major, ".", R.version$minor))
  Sys.setenv(R_LIBS_USER = user_lib)
}
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

message("R version: ", R.version.string)
message("User lib:  ", user_lib)
message("LibPaths:  ", paste(.libPaths(), collapse = " | "))

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org", lib = user_lib)
}

# sanity check
stopifnot(requireNamespace("renv", quietly = TRUE))
message("renv installed: ", as.character(utils::packageVersion("renv")))
