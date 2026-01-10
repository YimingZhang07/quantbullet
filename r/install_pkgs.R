# Rscript .\r\install_pkgs.R
renv::init(bare = TRUE)              # creates renv infrastructure
renv::install("mgcv")                # installs mgcv into renv library (or cache)
renv::snapshot()                     # writes renv.lock
