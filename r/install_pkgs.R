# Rscript .\r\install_pkgs.R
renv::init(bare = TRUE)              # creates renv infrastructure
renv::install("mgcv")                # installs mgcv into renv library (or cache)
renv::install("dplyr")               # installs dplyr into renv library (or cache)
renv::install("arrow")
renv::install("stringr")
renv::install("data.table")
renv::install("ggplot2")
renv::install("gplots")
renv::install("parallel")
renv::install("snow")
renv::install("doSNOW")
renv::install("foreach")
renv::install("itertools")
renv::install("glue")
renv::install("pdftools")
renv::install("tidyverse")
renv::snapshot()                     # writes renv.lock
