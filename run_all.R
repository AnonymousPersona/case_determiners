# run_all.R

if (!requireNamespace("here", quietly = TRUE)) {
  stop("Package 'here' is required. Install it with install.packages('here')")
}
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  stop("Package 'cmdstanr' is required. Install it with install.packages('cmdstanr')")
}

# declare project root
here::i_am("run_all.R")

# check CmdStan installation
if (!isTRUE(
  tryCatch(!is.null(cmdstanr::cmdstan_version()), error = function(e) FALSE)
)) {
  stop(
    "CmdStan is not installed.\n",
    "Install it with:\n",
    "  cmdstanr::install_cmdstan(cores = parallel::detectCores())"
  )
}

# run analysis
source(file.path("scripts", "r", "run_models.R"))