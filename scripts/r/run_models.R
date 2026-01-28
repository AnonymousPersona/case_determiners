# scripts/r/run_models.R

library(here)
library(tidyverse)
library(ape)
library(XML)
library(xml2)
library(cmdstanr)

# parallel options (kept for non-Stan parallel usage)
options(mc.cores = parallel::detectCores())

message("Using CmdStan: ", cmdstanr::cmdstan_version())
message("here() -> ", here())

# ensure output directory exists
if (!dir.exists(here("output"))) dir.create(here("output"), recursive = TRUE)

# create CSV output directory for CmdStan 
stan_csv_dir <- here("output", "stan-csv")
if (!dir.exists(stan_csv_dir)) dir.create(stan_csv_dir, recursive = TRUE)

# ---- Load data ----
df_path <- here("data", "repo_data.csv")
if (!file.exists(df_path)) stop("Data file missing: ", df_path)
df <- read.csv(df_path, stringsAsFactors = FALSE)
message("Loaded df: ", df_path)

# geographic distance matrix
d_geo_path <- here("data", "geo_distances.csv")
if (!file.exists(d_geo_path)) stop("Geo distance file missing: ", d_geo_path)
d_geo <- as.matrix(read.csv(d_geo_path, header = TRUE, stringsAsFactors = FALSE))
message("Loaded geo distances: ", d_geo_path)

# ---- Load phylogenetic trees ----
ieo50_path <- here("data", "ieo_fifty.nex")
if (!file.exists(ieo50_path)) stop("Tree file missing: ", ieo50_path)
ieo50 <- read.nexus(ieo50_path)
message("Loaded phylogenetic trees: ", ieo50_path)

# Rename tip labels (defensive: requires N and ie_data$lang_formatted or df$lang_formatted)
if (!exists("N")) {
  if ("N" %in% names(df)) {
    N <- df$N[1]
  } else {
    N <- nrow(df)
    message("N not found in environment; defaulting to nrow(df) = ", N)
  }
}

ieo50 <- lapply(ieo50, function(tr) {
  if (length(tr$tip.label) != N) {
    warning("Tree has ", length(tr$tip.label), " tips, expected ", N)
  }
  if (exists("ie_data") && is.list(ie_data) && "lang_formatted" %in% names(ie_data)) {
    tr$tip.label <- ie_data$lang_formatted
  } else if ("lang_formatted" %in% names(df)) {
    tr$tip.label <- df$lang_formatted
  } else {
    warning("No lang_formatted vector found (ie_data or df). Tip labels left unchanged for this tree.")
  }
  tr
})

# ---- Build VCV matrices (symmetrize + scale) ----
Ks <- lapply(ieo50, function(tr) {
  K <- vcv(tr)
  K <- 0.5 * (K + t(K))
  K <- K / mean(diag(K))
  eig_min <- min(eigen(K, symmetric = TRUE, only.values = TRUE)$values)
  if (eig_min < 1e-10) K <- K + diag(1e-8, nrow(K))
  K
})

M <- length(Ks)
if (M < 1) stop("No phylogenetic matrices produced (M < 1).")
K_phy_arr <- array(NA_real_, dim = c(M, N, N))
for (m in seq_len(M)) K_phy_arr[m, , ] <- Ks[[m]]
message("Constructed K_phy array with dims: ", paste(dim(K_phy_arr), collapse = " x "))

# ---- Utility: compile Stan model via cmdstanr ----
compile_model <- function(stan_path) {
  if (!file.exists(stan_path)) stop("Stan file missing: ", stan_path)
  message("Compiling: ", stan_path)
  mod <- cmdstan_model(stan_path)
  message("Compiled: ", stan_path)
  mod
}

# ---- Model 1 ----
model1_data <- list(
  N       = nrow(df),
  K       = max(df$n_case, na.rm = TRUE),
  n_cases = as.integer(df$n_case),
  def     = as.numeric(df$def),
  indef   = as.numeric(df$def) # original used def twice; keep as-is but verify
)

sm1_path <- here("stan", "repo_model1_no_gp_no_phylo.stan")
sm1_mod <- compile_model(sm1_path)

# CmdStan sample: total iterations 2000, warmup 500 -> iter_sampling = 1500
fit1 <- sm1_mod$sample(
  data = model1_data,
  seed = 1234,
  chains = 4,
  parallel_chains = min(4, parallel::detectCores()),
  iter_warmup = 500,
  iter_sampling = 1500,
  thin = 2,
  refresh = 250,
  output_dir = stan_csv_dir
)
saveRDS(fit1, file = here("output", "fit1.rds"))
message("Model 1 complete; saved fit to output/fit1.rds")

# ---- Model 2 ----
model2_data <- list(
  N       = nrow(df),
  K       = max(df$n_case, na.rm = TRUE),
  M       = M,
  n_cases = as.integer(df$n_case),
  def     = as.numeric(df$def),
  indef   = as.numeric(df$def),
  K_phy   = K_phy_arr,
  log_w   = rep(-log(M), M)
)

sm2_path <- here("stan", "repo_model2_phylo.stan")
sm2_mod <- compile_model(sm2_path)

fit2 <- sm2_mod$sample(
  data = model2_data,
  seed = 1234,
  chains = 4,
  parallel_chains = min(4, parallel::detectCores()),
  iter_warmup = 500,
  iter_sampling = 1500,
  thin = 2,
  refresh = 250,
  output_dir = stan_csv_dir
)
saveRDS(fit2, file = here("output", "fit2.rds"))
message("Model 2 complete; saved fit to output/fit2.rds")

# ---- Model 3 ----
model3_data <- list(
  N       = nrow(df),
  K       = max(df$n_case, na.rm = TRUE),
  M       = M,
  n_cases = as.integer(df$n_case),
  def     = as.integer(df$def_binary_narrow),
  indef   = as.integer(df$indef_binary_narrow),
  Dgeo    = d_geo,
  log_w   = rep(-log(M), M),
  t_lo    = df$t_lo,
  t_hi    = df$t_hi,
  jitter  = 1e-8
)

sm3_path <- here("stan", "repo_model3_stgp.stan")
sm3_mod <- compile_model(sm3_path)

fit3 <- sm3_mod$sample(
  data = model3_data,
  seed = 1234,
  chains = 4,
  parallel_chains = min(4, parallel::detectCores()),
  iter_warmup = 500,
  iter_sampling = 1500,
  thin = 2,
  refresh = 250,
  output_dir = stan_csv_dir
)
saveRDS(fit3, file = here("output", "fit3.rds"))
message("Model 3 complete; saved fit to output/fit3.rds")

# ---- Model 4 ----
model4_data <- list(
  N       = nrow(df),
  K       = max(df$n_case, na.rm = TRUE),
  M       = M,
  n_cases = as.integer(df$n_case),
  def     = as.integer(df$def),
  indef   = as.integer(df$indef),
  K_phy   = K_phy_arr,
  Dgeo    = d_geo,
  log_w   = rep(-log(M), M),
  t_lo    = df$t_lo,
  t_hi    = df$t_hi,
  jitter  = 1e-8
)

sm4_path <- here("stan", "repo_model4_phylo_stgp.stan")
sm4_mod <- compile_model(sm4_path)

fit4 <- sm4_mod$sample(
  data = model4_data,
  seed = 1234,
  chains = 4,
  parallel_chains = min(4, parallel::detectCores()),
  iter_warmup = 500,
  iter_sampling = 1500,
  thin = 2,
  refresh = 250,
  output_dir = stan_csv_dir
)
saveRDS(fit4, file = here("output", "fit4.rds"))
message("Model 4 complete; saved fit to output/fit4.rds")

message("All models finished.")