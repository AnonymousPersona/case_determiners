// BM-only model: full model with the separable space–time GP removed.
// - keeps same `tau` prior but now tau = phylogenetic scale
// - removes all spatiotemporal GP data, parameters, and code
// - keeps identical priors for other components to enable direct comparability

data {
  // -------- Outcomes & predictors
  int<lower=1> N;                      // number of languages
  int<lower=2> K;                      // max number of cases
  array[N] int<lower=1, upper=K> n_cases;

  array[N] int<lower=0, upper=1> def;
  array[N] int<lower=0, upper=1> indef;

  // -------- Phylogeny: M trees (BM VCVs), already scaled (e.g., mean diag = 1)
  int<lower=1> M;                      // number of trees used in the mixture
  array[M] matrix[N, N] K_phy;         // one VCV per tree
  vector[M] log_w;                     // log-weights, e.g., rep_vector(-log(M), M)
}

transformed data {
  // Precompute Cholesky for each tree's K (add tiny jitter for safety)
  array[M] matrix[N, N] L_phy;
  for (m in 1:M) {
    matrix[N, N] Km = K_phy[m] + diag_matrix(rep_vector(1e-8, N));
    L_phy[m] = cholesky_decompose(Km);
  }
}

parameters {
  // -------- Intercepts
  real beta_zero_def;
  real beta_zero_indef;

  // -------- Def -> Indef cross-link
  real beta_def_indef;

  // -------- Ordinal/monotone case effects (per outcome)
  real theta_def;
  simplex[K-1] w_def;
  real theta_indef;
  simplex[K-1] w_indef;

  // -------- Phylogeny (latent effect with mixture-of-Gaussian prior)
  vector[N] f_phy;

  // reparameterization for magnitudes
  real<lower=0> tau;                 // total phylogenetic GP scale (now all to phylogeny)
}

transformed parameters {
  // -------- Monotone transforms for cases
  vector[K] g_def;
  vector[K] g_indef;
  g_def[1] = 0;
  g_indef[1] = 0;
  for (k in 2:K) {
    g_def[k]   = g_def[k-1]   + theta_def   * w_def[k-1];
    g_indef[k] = g_indef[k-1] + theta_indef * w_indef[k-1];
  }

  // phylogenetic scale
  real<lower=0> sigma_phy = tau;
}

model {
  // -------- Baselines (your Grambank-informed priors)
  beta_zero_def   ~ normal(-0.51, 1.0);
  beta_zero_indef ~ normal(-1.85, 1.0);

  // -------- Cross-link and ordinal scales
  beta_def_indef  ~ normal(0, 1);
  theta_def       ~ normal(0, 1);
  theta_indef     ~ normal(0, 1);
  // w_def, w_indef are uniform-on-simplex by default
  w_def   ~ dirichlet(rep_vector(2.0, K-1));
  w_indef ~ dirichlet(rep_vector(2.0, K-1));

  // -------- Phylogeny hyper/latent
  // total magnitude: modest (keeps same prior as the full model)
  tau ~ normal(0, 0.3) T[0, ];

  // -------- Mixture prior for f_phy (BM under tree uncertainty)
  {
    array[M] real lps;
    for (m in 1:M) {
      lps[m] = log_w[m]
               + multi_normal_cholesky_lpdf(f_phy | rep_vector(0, N),
                                            sigma_phy * L_phy[m]);
    }
    target += log_sum_exp(lps);
  }

  // -------- Likelihoods (only phylogenetic latent)
  for (n in 1:N) {
    real eta_def_n   = beta_zero_def
                       + g_def[n_cases[n]]
                       + f_phy[n];
    def[n] ~ bernoulli_logit(eta_def_n);

    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + f_phy[n];
    indef[n] ~ bernoulli_logit(eta_indef_n);
  }
}

generated quantities {
  vector[N] log_lik_def;
  vector[N] log_lik_indef;
  vector[N] log_lik_total;
  array[N] int y_rep_def;
  array[N] int y_rep_indef;

  // Reuse f_phy from parameters
  for (n in 1:N) {
    real eta_def_n   = beta_zero_def
                       + g_def[n_cases[n]]
                       + f_phy[n];
    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + f_phy[n];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }

  // -----------------------
  // Variance / share summaries (manual sample variance)
  // -----------------------
  real mean_f_phy = 0;
  for (n in 1:N) mean_f_phy += f_phy[n];
  mean_f_phy /= N;

  real ss_f_phy = 0;
  for (n in 1:N) ss_f_phy += square(f_phy[n] - mean_f_phy);

  // sample variance (use N-1 if N>1, else set to 0)
  real var_f_phy = (N > 1) ? ss_f_phy / (N - 1) : 0;

  // avoid division by zero
  real tiny = 1e-12;
  real denom = var_f_phy + tiny;

  real share_phy = var_f_phy / denom;
}
