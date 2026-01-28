// Bivariate logistic with ordinal case effect,
// (NO phylogeny, NO space–time GP) — reduced model for direct comparison.

data {
  // -------- Outcomes & predictors
  int<lower=1> N;                      // number of languages
  int<lower=2> K;                      // max number of cases
  array[N] int<lower=1, upper=K> n_cases;

  array[N] int<lower=0, upper=1> def;
  array[N] int<lower=0, upper=1> indef;

  // NOTE: phylogeny and GP inputs removed (no M, K_phy, log_w, Dgeo, t_lo, t_hi, jitter)
}

transformed data {
  // empty — nothing precomputed here
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

  // (No f_phy, no sigma_phy, no GP latent/prior parameters)
}

transformed parameters {
  // -------- Monotone transforms for cases (same as full model)
  vector[K] g_def;
  vector[K] g_indef;
  g_def[1] = 0;
  g_indef[1] = 0;
  for (k in 2:K) {
    g_def[k]   = g_def[k-1]   + theta_def   * w_def[k-1];
    g_indef[k] = g_indef[k-1] + theta_indef * w_indef[k-1];
  }
}

model {
  // -------- Baselines (same priors as in the full model)
  beta_zero_def   ~ normal(-0.51, 1.0);
  beta_zero_indef ~ normal(-1.85, 1.0);


  // -------- Cross-link
  beta_def_indef  ~ normal(0, 1);

  // -------- Ordinal scales
  theta_def       ~ normal(0, 1);
  theta_indef     ~ normal(0, 1);
  // w_def, w_indef are uniform-on-simplex by default

  // -------- Likelihoods (no phylogeny, no GP, no mediation)
    for (n in 1:N) {
    real eta_def_n   = beta_zero_def
                       + g_def[n_cases[n]];
    def[n] ~ bernoulli_logit(eta_def_n);

    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + g_indef[n_cases[n]];
    indef[n] ~ bernoulli_logit(eta_indef_n);
  }
}


generated quantities {
  vector[N] log_lik_def;
  vector[N] log_lik_indef;
  vector[N] log_lik_total;
  array[N] int y_rep_def;
  array[N] int y_rep_indef;

  for (n in 1:N) {
    real eta_def_n   = beta_zero_def
                       + g_def[n_cases[n]];
    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + g_indef[n_cases[n]];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }
}
