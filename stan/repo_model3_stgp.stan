// Space-time GP only

functions {
  // Matérn 3/2 covariance given a matrix of pairwise distances (already scaled)
  matrix matern32_cov(matrix D, real ell) {
    int N = rows(D);
    matrix[N, N] K;
    real c = sqrt(3.0) / ell;
    for (i in 1:N) {
      K[i, i] = 1.0;
      for (j in (i + 1):N) {
        real r = D[i, j];
        real a = 1.0 + c * r;
        real v = a * exp(-c * r);
        K[i, j] = v;
        K[j, i] = v;
      }
    }
    return K;
  }
}

data {
  // -------- Outcomes & predictors
  int<lower=1> N;                      // number of languages
  int<lower=2> K;                      // max number of cases
  array[N] int<lower=1, upper=K> n_cases;

  array[N] int<lower=0, upper=1> def;
  array[N] int<lower=0, upper=1> indef;

  // -------- Space: pairwise great-circle distances (in 1000 km units)
  matrix[N, N] Dgeo;

  // -------- Time bounds per language (centuries)
  // For moderns, set t_lo[i] = t_hi[i] (e.g., 0)
  vector[N] t_lo;
  vector[N] t_hi;

  // small jitter for numerical safety in GP covariance
  real<lower=0> jitter;
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

  // -------- Space–time GP hyper/latent (separable Matérn 3/2 in space and time)
  vector[N] eta_st;                    // std-normal latent
  real<lower=0> sigma_eps;             // nugget (residual GP jitter)

  // length-scales (in scaled units)
  real<lower=0> ell_space;             // in 1000 km units
  real<lower=0> ell_time;              // in centuries

  // -------- Latent times (for date-uncertain languages)
  // Map an unconstrained (0,1) to [t_lo, t_hi]
  vector<lower=0, upper=1>[N] t_u;

  // reparameterization for magnitudes
  real<lower=0> tau; // total GP scale (all assigned to space-time here)
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

  // -------- Realized times (centuries)
  vector[N] t;
  for (i in 1:N) {
    t[i] = t_lo[i] + t_u[i] * (t_hi[i] - t_lo[i]);
  }

  // convert tau to component scale: here all latent variance goes to space-time GP
  real<lower=0> sigma_st = tau;

  // -------- Build separable space–time GP draw
  vector[N] f_st;
  {
    // Space distances provided as matrix
    matrix[N, N] Dspace = Dgeo;

    // Time distance matrix
    matrix[N, N] Dtime;
    for (i in 1:N) {
      Dtime[i, i] = 0;
      for (j in (i + 1):N) {
        real d = abs(t[i] - t[j]);
        Dtime[i, j] = d;
        Dtime[j, i] = d;
      }
    }

    // Separable kernel: K = K_space ∘ K_time (Hadamard/elementwise product)
    matrix[N, N] Ks = matern32_cov(Dspace,  ell_space);
    matrix[N, N] Kt = matern32_cov(Dtime,   ell_time);
    matrix[N, N] K_sep = elt_multiply(Ks, Kt);

    matrix[N, N] SIGMA =
      square(sigma_st) * K_sep
      + diag_matrix(rep_vector(square(sigma_eps), N))
      + diag_matrix(rep_vector(jitter, N));

    matrix[N, N] L = cholesky_decompose(SIGMA);
    f_st = L * eta_st;   // eta_st ~ N(0, I)
  }
}

model {
  // -------- Baselines
  beta_zero_def   ~ normal(-0.51, 1.0);
  beta_zero_indef ~ normal(-1.85, 1.0);

  // -------- Cross-link and ordinal scales
  beta_def_indef  ~ normal(0, 1);
  theta_def       ~ normal(0, 1);
  theta_indef     ~ normal(0, 1);

  // w_def, w_indef are uniform-on-simplex by default
  w_def   ~ dirichlet(rep_vector(2.0, K-1));
  w_indef ~ dirichlet(rep_vector(2.0, K-1));

  // -------- Space–time GP hyper/latent
  eta_st     ~ std_normal();

  // total magnitude
  tau ~ normal(0, 0.25) T[0, ];

  // nugget
  sigma_eps ~ normal(0, 0.05) T[0, ];

  // Length-scales
  ell_space ~ lognormal(log(0.3), 0.5);   // distance units = 1000 km
  ell_time  ~ lognormal(log(4), 0.5);   // time units = centuries

  // -------- Likelihoods (no phylogenetic f_phy term)
  for (n in 1:N) {
    real eta_def_n   = beta_zero_def
                       + g_def[n_cases[n]]
                       + f_st[n];
    def[n] ~ bernoulli_logit(eta_def_n);

    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + f_st[n];
    indef[n] ~ bernoulli_logit(eta_indef_n);
  }
}

generated quantities {
  vector[N] log_lik_def;
  vector[N] log_lik_indef;
  vector[N] log_lik_total;
  array[N] int y_rep_def;
  array[N] int y_rep_indef;

  // Reuse f_st and t from transformed parameters
  for (n in 1:N) {
    real eta_def_n   = beta_zero_def
                       + g_def[n_cases[n]]
                       + f_st[n];
    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + f_st[n];

    log_lik_def[n]   = bernoulli_logit_lpmf(def[n]   | eta_def_n);
    log_lik_indef[n] = bernoulli_logit_lpmf(indef[n] | eta_indef_n);
    log_lik_total[n] = log_lik_def[n] + log_lik_indef[n];

    y_rep_def[n]   = bernoulli_logit_rng(eta_def_n);
    y_rep_indef[n] = bernoulli_logit_rng(eta_indef_n);
  }

  // -----------------------
  // Variance / share summaries
  // -----------------------
  real mean_f_phy = 0;
  real mean_f_st  = 0;
  for (n in 1:N) {
    mean_f_st  += f_st[n];
  }
  mean_f_st  /= N;

  real ss_f_phy = 0;
  real ss_f_st  = 0;
  for (n in 1:N) {
    // phylogenetic contribution absent in this BM-free model
    ss_f_phy += 0;
    ss_f_st  += square(f_st[n]  - mean_f_st);
  }

  // sample variance (use N-1 if N>1, else set to 0)
  real var_f_phy = 0;
  real var_f_st  = (N > 1) ? ss_f_st  / (N - 1) : 0;

  // include nugget in denominator; add tiny eps to avoid division by zero
  real tiny = 1e-12;
  real denom = var_f_phy + var_f_st + square(sigma_eps) + tiny;

  real share_phy = 0;
  real share_st  = var_f_st  / denom;

  // prior allocation to space-time GP (in BM-free model all goes to ST)
  real prior_share_st = 1.0;
}
