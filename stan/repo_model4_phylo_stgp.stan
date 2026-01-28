// Bivariate logistic with ordinal case effect,
// phylogenetic uncertainty via a mixture over trees,
// and a separable space–time GP (for direct comparability).

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

  // -------- Phylogeny: M trees (BM VCVs), already scaled (e.g., mean diag = 1)
  int<lower=1> M;                      // number of trees used in the mixture
  array[M] matrix[N, N] K_phy;         // one VCV per tree
  vector[M] log_w;                     // log-weights, e.g., rep_vector(-log(M), M)

  // -------- Space: pairwise great-circle distances (in 1000 km units)
  // (Use a matrix to match the spatiotemporal model you want to compare to.)
  matrix[N, N] Dgeo;

  // -------- Time bounds per language (centuries).
  // For moderns, set t_lo[i] = t_hi[i] (e.g., 0)
  vector[N] t_lo;
  vector[N] t_hi;

  // small jitter for numerical safety in GP covariance
  real<lower=0> jitter;
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
  real<lower=0> tau;                 // total GP scale
  real<lower=0, upper=1> rho;        // share for phylogeny
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
  
  // convert (tau, rho) to component scales
  real<lower=0> sigma_phy = tau * sqrt(rho);
  real<lower=0> sigma_st  = tau * sqrt(1 - rho);

  // -------- Build separable space–time GP draw
  vector[N] f_st;
  {
    // Space distances already provided as matrix
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

  // -------- Space–time GP hyper/latent
  eta_st     ~ std_normal();

  // total magnitude: modest
  tau ~ normal(0, 0.25) T[0, ];      // 95% roughly up to 0.7 on logit scale
  //tau ~ exponential(4.9);
  //tau ~ exponential(6);
  
  // favor phylogeny: mean a/(a+b)
  //rho ~ beta(3, 1.5);                // mean ≈ 0.67; adjust to taste
  //rho ~ beta(5, 2);
  //rho ~ beta(4, 2);
  rho ~ beta(2, 2);
  // (keep your other priors the same; remove separate priors on sigma_phy/sigma_st)
  //sigma_eps ~ normal(0, 0.04) T[0, ];   // tiny nugget; prevents soaking up structure
  //sigma_eps ~ normal(0, 0.03) T[0, ];

  sigma_eps ~ normal(0, 0.05) T[0, ];

  // Length-scales: favor very smooth, near-constant fields
  // Articles spread locally but endure globally
  //ell_space ~ lognormal(log(2), 0.5);   // median ≈ 2 (≈ 2000 km if units=1000 km)
  //ell_time  ~ lognormal(log(8), 0.6);   // median ≈ 8 centuries

  // shorter ranges (moderate)
  //ell_space ~ lognormal(log(0.8), 0.7);   // ~800 km median
  //ell_time  ~ lognormal(log(3.0), 0.7);   // ~3 centuries median

  //ell_space ~ lognormal(log(0.6), 0.5);
  //ell_time  ~ lognormal(log(2.5), 0.5);

  //ell_space ~ exponential(10);
  //ell_time  ~ exponential(0.25);

  ell_space ~ lognormal(log(0.3), 0.5);   // distance units = 1000 km
  ell_time  ~ lognormal(log(4), 0.5);   // time units = centuries

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

  // -------- Likelihoods
  for (n in 1:N) {
    real eta_def_n   = beta_zero_def
                       + g_def[n_cases[n]]
                       + f_phy[n] + f_st[n];
    def[n] ~ bernoulli_logit(eta_def_n);

    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + f_phy[n] + f_st[n];
    indef[n] ~ bernoulli_logit(eta_indef_n);
  }
}

generated quantities {
  vector[N] log_lik_def;
  vector[N] log_lik_indef;
  vector[N] log_lik_total;
  array[N] int y_rep_def;
  array[N] int y_rep_indef;

  // Reuse f_phy, f_st and t from transformed parameters
  for (n in 1:N) {
    real eta_def_n   = beta_zero_def
                       + g_def[n_cases[n]]
                       + f_phy[n] + f_st[n];
    real eta_indef_n = beta_zero_indef
                       + g_indef[n_cases[n]]
                       + beta_def_indef * def[n]
                       + f_phy[n] + f_st[n];

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
  real mean_f_st  = 0;
  for (n in 1:N) {
    mean_f_phy += f_phy[n];
    mean_f_st  += f_st[n];
  }
  mean_f_phy /= N;
  mean_f_st  /= N;

  real ss_f_phy = 0;
  real ss_f_st  = 0;
  for (n in 1:N) {
    ss_f_phy += square(f_phy[n] - mean_f_phy);
    ss_f_st  += square(f_st[n]  - mean_f_st);
  }

  // sample variance (use N-1 if N>1, else set to 0)
  real var_f_phy = (N > 1) ? ss_f_phy / (N - 1) : 0;
  real var_f_st  = (N > 1) ? ss_f_st  / (N - 1) : 0;

  // include nugget in denominator; add tiny eps to avoid division by zero
  real tiny = 1e-12;
  real denom = var_f_phy + var_f_st + square(sigma_eps) + tiny;

  real share_phy = var_f_phy / denom;
  real share_st  = var_f_st  / denom;

  // prior allocation to space-time GP (in full model)
  real prior_share_st = 1.0 - rho;

  // ---- Population-level (fixed-effects only)
  vector[K] p_def_pop;                // P(def=1 | avg language, case=k)
  vector[K] p_indef_pop_def0;         // P(indef=1 | avg language, case=k, def=0)
  vector[K] p_indef_pop_def1;         // P(indef=1 | avg language, case=k, def=1)

  // first differences (adjacent-level jumps) population-level
  vector[K-1] delta_def_pop;
  vector[K-1] delta_indef_pop_def0;
  vector[K-1] delta_indef_pop_def1;

  // indicators whether p>0.5 (population-level)
  array[K] int p_def_pop_gt_half;
  array[K] int p_indef_pop_def0_gt_half;
  array[K] int p_indef_pop_def1_gt_half;

  // crossing index (smallest k with p>0.5), K+1 means never crosses
  int kstar_def_pop;
  int kstar_indef_pop_def0;
  int kstar_indef_pop_def1;

  // ---- Sample-marginal (average over observed languages using f_phy + f_st)
  vector[K] p_def_avg;                // average_i P(def=1 | case=k, language i)
  vector[K] p_indef_avg;              // average_i P(indef=1 | case=k, language i) (uses observed def[i])

  // first differences (adjacent-level jumps) sample-marginal
  vector[K-1] delta_def_avg;
  vector[K-1] delta_indef_avg;

  // indicators whether p>0.5 (sample-marginal)
  array[K] int p_def_avg_gt_half;
  array[K] int p_indef_avg_gt_half;

  // crossing index for averaged curves
  int kstar_def_avg;
  int kstar_indef_avg;

  // ---- optional cumulative change from level 1
  vector[K] cumdef_pop;    // p_k - p_1 (population)
  vector[K] cumdef_avg;    // p_k - p_1 (average)

  // initialize
  {
    // default initialization
    for (i in 1:K) {
      p_def_pop[i] = 0;
      p_indef_pop_def0[i] = 0;
      p_indef_pop_def1[i] = 0;
      p_def_avg[i] = 0;
      p_indef_avg[i] = 0;
      p_def_pop_gt_half[i] = 0;
      p_indef_pop_def0_gt_half[i] = 0;
      p_indef_pop_def1_gt_half[i] = 0;
      p_def_avg_gt_half[i] = 0;
      p_indef_avg_gt_half[i] = 0;
      cumdef_pop[i] = 0;
      cumdef_avg[i] = 0;
    }

    // compute population-level probabilities (fixed effects only)
    for (i in 1:K) {
      real eta_def_pop_i = beta_zero_def + g_def[i];
      p_def_pop[i] = inv_logit(eta_def_pop_i);

      // for indef we provide two conditionals (def = 0 and def = 1)
      real eta_indef_pop_def0 = beta_zero_indef + g_indef[i] + beta_def_indef * 0;
      real eta_indef_pop_def1 = beta_zero_indef + g_indef[i] + beta_def_indef * 1;
      p_indef_pop_def0[i] = inv_logit(eta_indef_pop_def0);
      p_indef_pop_def1[i] = inv_logit(eta_indef_pop_def1);
    }

    // population-level deltas and gt.5 indicators
    for (i in 1:(K-1)) {
      delta_def_pop[i] = p_def_pop[i+1] - p_def_pop[i];
      delta_indef_pop_def0[i] = p_indef_pop_def0[i+1] - p_indef_pop_def0[i];
      delta_indef_pop_def1[i] = p_indef_pop_def1[i+1] - p_indef_pop_def1[i];
    }

    for (i in 1:K) {
      p_def_pop_gt_half[i] = p_def_pop[i] > 0.5 ? 1 : 0;
      p_indef_pop_def0_gt_half[i] = p_indef_pop_def0[i] > 0.5 ? 1 : 0;
      p_indef_pop_def1_gt_half[i] = p_indef_pop_def1[i] > 0.5 ? 1 : 0;
    }

    // kstar population-level: smallest k with p>0.5 (K+1 if none)
    kstar_def_pop = K + 1;
    kstar_indef_pop_def0 = K + 1;
    kstar_indef_pop_def1 = K + 1;
    for (i in 1:K) {
      if (kstar_def_pop == K + 1 && p_def_pop[i] > 0.5) kstar_def_pop = i;
      if (kstar_indef_pop_def0 == K + 1 && p_indef_pop_def0[i] > 0.5) kstar_indef_pop_def0 = i;
      if (kstar_indef_pop_def1 == K + 1 && p_indef_pop_def1[i] > 0.5) kstar_indef_pop_def1 = i;
    }

    // cumulative differences from level 1 (population)
    for (i in 1:K) cumdef_pop[i] = p_def_pop[i] - p_def_pop[1];

    // ----- Sample-marginal: average over observed languages using f_phy + f_st
    // For def: average P(def=1 | case=k, f_phy[i]+f_st[i]) across i
    // For indef: use observed def[i] to keep same conditional structure as likelihood
    {
      vector[K] sum_p_def_i;    // accumulator across languages
      vector[K] sum_p_indef_i;

      // zero accumulators
      for (i in 1:K) {
        sum_p_def_i[i] = 0;
        sum_p_indef_i[i] = 0;
      }

      for (i in 1:N) {
        real lat_i = f_phy[i] + f_st[i]; // latent term for language i
        for (k in 1:K) {
          real eta_def_ik = beta_zero_def + g_def[k] + lat_i;
          real p_def_ik = inv_logit(eta_def_ik);
          sum_p_def_i[k] += p_def_ik;

          // for indef use observed def[i] to mimic the likelihood's conditioning
          real eta_indef_ik = beta_zero_indef + g_indef[k] + beta_def_indef * def[i] + lat_i;
          real p_indef_ik = inv_logit(eta_indef_ik);
          sum_p_indef_i[k] += p_indef_ik;
        }
      }

      // finish averages
      for (i in 1:K) {
        p_def_avg[i] = sum_p_def_i[i] / N;
        p_indef_avg[i] = sum_p_indef_i[i] / N;
      }
    }

    // deltas and indicators for averaged curves
    for (i in 1:(K-1)) {
      delta_def_avg[i] = p_def_avg[i+1] - p_def_avg[i];
      delta_indef_avg[i] = p_indef_avg[i+1] - p_indef_avg[i];
    }

    for (i in 1:K) {
      p_def_avg_gt_half[i] = p_def_avg[i] > 0.5 ? 1 : 0;
      p_indef_avg_gt_half[i] = p_indef_avg[i] > 0.5 ? 1 : 0;
    }

    // crossing index for averaged curves
    kstar_def_avg = K + 1;
    kstar_indef_avg = K + 1;
    for (i in 1:K) {
      if (kstar_def_avg == K + 1 && p_def_avg[i] > 0.5) kstar_def_avg = i;
      if (kstar_indef_avg == K + 1 && p_indef_avg[i] > 0.5) kstar_indef_avg = i;
    }

    // cumulative for averaged curve
    for (i in 1:K) cumdef_avg[i] = p_def_avg[i] - p_def_avg[1];
  } // end local block
}
