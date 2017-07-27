
leveliii_model = """
data {
  int<lower=0> Ni;   # Number of level-1 observations (individual competed races) -   
  int<lower=0> Nj;   # Number of level-2 clusters (districts - unique identifiers)  
  int<lower=0> Nk;   # Number of level-3 clusters (states - <= 50)  
  int<lower=0> Nl;   # Number of level-2 clusters (years)

  # Cluster IDs
  int<lower=1> district_id[Ni]; # index <= Nj indicating district
  int<lower=1> state_id[Ni];    # index <= Nk indicating state
  int<lower=1> year[Ni];        # index <= Nl indicating year
  
  # Level 3 look up vector for level 2
  int<lower=1> state_Lookup[Nj];  # lookup table so we can determine state index from district index

  # Continuous outcome
  real y[Ni];                     # percentage voting democratic in each race
  
  # Discrete predictor
                                  # need to update input data so that there are two parameters here
  int<lower=0> IncD[Ni];           # 1=Dem Incumbent, 0=Not
  int<lower=0> IncR[Ni];           # 1=Rep Incumbent, 0=Not
  int<lower=0> winD[Ni];           # 1=Dem winner, 0=Not
  int<lower=0> winR[Ni];           # 1=Rep winner, 0=Not
}

parameters {
  # Define parameters to estimate
  # Population intercept (a real number)
  # real beta_0;

  # Level-1 errors
  real<lower=0> sigma_e0;

  # Level-2 random effect (districts)
  real u_0jk[Nj];
  real<lower=0> sigma_u0jk;

  # Level-2 random effect (years)
  real v_0l[Nl];
  real<lower=0> sigma_v0l;

  # Level-3 random effect (states)
  real u_0k[Nk];
  real<lower=0> sigma_u0k;

  # incumbency predictors
  real delta_D;
  real delta_R;
  real win_D;
  real win_R;
}

transformed parameters  {
  # Varying intercepts
  real beta_0jk[Nj]; # expected dem vote by district
  real beta_0k[Nk];  # expected dem vote by state
  real beta_0l[Nl];  # adjustment to expected dem vote by year

  # Individual mean
  real mu[Ni];       # modeled democratic vote for each race

  # Level-3 (50 level-3 random intercepts - one for each state)
  for (k in 1:Nk) {
     # beta_0k[k] = beta_0 + u_0k[k];
     beta_0k[k] = 0.5 + u_0k[k];
  }
  # Level-2 (many level-2 random intercepts - one for each district - state info has been incorporated)
  for (j in 1:Nj) {
    beta_0jk[j] = beta_0k[state_Lookup[j]] + u_0jk[j];
  }
  
  # Individual mean - includes district + state + year + incumbency
  for (i in 1:Ni) {
    mu[i] = beta_0jk[district_id[i]] + v_0l[year[i]] + delta_D*IncD[i] + delta_R*IncR[i] + win_D*winD[i] + win_R*winR[i];
  }
}

model {
  # Prior part of Bayesian inference
  # Flat prior for mu (no need to specify if non-informative)

  # Random effects distribution
  u_0k  ~ normal(0, sigma_u0k); # expect state variation from global average to be normal around 0
  u_0jk ~ normal(0, sigma_u0jk); # expect district variation from given state average (or global?) to be normal around 0
  v_0l ~ normal(0, sigma_v0l); # expect year variation from global average to be normal around 0

  # Likelihood part of Bayesian inference
  # Outcome model N(mu, sigma^2) (use SD rather than Var)
  for (i in 1:Ni) {
    y[i] ~ normal(mu[i], sigma_e0);
  }
#     generated quantities {
#     vector[N_tilde] y_tilde;
#     for (n in 1:N_tilde)
#        y_tilde[n] <- normal_rng(x_tilde[n] * beta, sigma);
#    }
#     beta_0 + u_0k[k] + u_0jk[j] + v_0l[year[i]]
}
"""
 
# print sorted(distnums)[-10:]
# print len(yearnums),len(pvote)
# traceplot(leveliii_fit, pars = c("beta_0","sigma_e0","sigma_u0jk","sigma_u0k","sigma_v0l"), inc_warmup = TRUE)
# print leveliii_fit
# leveliii_fit.plot(['u_0k[2]','u_0k[5]','u_0k[9]','u_0k[10]'])
# leveliii_fit.plot(['u_0k'])
# plt.savefig('/home/gswarrin/u0k.png')
