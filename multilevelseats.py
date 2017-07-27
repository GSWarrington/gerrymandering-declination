import pystan

seats_model = """
data {
  int<lower=0> Ni;   # Number of level-1 observations (individual competed races) -   
  int<lower=0> Nk;   # Number of level-2 clusters (states - <= 50)  
  int<lower=0> Nl;   # Number of level-2 clusters (years)

  # Cluster IDs
  int<lower=1> state_id[Ni];    # index <= Nk indicating state
  int<lower=1> year[Ni];        # index <= Nl indicating year
  
  # Discrete outcome
  int y[Ni];                    # 1=Dem win; 0=Rep win
  
  # Discrete predictor
  int<lower=0> incD[Ni];           # 1=Dem Incumbent, 0=Not
  int<lower=0> incR[Ni];           # 1=Rep Incumbent, 0=Not
  real<lower=0> demV[Ni];           # democratic fraction of vote
}

parameters {
  # Define parameters to estimate
  # Population intercept (a real number)
  real beta_0;

  # Level-1 errors
  real<lower=0> sigma_e0;

  # Level-2 random effect (states)
  real u_0k[Nk];
  real<lower=0> sigma_u0k;

  # Level-2 random effect (years)
  real v_0l[Nl];
  real<lower=0> sigma_v0l;

  # incumbency predictors
  real delta_D;
  real delta_R;
  # real<lower=0> sigma_D;
  # real<lower=0> sigma_R;

  # vote predictor
  real delta_V;
  # real<lower=0> sigma_V;
}

transformed parameters  {
  # Individual mean
  # real mu[Ni];       # modeled democratic vote for each race

  # Individual mean - includes district + state + year + incumbency
  # for (i in 1:Ni) {
  #   mu[i] = u_0k[district_id[i]] + v_0l[year[i]] + delta_D*IncD[i] + delta_R*IncR[i] + delta_V*DemV[i];
  # }
}

model {
  for(i in 1:Ni) {
    # y[i] ~ bernoulli(inv_logit(beta_0 + u_0k[state_id[i]] + v_0l[year[i]] + delta_D*incD[i] + delta_R*incR[i] + delta_V*demV[i]));
    y[i] ~ bernoulli(inv_logit(beta_0 + delta_V*demV[i]));
  }

  # Likelihood part of Bayesian inference
  # Outcome model N(mu, sigma^2) (use SD rather than Var)
  # for (i in 1:Ni) {
  #   y[i] ~ bernoulli(inv_logit( alpha + a[g[n]] + x[n]*beta));
  #   y[i] ~ normal(mu[i], sigma_e0);
  # }
}
"""
 
def populate_seats_model():
    """ read in data for the multilevel model
    """
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')

    cnt = 0 # how many lines we've seen
    tot = 0
    ans = []
    yrs = []
    states = []
    myids = []
    sq = [[0,0],[0,0]]

    xarr = []
    zarr = []

    demV_Ni = []
    states_Ni = []
    years_Ni = []
    incR_Ni = []
    incD_Ni = []
    winner_Ni = []

    for line in f:
        cnt += 1 
        if cnt == 1:
            hdrs = line.strip().split(',')
            continue
        l = line.strip().split(',')

        #  3 is who won (0=Rep; 1=Dem)
        # 12 dem's share of presidential vote - uses most recent election
        # 2 is incumbency (0=Rep; 1=Dem)
        # let's make incumbency positive for republicans
        # incum = 2*int(l[2]) - 1 # so it's symmetric
        if 1972 <= int(l[0]) <= 2016 and l[12] != '' and l[2] != '' and l[3] != '' and \
           int(l[3]) <= 1:

            demV_Ni.append(float(l[12])/100)
            states_Ni.append(int(l[1][:-2]))
            years_Ni.append(l[0])
            incR = 1
            incD = 1
            if int(l[2]) == 0: # Rep incumbent
                incR = 1

            if int(l[2]) == 1: # Dem incumbent
                incD = 1

            incR_Ni.append(incR)
            incD_Ni.append(incD)
            winner_Ni.append(int(l[3]))

    # get list of all states seen
    setstates = sorted(list(set(states_Ni)))
    # put them into a dictionary so can assign a number to each
    state_lookup = dict(zip(setstates, range(len(setstates))))
    # replace list of states with indices of them
    statenums = map(lambda x: state_lookup[x]+1, states_Ni) # srrs_mn.county.replace(county_lookup).values
    
    # get list of all years seen
    setyears = sorted(list(set(years_Ni)))
    # put them into a dictionary so can assign a number to each
    year_lookup = dict(zip(setyears, range(len(setyears))))
    # replace list of yearss with indices of them
    yearnums = map(lambda x: year_lookup[x]+1, years_Ni) # srrs_mn.county.replace(county_lookup).values

    # print out seats data
    # for i in range(len(demV_Ni)):
    #     print "%s %s %d % .2f %s %s" % \
    #         (yearnums[i],statenums[i],winner_Ni[i],demV_Ni[i],incD_Ni[i]==1,incR_Ni[i]==0)

    seats_data = {'Ni': len(demV_Ni),
              'Nk': len(statenums),
              'Nl': len(yearnums),
              'state_id': statenums,
              'year': yearnums,
              'y': winner_Ni,
              'incD': incD_Ni,
              'incR': incR_Ni,
              'demV': demV_Ni}

    niter = 2000
    nchains = 2
    seats_fit = pystan.stan(model_code=seats_model, data=seats_data, iter=niter, chains=nchains)
    return seats_fit

