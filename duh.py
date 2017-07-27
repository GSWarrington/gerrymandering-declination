import pystan

V_model = """
data {
  int<lower=0> Ni;   # Number of level-1 observations (individual competed races) -   

  # Discrete outcome
  int y[Ni];                    # 1=Dem win; 0=Rep win
  
  real<lower=0> demV[Ni];           # democratic fraction of vote
}

parameters {
  # Population intercept (a real number)
  real beta_0;

  real delta_V;
}

transformed parameters  {}

model {
  for(i in 1:Ni) {
    y[i] ~ bernoulli(inv_logit(beta_0 + delta_V*demV[i]));
  }
}
"""

################################################################################## 

DRV_model = """
data {
  int<lower=0> Ni;   # Number of level-1 observations (individual competed races) -   

  # Discrete outcome
  int y[Ni];                    # 1=Dem win; 0=Rep win
  
  # int<lower=1> state_id[Ni];    # index <= Nk indicating state
  # int<lower=1> year[Ni];        # index <= Nl indicating year

  # Discrete predictor
  # int<lower=0> incDR[Ni];           # 1=Dem Incumbent, -1=Not
  int<lower=0> incD[Ni];           # 1=Dem Incumbent, 0=Not
  int<lower=0> incR[Ni];           # 1=Rep Incumbent, 0=Not
  real<lower=0> demV[Ni];           # democratic fraction of vote
}

parameters {
  # Population intercept (a real number)
  real beta_0;

  # Level-1 errors
  # real<lower=0> sigma_e0;

  # Level-2 random effect (years)
  # real u_0k[Nk];
  # real v_0l[Nl];
  # real<lower=0> sigma_v0l;

  # incumbency predictors
  real delta_D;
  real delta_R;
  # real delta_DR;

  # vote predictor
  real delta_V;
  # real<lower=0> sigma_V;
}

transformed parameters  {}

model {
  # beta_0 ~ normal(0,sigma_e0);
  # delta_V ~ normal(0,sigma_V);
  for(i in 1:Ni) {
    y[i] ~ bernoulli(inv_logit(beta_0 + delta_V*demV[i] + delta_D*incD[i] + delta_R*incR[i]));
  }
}
"""
 
######################################################################################
# Include a year adjustment as well; remove beta_0 adjustment

YearDRV_model = """
data {
  int<lower=0> Ni;   # Number of level-1 observations (individual competed races) -   
  int<lower=0> Nl;   # Number of level-2 clusters (years)

  # Discrete outcome
  int y[Ni];                    # 1=Dem win; 0=Rep win
  
  # int<lower=1> state_id[Ni];    # index <= Nk indicating state
  int<lower=1> year[Ni];        # index <= Nl indicating year

  # Discrete predictor
  # int<lower=0> incDR[Ni];           # 1=Dem Incumbent, -1=Not
  int<lower=0> incD[Ni];           # 1=Dem Incumbent, 0=Not
  int<lower=0> incR[Ni];           # 1=Rep Incumbent, 0=Not
  real<lower=0> demV[Ni];           # democratic fraction of vote
}

parameters {
  # Population intercept (a real number)
  # real beta_0;

  # Level-1 errors
  # real<lower=0> sigma_e0;

  # Level-2 random effect (years)
  # real u_0k[Nk];
  real v_0l[Nl];
  # real<lower=0> sigma_v0l;

  # incumbency predictors
  real delta_D;
  real delta_R;
  # real delta_DR;

  # vote predictor
  real delta_V;
  # real<lower=0> sigma_V;
}

transformed parameters  {}

model {
  # beta_0 ~ normal(0,sigma_e0);
  # delta_V ~ normal(0,sigma_V);
  for(i in 1:Ni) {
    y[i] ~ bernoulli(inv_logit(v_0l[year[i]] + delta_V*demV[i] + delta_D*incD[i] + delta_R*incR[i]));
  }
}
"""

######################################################################################

YearV_model = """
data {
  int<lower=0> Ni;   # Number of level-1 observations (individual competed races) -   
  int<lower=0> Nl;   # Number of level-2 clusters (years)

  # Discrete outcome
  int y[Ni];                    # 1=Dem win; 0=Rep win
  
  # int<lower=1> state_id[Ni];    # index <= Nk indicating state
  int<lower=1> year[Ni];        # index <= Nl indicating year

  # Discrete predictor
  # int<lower=0> incDR[Ni];           # 1=Dem Incumbent, -1=Not
  real<lower=0> demV[Ni];           # democratic fraction of vote
}

parameters {
  # Population intercept (a real number)
  # real beta_0;

  # Level-1 errors
  # real<lower=0> sigma_e0;

  # Level-2 random effect (years)
  # real u_0k[Nk];
  real v_0l[Nl];
  # real<lower=0> sigma_v0l;

  # incumbency predictors
  # real delta_DR;

  # vote predictor
  real delta_V;
  # real<lower=0> sigma_V;
}

transformed parameters  {}

model {
  # beta_0 ~ normal(0,sigma_e0);
  # delta_V ~ normal(0,sigma_V);
  for(i in 1:Ni) {
    y[i] ~ bernoulli(inv_logit(v_0l[year[i]] + delta_V*demV[i]));
  }
}
"""


##############################################################################################

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
    incDR_Ni = []
    winner_Ni = []
    yijit = []

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
        # incum = 2*int(l[2]) - 1 # so it's symmetric # int(l[0]) != 2008 and \
        if 2008 <= int(l[0]) <= 2012 and (int(l[0])%4 == 0) and \
           l[12] != '' and l[2] != '' and l[3] != '' and \
           int(l[3]) <= 1: #  and int(l[2]) > 1:

            demV_Ni.append(float(l[12])/100)

            states_Ni.append(int(l[1][:-2]))
            years_Ni.append(l[0])
            incR = 0
            incD = 0
            if int(l[2]) == 0: # Rep incumbent
                incR = 1

            if int(l[2]) == 1: # Dem incumbent
                incD = 1

            incR_Ni.append(incR)
            incD_Ni.append(incD)
            winner_Ni.append(int(l[3]))
            yijit.append(int(l[3]) + np.random.rand(1,1)[0][0]/10-0.05)        # who won

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
    print "----"
    print year_lookup
    print setyears
    # print yearnums
    print "----"

    # print out seats data
    # for i in range(len(demV_Ni)):
    #     print "%s %s %d % .2f %s %s" % \
    #         (yearnums[i],statenums[i],winner_Ni[i],demV_Ni[i],incD_Ni[i]==1,incR_Ni[i]==0)

    # seats_data = {'Ni': len(demV_Ni),
    #           'Nk': len(statenums),
    #           'Nl': len(yearnums),
    #           'state_id': statenums,
    #           'year': yearnums,
    #           'y': winner_Ni,
    #           'incD': incD_Ni,
    #           'incR': incR_Ni,
    #           'demV': demV_Ni}

    # V model
    if 1 == 1:
        modelstr = V_model
        seats_data = {'Ni': len(demV_Ni),
              'y': winner_Ni,
              'demV': demV_Ni}

    # DRV model
    if 1 == 0:
        modelstr = DRV_model
        seats_data = {'Ni': len(demV_Ni),
              'y': winner_Ni,
              'incD': incD_Ni,
              'incR': incR_Ni,
              'demV': demV_Ni}

    # "Year V model"
    if 1 == 0:
        modelstr = YearV_model
        seats_data = {'Ni': len(demV_Ni),
              'Nl': len(setyears),
              'y': winner_Ni,
              'year': yearnums,
              'demV': demV_Ni}

    # "Year DRV model"
    if 1 == 0:
        modelstr = YearDRV_model
        seats_data = {'Ni': len(demV_Ni),
              'Nl': len(setyears),
              'y': winner_Ni,
              'year': yearnums,
              'incD': incD_Ni,
              'incR': incR_Ni,
              'demV': demV_Ni}

    make_scatter("blah",demV_Ni,yijit)
    print set(yearnums)
    # for i in range(len(demV_Ni)):
    #      print "%s % .2f %s" % \
    #         (winner_Ni[i],demV_Ni[i],yearnums[i])

    niter = 1000
    nchains = 2
    seats_fit = pystan.stan(model_code=modelstr, data=seats_data, iter=niter, chains=nchains)
    return seats_fit

def try_bootstrap(yr,beta0,deltaV,probs,NumBoot=1000):
    """ NumBoot is number of bootstrap samples to try
    """

    est = []
    for i in range(NumBoot):
        #Create a bootstrap sample
        boot = np.random.choice(probs,len(probs),replace=True)        

        #Calculate the sum of p
        est.append(sum([1/(1+np.exp(-(beta0+deltaV*t))) for t in boot]))
    print np.mean(est),np.std(est)
    print np.percentile(est,2.5),np.percentile(est,97.5)
    make_histogram('ci' + str(yr),est)

def inc_bootstrap(yr,beta0,incR,incD,deltaV,probs,inclist,NumBoot=1000):
    """ NumBoot is number of bootstrap samples to try
    Using beta0 and deltaV from fit with incumbency
    """

    incum_est = []
    noincum_est = []
    rr = range(len(probs))
    for i in range(NumBoot):
        #Create a bootstrap sample
        bootidx = np.random.choice(rr,len(rr),replace=True)        
        bootprob = [probs[x] for x in bootidx]
        bootinc = [inclist[x] for x in bootidx]

        #Calculate the sum of p
        noincum_est.append(sum([1/(1+np.exp(-(beta0+deltaV*probs[t]))) for t in bootidx]))

        #Calculate the sum of p
        tmp = 0
        for j in bootidx:
            if inclist[j] == 1:
                tmp += 1/(1+np.exp(-(beta0+incD+deltaV*probs[j])))
            elif inclist[j] == 0:
                tmp += 1/(1+np.exp(-(beta0+incR+deltaV*probs[j])))
            else:
                tmp += 1/(1+np.exp(-(beta0+deltaV*probs[j])))
        incum_est.append(tmp)
    print "Using incumbency data:"
    print np.mean(incum_est),np.std(incum_est)
    print np.percentile(incum_est,2.5),np.percentile(incum_est,97.5)
    make_histogram('incum_ci' + str(yr),incum_est)

    print "Not using incumbency data:"
    print np.mean(noincum_est),np.std(noincum_est)
    print np.percentile(noincum_est,2.5),np.percentile(noincum_est,97.5)
    make_histogram('noincum_ci' + str(yr),noincum_est)

def get_data_points(Rincum,Dincum):
    """ read in democratic vote for a given year
    """
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')

    ans = []
    cnt = 0
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
        if int(l[0]) >= 1972 and l[12] != '' and l[3] != '' and \
           int(l[3]) <= 1:

            if (Rincum and int(l[2]) == 0) or \
               (Dincum and int(l[2]) == 1) or \
               (not Rincum and not Dincum and int(l[2]) > 1):
                ans.append([float(l[12])/100,int(l[3])])
    f.close()
    return ans

def get_probs_yr(yr,inctoo=False):
    """ read in democratic vote for a given year
    """
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')

    ans = []
    cnt = 0
    inc = []
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
        if int(l[0]) == yr and \
           l[12] != '' and l[3] != '' and \
           int(l[3]) <= 1:

            ans.append(float(l[12])/100)
            if inctoo:
                inc.append(int(l[2]))
    if inctoo:
        return ans,inc
    else:
        return ans

def plot_logit_curves(pltcum=False):
    """ so we can compare different fits
    """
    fig = plt.figure(figsize=(8,8))
    xvals = np.linspace(0,1,101)

    beta0 = -3.86
    delta_D = 3.02
    delta_R = -2.58
    delta_V = 7.96

    rinc = get_data_points(True,False)
    dinc = get_data_points(False,True)
    ninc = get_data_points(False,False)
    allinc = rinc + dinc + ninc
    print len(rinc),len(dinc),len(ninc)

    xmin = 0.3
    xmax = 0.7
    xunits = (xmax-xmin)*100+1
    dotty = []
    for i,arr in enumerate([rinc,dinc,ninc,allinc]):
        tmp = []
        for cp in np.linspace(xmin,xmax,xunits):
            toleft = filter(lambda x: cp-0.05 <= x[0] <= cp+0.05, arr)
            numer = filter(lambda x: x[1] == 1, toleft)
            if len(toleft) > 0:
                tmp.append(len(numer)*1.0/len(toleft))
            else:
                tmp.append(0)
        dotty.append(tmp)
    cols = ['red','blue','green','grey']
    for i,duh in enumerate(dotty):
        plt.plot(np.linspace(xmin,xmax,xunits),duh,color=cols[i],linestyle='dashed')

    ryvals = [x[1] + 0.05 + np.random.rand(1,1)[0][0]/10-0.05 for x in rinc]
    dyvals = [x[1] - 0.05 + np.random.rand(1,1)[0][0]/10-0.05 for x in dinc]
    nyvals = [x[1] + np.random.rand(1,1)[0][0]/10-0.05 for x in ninc]

    plt.scatter([x[0] for x in rinc],[x for x in ryvals],color='red',s=2)
    plt.scatter([x[0] for x in dinc],[x for x in dyvals],color='blue',s=2)
    plt.scatter([x[0] for x in ninc],[x for x in nyvals],color='green',s=2)

    rvals = [1/(1+np.exp(-(beta0 + delta_R + delta_V*t))) for t in xvals]
    plt.plot(xvals,rvals,'r-')
    dvals = [1/(1+np.exp(-(beta0 + delta_D + delta_V*t))) for t in xvals]
    plt.plot(xvals,dvals,'b-')
    nvals = [1/(1+np.exp(-(beta0 + delta_V*t))) for t in xvals]
    plt.plot(xvals,nvals,'g-')

    # where does this come from
    beta0 = -10.93
    delta_V = 21.08
    ovals = [1/(1+np.exp(-(beta0 + delta_V*t))) for t in xvals]
    plt.plot(xvals,ovals,color='black')

    beta0 = -5.44
    delta_V = 11.31
    pvals = [1/(1+np.exp(-(beta0 + delta_V*t))) for t in xvals]
    plt.plot(xvals,pvals,color='grey')

    # print yvals[:10]
    # plt.plot(xvals,yvals,'b-')
    # plt.plot(xvals,zvals,'r-')
    plt.axhline(0.5)
    plt.axvline(0.5)
    plt.grid()
    plt.savefig('/home/gswarrin/research/gerrymander/seats-curves')

    plt.close()

def cc_percentages():
    """ read in who won the district versus mccain share of vote
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    nonolist = ['AK','DE','MT','ND','SD','VT','WY','OR']
    plist = []
    for st in statelist[1:]:
        if st not in nonolist:
            f = open('/home/gswarrin/research/gerrymander/data/ChenCottrell/' + st + 'simul.txt','r')
            for line in f:
                l = line.rstrip().split('\t')
                l2 = map(lambda x: float(x), l[1:]) # skip simulation number
                for j in range(len(l2)):
                    plist.append(1-l2[j])
    outside = len(filter(lambda x: x < 0.4 or x > 0.6, plist))
    print outside,len(plist)
    make_histogram('simul_hist',plist)

def compare_cong_and_pres():
    """ compare candidate and presidential votes by district
    """
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')

    ans = []
    cnt = 0
    inc = []
    arr = [[0,0],[0,0]]
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
        if int(l[0]) == 2008 and \
           l[12] != '' and l[3] != '' and l[4] != '' and l[4] != ' ' and \
           int(l[3]) <= 1:
            # print ".%s.%s." % (l[12],l[4])
            ans.append([float(l[12])/100,float(l[4])/100])
            presvote = float(l[12])/100
            congvote = float(l[4])/100
            if presvote < 0.5:
                if congvote < 0.5:
                    arr[0][0] += 1
                else:
                    arr[0][1] += 1
            else:
                if congvote < 0.5:
                    arr[1][0] += 1
                else:
                    arr[1][1] += 1
    print arr[0][0]*1.0/(arr[0][0]+arr[0][1]),arr[1][1]*1.0/(arr[1][0]+arr[1][1])
    print arr

    fig = plt.figure(figsize=(8,8))
    plt.scatter([x[0] for x in ans],[x[1] for x in ans],s=1)
    plt.axhline(0.5)  
    plt.axvline(0.5)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/cong_pres_compare')
    plt.close()
    # make_scatter('cong_pres_compare',)

    plt.figure(figsize=(12,8))

    n, bins, patches = plt.hist([x[0] for x in ans], facecolor='g', alpha=0.75, normed = True)
    n, bins, patches = plt.hist([x[1] for x in ans], facecolor='b', alpha=0.75, normed = True)

    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'second_vote_dist_compare')
    plt.show()
    plt.close()


def compare_vote_hist(elecs):
    """ compare the vote histograms for presidential and candidate
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    nonolist = ['AK','DE','MT','ND','SD','VT','WY','OR']
    plist = []
    clist = []
    for st in statelist[1:]:
        if st not in nonolist:
            f = open('/home/gswarrin/research/gerrymander/data/ChenCottrell/' + st + 'simul.txt','r')
            for line in f:
                l = line.rstrip().split('\t')
                l2 = map(lambda x: float(x), l[1:]) # skip simulation number
                for j in range(len(l2)):
                    plist.append(1-l2[j])
            f.close()

    for elec in elecs.values():
        if elec.chamber == '11' and int(elec.yr) >= 1972:
            for x in elec.demfrac:
                clist.append(x)

    plt.figure(figsize=(12,8))

    n, bins, patches = plt.hist(plist, facecolor='g', alpha=0.75, normed = True)
    n, bins, patches = plt.hist(clist, facecolor='b', alpha=0.75, normed = True)

    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'vote_dist_compare')
    plt.show()
    plt.close()

    outside = len(filter(lambda x: x < 0.4 or x > 0.6, plist))
    print outside,len(plist)
    make_histogram('simul_hist',plist)

# a,b=get_probs_yr(2012,True)
# a,b=get_probs_yr(2008,True)
# print len(a),len(b)
# inc_bootstrap(2008,-3.86,-2.58,3.02,7.96,a,b)

# try_bootstrap(-10.93,21.08,get_probs_yr(2008),10000)
# 2008
# try_bootstrap(2008,-8.99,18.29,get_probs_yr(2008),10000)
# 2012
# try_bootstrap(2012,-9.61,18.29,get_probs_yr(2012),10000)

# plot_logit_curves()

# probs = get_probs_yr(2012)
# print len(filter(lambda x: x < 0.3 or x > 0.7, probs)),len(probs)
# compare_vote_hist
