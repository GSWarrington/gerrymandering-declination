for i in range(elections['2014_PA_11'].Ndists):
    print i,elections['2014_PA_11'].compute_district_vote(i)

# look how fgap changes over time
clist = ['bs-','r^-','go-','cs-','m^-','yo-']
chamberlist = [l9,l11]
for st in states:
    plt.figure(figsize=(8,8))
    plt.axis([1950,2016,-0.25,0.25])
    for j in range(len(chamberlist)):
        for x in filter(lambda y: y[0] == st, chamberlist[j]):
            plt.plot(x[1],x[2],clist[j])
            # plt.plot(x[1],x[3],clist[j+3])
    plt.grid(True)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'sta12-' + st)
    plt.show()
    plt.close()    
# plt.plot([[1,2],[2,2],[3,2.5]],[[1,5],[2,4],[3,3.5]])
# for i in range(len(ll)):
#    plt.plot(ll[i][0],map(lambda x: x + i*3.0/len(ll),ll[i][1]))
# plt.plot([1,2,3,4,5,6,7,8],[8,7,6,5,4,3,2,1])
# plt.plot([2.5,3.5,4.5],[7.5,1.5,3.4])

slope, intercept, r_value, p_value, std_err = stats.linregress(aa,bb)
print r_value**2
print slope, intercept
# stats.kstest(egaps,'t',(3,))

slope, intercept, r_value, p_value, std_err = stats.linregress(fgaps,ggaps)
print r_value**2
print slope
stats.kstest(egaps,'t',(3,))

import scipy.odr as odr
def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]
linear = odr.Model(f)
# mydata = odr.Data([x[0] for x in stabe],[x[1] for x in stabe]) # , sx=sx, sy=sy) # wd=1./power(sx,2), we=1./power(sy,2))
mydata = odr.Data(aa,bb)
myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
myoutput = myodr.run()
myoutput.pprint()
print stats.pearsonr([x[0] for x in stabe],[x[1] for x in stabe])
print stats.pearsonr([x[0] for x in stabf],[x[1] for x in stabf])
print stats.pearsonr([x[0] for x in stabg],[x[1] for x in stabg])


mydata = odr.Data([x[0] for x in stabf],[x[1] for x in stabf]) # , sx=sx, sy=sy) # wd=1./power(sx,2), we=1./power(sy,2))
myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
myoutput = myodr.run()
myoutput.pprint()
print myoutput.sum_square

mydata = odr.Data([x[0] for x in stabg],[x[1] for x in stabg]) # , sx=sx, sy=sy) # wd=1./power(sx,2), we=1./power(sy,2))
myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
myoutput = myodr.run()
myoutput.pprint()
print myoutput.sum_square

aa = []
bb = []
cc = []
dd = []
ee = []
for k in elections.keys():
    if elections[k].Ndists >= 11: #if int(elections[k].yr) >= 1992 and int(elections[k].yr) <= 2000:
        aa.append(elections[k].fgap)
        bb.append(elections[k].Dfrac)
        cc.append(elections[k].egap)
        dd.append(elections[k].hgap)
        ee.append(elections[k].igap)
        print "%.4f %.4f %d %s %s %s lll" % \
        (elections[k].igap,elections[k].fgap,\
        int(elections[k].Ndists),elections[k].yr,\
        elections[k].state,elections[k].chamber)
# make_scatter('drift',[elec.fgap for elec in elections],[elec.Dfrac for elec in elections],'')
# make_scatter('driftf11',bb,aa,'fgap',[0.25,0.75,-0.25,0.25])
# make_scatter('drifte11',bb,cc,'egap')
# make_scatter('drifth11',bb,dd,'hgap')
# make_scatter('drifti11',bb,ee,'igap')
# get_stabilizers(elections)
# make_histogram('hduh',[t.hgap for t in elections.values()])
plot_timeseries(elections)

####################################################
# pooled model
# lump together all races, keep incumbency as covariate



pooled_data = """
data {
  int<lower=0> N; 
  vector[N] x;
  vector[N] y;
}
"""

pooled_parameters = """
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
"""

pooled_model = """
model {
  y ~ normal(beta[1] + beta[2] * x, sigma);
}
"""

# mn_counties = srrs_mn.county.unique()
# counties = len(mn_counties)

# county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
# county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values



pooled_data_dict = {'N': len(pvote),
               'x': incum_list,
               'y': pvote}

# pooled_fit = pystan.stan(model_code=pooled_data + pooled_parameters + pooled_model, 
#                          data=pooled_data_dict, iter=1000, chains=2)
                         
# pooled_sample = pooled_fit.extract(permuted=True)
# b0, m0 = pooled_sample['beta'].T.mean(1)
# plt.figure(figsize=(12,8))

# plt.scatter(incum_list + np.random.randn(len(incum_list))*0.01, pvote, alpha=0.4)
# xvals = np.linspace(-1.2, 1.2)
# plt.plot(xvals, m0*xvals+b0, 'r--',alpha=0.4)
# plt.savefig('/home/gswarrin/research/gerrymander/gerrypooled.png')
# plt.close()

# So democrats gain 0.4968 of vote in non-incumbency races
# incumbency confers about 0.023 to holder
# print b0,m0
# print setstates[-1]
# print state_lookup['WA']
# for k in state_lookup

####################################################################
# unpooled model - ver 1 - only keep states separate

unpooled_model = """data {
  int<lower=0> N; 
  int<lower=1,upper=51> state[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[51] a;
  real beta;
  real<lower=0,upper=100> sigma;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] = beta * x[i] + a[state[i]];
}
model {
  y ~ normal(y_hat, sigma);
}"""

unpooled_data = {'N': len(pvote),
               'state': statenums, # Stan counts starting at 1
               'x': incum_list,
               'y': pvote}

# unpooled_fit = pystan.stan(model_code=unpooled_model, data=unpooled_data, iter=1000, chains=2)

unpooled_estimates = pd.Series(unpooled_fit['a'].mean(0), index=setstates)
unpooled_se = pd.Series(unpooled_fit['a'].std(0), index=setstates)

order = unpooled_estimates.sort_values().index

plt.scatter(range(len(unpooled_estimates)), unpooled_estimates[order])
for i, m, se in zip(range(len(unpooled_estimates)), unpooled_estimates[order], unpooled_se[order]):
    plt.plot([i,i], [m-se, m+se], 'b-')
plt.xlim(-1,52); plt.ylim(0,1)
plt.ylabel('Dem vote estimate');plt.xlabel('Ordered states');
plt.savefig('/home/gswarrin/gerry-unpooled-1.png')
plt.close()

# sample_states = ('LAC QUI PARLE', 'AITKIN', 'KOOCHICHING', 
#                     'DOUGLAS', 'CLAY', 'STEARNS', 'RAMSEY', 'ST LOUIS')
sample_states = ('WA','SD','MA','KS','MI','WI','IL','IN')

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
axes = axes.ravel()
m = unpooled_fit['beta'].mean(0)
for i,c in enumerate(sample_states):
    print 
    # y = srrs_mn.log_radon[srrs_mn.county==c]
    # x = srrs_mn.floor[srrs_mn.county==c]
    # only keep 
    y = map(lambda z: pvote[z], filter(lambda i: lstates[i] == c,range(len(pvote))))
    x = map(lambda z: incum_list[z], filter(lambda i: lstates[i] == c, range(len(pvote))))
    axes[i].scatter(x + np.random.randn(len(x))*0.01, y, alpha=0.4)
    
    # No pooling model
    b = unpooled_estimates[c]
    
    # Plot both models and data
    xvals = np.linspace(-1.1, 1.1)
    axes[i].plot(xvals, m*xvals+b)
    axes[i].plot(xvals, m0*xvals+b0, 'r--')
    axes[i].set_xticks([-1,0,1])
    axes[i].set_xticklabels(['D Incum', 'None', 'R Incum'])
    axes[i].set_ylim(-0.2, 1.2)
    axes[i].set_title(c)
    if not i%2:
        axes[i].set_ylabel('D share of vote')
plt.savefig('/home/gswarrin/gerry-unpoooled-1-sample.png')
plt.close()

hierarchical_guess = """
data {
  int<lower=0> M; # maximum number of districts in any state
  int<lower=0> J; # number of states
  int<lower=0> K; # number of years
  int<lower=0> N; # number of data points
  int<lower=0> Ndist[J];           # number of districts in state j
  int<lower=1,upper=M> dists[N];   # Q: number of districts varies by state...
  int<lower=1,upper=J> state[N]; 
  int<lower=1,upper=K> years[N];
  vector[N] DemInc; # indicator for whether or not there is a Democrat incumbent
  vector[N] RepInc; # indicator for whether or not there is a Republican incumbent
  vector[N] y;      # democratic percentage of vote
} 
parameters {
  vector[J] a; # state average (over all districts and all years)
  vector[M] 
  vector[2] b;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;
  vector[N] m;

  for (i in 1:N) {
    m[i] = a[county[i]] + u[i] * b[1]; # depends on function of county + globally weighted uranium measurement
    y_hat[i] = m[i] + x[i] * b[2];     # intercept depending on county + global slope
  }
}
model {
  mu_a ~ normal(0.5, .15);     # I dunno....
  a ~ normal(mu_a, sigma_a);   # state distribution?
  b ~ normal(0, 1);
  y ~ normal(y_hat, sigma_y);  
}
"""

hierarchical_intercept_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1
                          'u': u,
                          'x': floor_measure,
                          'y': log_radon}

hierarchical_intercept_fit = pystan.stan(model_code=hierarchical_intercept, data=hierarchical_intercept_data, 
                                         iter=1000, chains=2)
                                         
#a_means = M_hierarchical.a.trace().mean(axis=0)
m_means = hierarchical_intercept_fit['m'].mean(axis=0)
plt.scatter(u, m_means)
g0 = hierarchical_intercept_fit['mu_a'].mean()
g1 = hierarchical_intercept_fit['b'][:, 0].mean()
xvals = np.linspace(-1, 0.8)
plt.plot(xvals, g0+g1*xvals, 'k--')
plt.xlim(-1, 0.8)

m_se = hierarchical_intercept_fit['m'].std(axis=0)
for ui, m, se in zip(u, m_means, m_se):
    plt.plot([ui,ui], [m-se, m+se], 'b-')
plt.xlabel('County-level uranium'); plt.ylabel('Intercept estimate')
plt.savefig('/home/gswarrin/hier.png')
plt.close()

def analyze_year_race(arr,yr,state,chamber,alpha):
    """ pull out all races in a given yr,state,chamber
    """
    curid = '_'.join([yr,state,chamber])
    tmp = []
    for x in arr:
        if x[1] == curid:
            tmp.append(x)
            # print x
    dists = [x[2] for x in tmp]
    demw = 0
    repw = 0
    numd = 0
    demvic = 0
    nume = 0
    numf = 0
    numg = 0
    totseats = 0
    totvotes = 0
    # compute efficiency gap
    for x in set(dists):
        duh = filter(lambda y: y[2] == x, tmp)
        if len(duh) > 2:
            continue

        if len(duh) == 1: # unopposed
            if duh[0][5] == '100': # unopposed democrat
                a = duh[0]
                b = [a[i] for i in range(len(a))]
                b[5] = '200'
                b[-1] = '0'
                a[3] = 38000
                b[3] = 6000
            else:
                b = duh[0]
                a = [b[i] for i in range(len(b))]
                a[-1] = '0'
                a[5] = '100'
                b[3] = 38000
                a[3] = 6000
        else:
            a = duh[0]
            b = duh[1]
            if a[5] == '200': # first candidate is always democratic
                c = b
                b = a
                a = c
        if a[3] == 0: # republican unopposed
            a[3] = 6000
            b[3] = 15000
        if b[3] == 0:
            a[3] = 15000
            b[3] = 6000
        # print "a: ",a
        # print "b: ",b
        totvotes = a[3] + b[3]
        demw += 2*(a[3]-totvotes/2)*1.0/totvotes
        # if totvotes == 0: # only a handful of these AR72, KY71, 82MO, 04AR, 02HI
        #    return [],[],[],0,0,0
            # print "nan: ",duh
        if a[-1] == '1': # winner
        #    demw += (a[3]-totvotes/2)*1.0/totvotes
        #    repw += b[3]*1.0/totvotes
            demvic += 1
        # else:
        #     repw += (b[3]-totvotes/2)*1.0/totvotes
        #     demw += a[3]*1.0/totvotes
            
        # compute alpha-gap
        if totvotes > 0:
            nume += 2*(a[3]-totvotes/2)/totvotes
            numf += pow(2*(a[3]-totvotes/2)/totvotes,alpha+1)
            numg += pow(2*(a[3]-totvotes/2)/totvotes,alpha+1)/(alpha+1) + 2*(a[3]-totvotes/2)/totvotes
        numd += 1

    # print "----",yr,demvic,numd
    if numd >= 8:
        # egap = (demw-repw)/numd
        egap = nume/numd + 0.5 - demvic*1.0/numd
        fgap = (2.0/(alpha+2))*(numf/numd + 0.5 - demvic*1.0/numd)
        ggap = 2*(alpha+1.0)*numg/(numd*(2*alpha+3.0)) + (2*(alpha + 2.0)/(2*alpha + 3))*(0.5 - demvic*1.0/numd)
        # if egap*egap + fgap*fgap > 0.2:
        #     print yr,state,chamber,numd,fgap    
        # print demw,repw,egap
        return egap,fgap,ggap,demvic,numd,demw*1.0/numd
    return [],[],[],demvic,numd,demw*1.0/numd
    
    def find_unop_dists(self):
        """ find districts without two candidates for any present year
        """
        # Assume it's an "unopposed" district until proven otherwise
        self.unop_dists = [True for i in range(self.Ndists)]
        for i in range(5):
            for j in range(self.Ndists):
                actual_votes,dv,rv = compute_district_vote(self[i], j)
                if actual_votes:
                    self.unop_dists[j] = False

elections0 = dict()
for x in elections.keys():
    if 2002 <= int(elections[x].yr) <= 2010 and elections[x].chamber == '9' and elections[x].state in ['FL','MA','OK']:
        # print "duh"
        elections0[x] = elections[x].mycopy()
        # print "blah",elections[x].dcands,elections0[x].dcands
print len(elections0.keys())

reset('x')
# plot([(x^(a+1)-(1-x)^(a+1))/(x^(a+1)+(1-x)^(a+1)) for a in [0,2]], 0.01,0.99)
# plot(2*x^(43/100)*100/143 - 100/143,0,1)
# plot((1-(1-x)^3)-(1 - (1-(1-x)^3)),0,1)
plot(1- 2*(1-x)^2,0,1)

clist = ['bs-','r^-','go-','cs-','m^-','yo-']
plt.figure(figsize=(12,8))
plt.axis([1966,2016,-0.15,0.15])
plt.plot(eyrs,yrnetavge,'m^-')
plt.plot(eyrs,yrabsavge,'yo-')
plt.plot(eyrs,yrnetavgf,'bs-')
plt.plot(eyrs,yrabsavgf,'r^-')     
plt.grid(True)
plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'lowerabsnet')
plt.show()
plt.close()

import scipy.stats as stats
import numpy as numpy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from sage.plot.histogram import Histogram

plt.figure(figsize=(12,8))

# print collist
# print 
numbins = 20
duhe = [value for value in egaps if not math.isnan(value)] # filter(lambda y: y != 'nan', egaps)
duhf = [value for value in fgaps if not math.isnan(value)]
n, bins, patches = plt.hist(egaps, 50, facecolor='g', alpha=0.75)

# plt.hist(egaps, numbins) # , facecolor=(0.9, 0.9, 0.9))
# plt.legend()
# plt.xlabel(repr(yr) + '   dashed is avg; dotted is median')
# for jj in range(len(lts)):
#     plt.axvline(avgs[jj], color=collist[jj], linestyle='dashed', linewidth=2)
#     plt.axvline(meds[jj], color=collist[jj], linestyle='dotted', linewidth=2)

plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'egap')
plt.show()
plt.close()

plt.figure(figsize=(12,8))
n, bins, patches = plt.hist(fgaps, 50, color='green', alpha=0.75)
plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'fgap')
plt.show()
plt.close()

# plt.figure(figsize=(12,8))
# n, bins, patches = plt.hist([egaps,ggaps], 50, color=['red','green'], alpha=0.75)
# plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'eggap')
# plt.show()
# plt.close()

# plt.figure(figsize=(12,8))
# n, bins, patches = plt.hist([fgaps,ggaps], 50, color=['red','green'], alpha=0.75)
# plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'fggap')
# plt.show()
# plt.close()

plt.figure(figsize=(12,8))
n, bins, patches = plt.hist([egaps,fgaps], 50, color=['red','green'], alpha=0.75)
plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'efgap')
plt.show()
plt.close()

# for j in range(len(dvics)):
#     print j,dvotes[j]
#     print "    ",dvics[j]
#     print "         ",tseats[j]
#     print "                ",dvotes[j] - dvics[j]*1.0/tseats[j]

# shows that proportionality satisfied by egap does not hold as closely for fgap
# need to look more closely at equation to make sure I'm plotting what I want to
plt.figure(figsize=(8,8))
plt.scatter(egaps,[dvotes[i] - dvics[i]*1.0/tseats[i] for i in range(len(dvics))])
plt.grid(True)
plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'svcurvee')
plt.show()
plt.close()

plt.figure(figsize=(8,8))
plt.scatter([egaps[i]-fgaps[i] for i in range(len(egaps))],[dvics[i]*1.0/tseats[i] for i in range(len(dvics))])
plt.grid(True)
plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'dev')
plt.show()
plt.close()

plt.figure(figsize=(8,8))
plt.scatter(fgaps,[dvotes[i] - dvics[i]*1.0/tseats[i] for i in range(len(dvics))])
plt.grid(True)
plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'svcurvef')
plt.show()
plt.close()

plt.figure(figsize=(8,8))
plt.scatter([x[0] for x in stabe],[x[1] for x in stabe])
plt.grid(True)
plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'stabe')
plt.show()
plt.close()

plt.figure(figsize=(8,8))
plt.scatter([x[0] for x in stabf],[x[1] for x in stabf])
plt.grid(True)
plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'stabf')
plt.show()
plt.close()

plt.figure(figsize=(8,8))
plt.scatter([x[0] for x in trendf],[x[1] for x in trendf])
plt.grid(True)
plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'ftrend')
plt.show()
plt.close()

# plt.figure(figsize=(8,8))
# plt.scatter([x[0] for x in stabg],[x[1] for x in stabg])
# plt.grid(True)
# plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'stabg')
# plt.show()
# plt.close()

plt.figure(figsize=(8,8))
plt.scatter(egaps,fgaps)
plt.grid(True)
plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'scatteref')
plt.show()
plt.close()

# plt.figure(figsize=(8,8))
# plt.scatter(egaps,ggaps)
# plt.grid(True)
# plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'scattereg')
# plt.show()
# plt.close()

# plt.figure(figsize=(8,8))
# plt.scatter(fgaps,ggaps)
# plt.grid(True)
# plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'scatterfg')
# plt.show()
# plt.close()

# for x in sorted(egaps):
#    print '.%.4f.' % (x)
slope, intercept, r_value, p_value, std_err = stats.linregress(stabf)
print r_value**2
print slope

# just look at one state

for st in ['TN']: # states: 
    for chamber in ['11']: # ['8','9','11']: 
        for yr in ['2008','2010','2012','2014']: # yrs:
            tmpe,tmpf,tmpg,seats,totseats,duh = analyze_year_race(arr,yr,st,chamber,2)
            # print yr,chamber,tmpe,tmpf,tmpg,seats,totseats,duh
            if tmpe != [] and tmpf != [] and not math.isnan(tmpe) and not math.isnan(tmpf):
                print yr,chamber,tmpf,tmpe


# for i in range(len(dvics)):
#    print len(egaps),len(fgaps),len(ggaps),len(dvics),len(tseats),len(dvotes)
# print l8[0]

# print out elections in a given state
st = 'GA'
chamber = '11'
ans = []
myyrs = ['2004','2006','2008','2010','2012','2014']
for x in arr:
    a,b,c = x[1].split('_')
    if b == st and c == chamber and a in myyrs:
        ans.append(x)
ans = sorted(ans)
for x in ans:
    print x

# plot fgap over time for each state

l8 = []
l9 = []
l11 = []
for st in states:
    for chamber in ['8','9','11']:
        tmpl = []
        tmpr = []
        tmps = []
        for yr in yrs:
            tmpe,tmpf,tmpg,demvic,totseats,demw = analyze_year_race(arr,yr,st,chamber,2)
            # print "return: ",st,chamber,yr,tmpe,tmpf,tmpg,demvic,totseats,demw
            if tmpe != [] and tmpf != [] and tmpg != [] and \
                not math.isnan(tmpe) and not math.isnan(tmpf) and not math.isnan(tmpg):
                tmpl.append(yr)
                tmpr.append(tmpf)
                tmps.append(tmpe)
        if chamber == '8':
            l8.append([st,tmpl,tmpr,tmps])
        if chamber == '9':
            l9.append([st,tmpl,tmpr,tmps])
        if chamber == '11':
            # print "blah: ",st,tmpl,tmpr
            l11.append([st,tmpl,tmpr,tmps])

# all states

egaps = []
fgaps = []
ggaps = []
edict = dict()
fdict = dict()
stabe = []
stabf = []
stabg = []
trendf = []
dvics = []
tseats = []
dvotes = []

yrnetavge = []
yrabsavge = []
yrnetavgf = []
yrabsavgf = []
eyrs = []

for yr in yrs:
    if int(yr) % 2 == 1:
        continue
    eyrs.append(int(yr))
    tyrnetavge = []
    tyrabsavge = []
    tyrnetavgf = []
    tyrabsavgf = []
    for st in states:
        otmpe = []
        otmpf = []
        otmpg = []
        for chamber in ['8','9']:
            # if int(yr) < 2002 or int(yr) >= 2012:
            #     continue
            tmpe,tmpf,tmpg,demvic,totseats,demw = analyze_year_race(arr,yr,st,chamber,2)
            # print st,chamber,yr,tmpe,tmpf,tmpg
            if tmpe != [] and tmpf != [] and tmpg != [] and \
                not math.isnan(tmpe) and not math.isnan(tmpf) and not math.isnan(tmpg):
                # print abs(tmpe-tmpf),yr,st,chamber,demvic,totseats
                egaps.append(tmpe)
                fgaps.append(tmpf)
                ggaps.append(tmpg)
                dvics.append(demvic)
                tseats.append(totseats)
                dvotes.append(demw)
                tyrnetavge.append(tmpe)
                tyrabsavge.append(abs(tmpe))
                tyrnetavgf.append(tmpf)
                tyrabsavgf.append(abs(tmpf))
                # print tmpe,tmpf,tmpg,demvic,totseats,demw
#                trenf.append([yr,tmpf])
                if otmpe != [] and not math.isnan(otmpe) and yr[-1] != '2':
                    stabe.append([otmpe,tmpe])
                if otmpf != [] and not math.isnan(otmpf) and yr[-1] != '2':
                    stabf.append([otmpf,tmpf])
                if otmpg != [] and not math.isnan(otmpg) and yr[-1] != '2':
                    stabg.append([otmpg,tmpg])
                otmpe = tmpe
                otmpf = tmpf
                otmpg = tmpg
    yrnetavge.append(1.0*sum(tyrnetavge)/len(tyrnetavge))       
    yrabsavge.append(1.0*sum(tyrabsavge)/len(tyrabsavge))       
    yrnetavgf.append(1.0*sum(tyrnetavgf)/len(tyrnetavgf))       
    yrabsavgf.append(1.0*sum(tyrabsavgf)/len(tyrabsavgf))
    print "Done with",yr       
print len(egaps),len(fgaps),len(ggaps)

slope, intercept, r_value, p_value, std_err = stats.linregress(aaa)
print r_value**2
print slope

#     def compute_linear_gaps(self):
#         """ compute gaps corresponding to three-segmented line and its perturbations
#         """  
#         numarr = [0 for j in range(len(self.numpow))]
#         totvotes = 0
#         self.Ddists = 0
#         # compute efficiency gap
#         for i in range(self.Ndists):
#             actual_votes,dv,rv = self.compute_district_vote(i)
#             if actual_votes:
#                 self.dcands[i].frac = 1.0*dv/(dv+rv)
#                 self.rcands[i].frac = 1.0*rv/(dv+rv)
#             if dv > rv:
#                 self.Ddists += 1        
#             totvotes = dv + rv
#             # compute alpha-gap
#             if totvotes > 0:
#                 ai = 2*(dv-totvotes/2)/totvotes
#                 for r1 in range(3):
#                     betaprime = max(0,self.posbeta + (r1-2)*self.delta)
#                     for c1 in range(3):
#                         bprime = max(0,self.posb + (c1-2)*self.delta)
#                         for r2 in range(3):
#                             gammaprime = min(1,max(betaprime,self.posgamma + (r2-2)*self.delta))
#                             for c2 in range(3):
#                                 pos = (3*r1+c1)*9 + 3*r2 + c2
#                                 cprime = min(1,max(bprime,self.posc + (c2-2)*self.delta))
# #                                 self.newtongaps
# #                 numarr[0] += ai
# #                 for j in range(81):
# #                     if ai >= 0:
# #                         linarr[j] += pow(ai,self.numpow[j]+1)
# #                     else:
# #                         linarr[j] -= pow(-ai,self.numpow[j]+1)
                       
        # self.egap = nume/self.Ndists + 0.5 - self.Ddists*1.0/self.Ndists
        # self.fgap = (2.0/(alpha+2))*(numf/self.Ndists + 0.5 - self.Ddists*1.0/self.Ndists)
        # self.hgap = self.fgap + self.Dfrac - 0.5
        # A = 1.0/3+42.0/125
        # self.igap = 2/(1+A)*(numi/self.Ndists + A/2 - self.Ddists*1.0*A/self.Ndists)
        # return self.egap,self.fgap,self.hgap,self.igap

#############################################################################

# predd = cycles113[0].leveliii_fit.extract()
print mean(predd['u_0jk'][:,1+randrange(len(predd['u_0jk'][0])-1)])
print cycles113[0].dwin[0]*np.random.randn(1)[0] + cycles113[0].dwin[1]

# electionsa = make_records(arra)
# new9cycles = create_cycles(electionsa)
# cycles113 = create_cycles(electionsa)
# for elec in elections.values():
#     if elec.yr == '2014' and elec.state == 'MN' and elec.chamber == '11':
#         elec.myprint()
# for cyc in cycles:
#     for k in cyc.elecs.keys():
#         if c.elecs[k].yr == '2014' and \
# c.elecs[k].state == 'NM' and c.elecs[k].chamber == '11':
#             c.elecs[k].myprint()
#             elec = c.elecs[k]
#             for i in range(elec.Ndists):
#                 print "%s %.3f %.3f %s" % 
# elec.dists[i],elec.dcands[i].votes,\
# elec.dcands[i].votes*1.0/(elec.dcands[i].votes+elec.rcands[i].votes),\
#                 elec.dcands[i].is_imputed)


####################################################################
# maybe for trying to figure out geographic influence
duh = []
for i in range(len(wipres2016)):
    for x in range(wictysz[i]/min(wictysz)):
        duh.append(wipres2016[i])
wipres = sorted(duh)
make_scatter('wipres.png',[i for i in range(len(wipres))], wipres)
print find_lines('blah',wipres)

duhFL = []
for i in range(len(FLpresper)):
    for x in range(FLpresto[i]/min(FLpresto)/4):
        duhFL.append(FLpresper[i])
FLpres = sorted(duhFL)
make_scatter('FLpres.png',[i for i in range(len(FLpres))], FLpres)
print find_lines('blahFL',FLpres)

ddd = [0.1098916059860295, 0.078314953888994648, 0.015536998333180747, 0.075781786996990688, 0.17556838945162218, 0.094709832184419807, 0.14985701254448008, 0.014318149652842605, 0.15991295946902689, 0.13961103915106715, 0.094991022996636768, 0.029211359206278341, 0.22379579732143715, 0.10957102431573586, 0.014633359180795492, 0.053923406712417091, 0.076442657670168335, 0.14388962569513211, 0.20668165141431019, 0.14312424841605442, 0.13093627676519914, 0.060678171221509106, 0.17624737838277865, 0.26385547878852972, 0.090353161199194149, 0.13298318433853409, 0.077751837188067871, -0.0086303671046180034, 0.16113351175552537, 0.27654618356595867, 0.17171486652749993, 0.26899620806856761, 0.11019690030812715, 0.21106395583690668, 0.16643454018350018, 0.10528020868842942, -0.010413174533659995, 0.18016555433360934, 0.043194711300751636, 0.043297832423240604, 0.21718405073384456, 0.18564773202668614, 0.11333225911748836, 0.23171705373256885, 0.19959353460792301, 0.28999076775124932, 0.026596387234388487, 0.068976410597764853, 0.14419118553565627, 0.065281759731061179, 0.0730639486582874, 0.17859468661952299, 0.016362253553092232, 0.072880864299818687, 0.04524137854708235, 0.10069361228154115, 0.071022668867859171, 0.23757080125106103, 0.15952327479083248, 0.093308958335607484, 0.22075232791416682, 0.065793349216747898, 0.072296979689182694, 0.08767249322252936, 0.065042040050871067, 0.020716308011371726]
rrr = [-0.097034704448389589, -0.065129879401265134, 0.0069443233283567199, 0.013777031137095797, -0.16029243465981244, 0.004309882798580288, -0.074158728464375656, 0.0025597300411217775, -0.071093428843062259, -0.024723951492172311, -0.12026031314544451, -0.018441236115663059, -0.15927904259437656, -0.078172012669366325, -0.020799317120319721, -0.077013054705263984, -0.037131497425568406, -0.072370256713452347, -0.030315472516537191, -0.032204168474284603, -0.05524887389722704, -0.064051639107358713, -0.038888691968359305, -0.0046453801104924764, -0.012220160247774201, -0.0058891442877285682, -0.062341528631831282, -0.021702934338637524, -0.092354094292833908, -0.040664211523826398, -0.06866225435483958, -0.070312300330018312, -0.045121998990645534, -0.0046201838450130083, -0.14319416952022981, -0.038827678917363127, -0.0089861550401909972, -0.0079693924605857211, -0.014060659072959005, -0.087960978213913063, -0.021102151988722985, -0.023955184334140588, -0.051207398647388956, -0.087390067218920023, -0.1014337179205422, -0.0091676041446700236, -0.050558489642862012, 0.012496402592797167, -0.026928977845758623, 0.019459531614180067, 0.00079572809468154992, -0.055036696404148926, -0.12943628843390576, 0.011309767955345842, -0.012005549127472721, 0.018622390223447076, -0.050584719830861193, -0.088605774243271526, -0.033641508731634344, -0.07559774391574911, 0.0044475015477650716, -0.1415416322250731, -0.094659297667659875, -0.077436153777579772, -0.08993125380944425, -0.07165255625699786, -0.074839266393063181, -0.088475530201886921, -0.040244435661597341, -0.13145132135753876, -0.047148390017807025, 0.019863323144659262, 0.029568136870770929, -0.013507234615792719, -0.099103388138028464, -0.041651167861076012, -0.0076987466068769153, -0.027381943004970731, -0.030730063043250225, -0.080101259868716509, -0.031532171159693659, -0.019321187981984549, -0.062796934679821184, -0.045299621429144961, -0.052149291296970685, -0.08220776448781425, -0.066831735209152351, -0.14626923859122457, -0.1301449833994506, -0.073161184726746029, -0.037848483546622333, -0.055523171761698367, -0.19620700281877504, -0.12060857419503708, -0.0045684259169825056, -0.10868496501682262, -0.11175496401874992, -0.0070109229669026414, -0.061385560117071504, -0.1077527830274831, -0.13420368880502503, -0.050614331497039804, -0.029281970154291868, -0.11338680346177872, -0.066083250253612294, 0.03979382337745619, -0.13329778002864107, -0.1058810096880944, -0.099128340305606716, -0.0042915936391773341, 0.0027497646131315009, -0.068587557820371695, -0.011206176125649208, -0.048021963884350399, -0.024708755255629291, -0.02175801132927498, -0.11696469771376004, -0.050259200510938497, -0.054300314485043986, -0.056329731238190427, -0.0089913414193152283, -0.042110532591252706, -0.041222227663628201, -0.030716179652804512, -0.047645605791835972, -0.023057399431856019, -0.029427861668090111, -0.044941819016303841, -0.059904332781576031, -0.061966813183841549, -0.052862109452579145, -0.029537608835177403, -0.018460450072940497, -0.039736970052435268, -0.07722410786295901, -0.016897996684091188, -0.048654839077223372, -0.020084923805364162, -0.0009980651168360645, -0.00037674925768748968, -0.031084512653234617, 0.015537727631878636, 0.048032590416757157, -0.046279808782090925, -0.066797319061708188, -0.04719223636310859, -0.055546162558405117, -0.068686421408219733, -0.1234995247987149, -0.070416558279676744, -0.0098677006416240422, -0.040377259895597588, -0.027974119080856633, 0.028861862480028033, -0.069830214007970307, 0.03053094448736559, -0.036797588303828725, -0.12543808802545126, -0.057715895199214849, -0.034362552854368102, -0.073688940870644495, -0.0022046701784624364, -0.059656882404032982, 0.035740542116192993, -0.055061383287299717, -0.072725069908226511, -0.11648387003654391, -0.17597711697312712]

make_histogram('demhisto.png',ddd)

make_histogram('rephisto.png',rrr)

for st in states:
    k = '2000_' + st + '_9'
    if k in elections.keys():
        print st,elections[k].Ndists

####################################################
# trying to model vote distributions by normals
# - Egap uses uniform from 25% to 75%
# - can do based on states or based on individual districts
# - looks like 0.07 to 0.10 for states; about 0.16 for individual districts

# import numpy as np
from scipy.stats import norm

def make_histogram_fit(fn,arr,mu,std):
    """
    """
    plt.figure(figsize=(12,8))

    numbins = 20
    n, bins, patches = plt.hist(arr, 50, normed=True, facecolor='g', alpha=0.75)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    plt.show()
    plt.close()

ans = []
for elec in elections.values():
    # elec.compute_overall()
    if not math.isnan(elec.Dfrac) and 0.70 >= elec.Dfrac >= 0.65: #  and 2010 >= int(elec.yr) >= 2000: 
        for i in range(elec.Ndists):
            actual_votes,dv,rv = elec.compute_district_vote(i)
            if actual_votes and (dv + rv > 0):
                ans.append((dv*1.0)/(dv+rv))
             # and elec.chamber=='9':
        # ans.append(elec.Dfrac)
mu, std = norm.fit(ans)
print mu,std
print len(ans)
make_histogram_fit('votedist.png',ans,mu,std)

ev = list(filter(lambda x: x.Ndists >= 8 and (2016 >= int(x.yr) >= 2002) and x.chamber == '9', elections.values()))
ev = filter(lambda x: len(filter(lambda z: z == True, map(lambda y: math.isnan(y), x.adjgaps[:2] + x.adjgaps[3:]))) == 0, ev)
# print len(filter(lambda y: y == True, map(lambda z: math.isnan(z), x.adjgaps)))

# nev = []
# for x in ev[:10]:
#     duh = x.adjgaps
#     print duh
    
arrmax = [[0 for x in range(6)] for j in range(len(ev))]
arrmin = [[0 for x in range(6)] for j in range(len(ev))]
for i in range(len(ev)):
    # print ev[i].yr,ev[i].state,ev[i].chamber,ev[i].adjgaps
    for j in range(len(ev)):
        for k in range(6):
            if ev[j].adjgaps[k] > ev[i].adjgaps[k]:
                arrmax[i][k] += 1
            if ev[j].adjgaps[k] < ev[i].adjgaps[k]:
                arrmin[i][k] += 1
for i in range(len(ev)):
    if (min(arrmax[i][:2]) <= 5 or min(arrmax[i][3:]) <= 5) and ev[i].Ndists >= 8:
        print "max ",i,ev[i].yr,ev[i].state,ev[i].chamber,arrmax[i]
for i in range(len(ev)):
    if (min(arrmin[i][:2]) <= 5 or min(arrmin[i][3:]) <= 5) and ev[i].Ndists >= 8:
        print "min ",i,ev[i].yr,ev[i].state,ev[i].chamber,arrmin[i]
for i in range(len(ev)):
    if ev[i].state == 'WI':
        print ev[i].yr,ev[i].chamber,arrmin[i],arrmax[i]
# restrict attention to certain states/years (using mmd_dict)?
# come up with examples under different functions
# see where WI fits under this....

def rank_outliers(elections,seyr,st,sechm):
    """ rank extremeness by running over various values of alpha
    """
    secnt = None # [0 for x in range(len(elec.numpow))]        # count how many states for each j
    for elec in elections.values():
        if elec.yr == seyr and elec.chamber == sechm and elec.Ndists >= 8:
            if secnt == None:
                secnt = [0 for x in range(len(elec.numpow))]
            # run through states that we'll rank according to each alpha
            setot = 0
            # for st in states:
            curkey = '_'.join([seyr,st,sechm])    # state we're holding constant
            setot += 1                            # this is just total number of states we're comparing to
            for j in range(len(elec.numpow)):
                # print "j: ",j
                # print elections[curkey].adjgaps[j]
                if not math.isnan(elec.adjgaps[j]) and abs(elec.adjgaps[j]) > abs(elections[curkey].adjgaps[j]):
                    # print "%s %d vs. %d; %.3f > %.3f" % (elec.state,elec.Ndists,elections[curkey].Ndists,\
                    #     elec.adjgaps[j],elections[curkey].adjgaps[j])
                    secnt[j] += 1
    # print secnt
    print "%s %2d %s" % (st,min(secnt),secnt)  #.index(min(secnt))
        # for i in range(len(subelec)):
        # print "%s: percent less normal %.3f" % (subelec[i],secnt[i]*1.0/setot[i])

for st in states:
    if elections['_'.join(['2014',st,'11'])].Ndists >= 8 and not math.isnan(elections['_'.join(['2014',st,'11'])].adjgaps[0]):
        rank_outliers(elections,'2014',st,'11')


        ##########################################################
# try to recompute EGap numbers 
# Working with 2000's MN 11 because nothing imputed
##########################################################
    
# make_scatter_stlabels('all-angle-alpha',blah,blah2,labs)
# print stats.pearsonr(blah,blah2)
for elec in []: # electionsa.values():
    if int(elec.yr) >= 1972 and True not in map(lambda i: elec.dcands[i].is_imputed, range(elec.Ndists)) and \
        elec.Ndists >= 8:
        print int((int(elec.yr)-2.0)/10),elec.state,elec.chamber 
        
print compute_egap_directly([0.7,0.7,0.7,0.54,0.54,0.54,0.54,0.54,0.35,0.35])
st = 'VA'
# for yr in ['2002','2004','2006','2008','2010']:
# for yr in ['1992','1994','1996','1998','2000']:
for yr in ['1972','1974','1976','1978','1980']:
    elec = electionsa[yr + '_' + st + '_9']
    N = elec.Ndists
    # print 
    tt = compute_egap_directly(elec.demfrac)
    print "%.3f %.3f" % (tt,tt*N)
    print compute_alpha_curve(elec.demfrac,0)/1 # (2.0/N)
    print

###################################################################################
# for deleting all those elections with issues...
###################################################################################
load(homepath + 'basic.py')
for elec in Melections.values():
    # elec.myprint()
    # print "yr: ",elec.yr
    if int(elec.yr) >= 1972 and (elec.yr not in Mmmd.keys() or elec.state not in Mmmd[elec.yr]):
        if min(elec.demfrac) < 0.05:
            ttmp = '_'.join([elec.yr,elec.state,elec.chamber])
            # print ttmp
            del Melections[ttmp]
            print elec.yr,elec.state,elec.chamber
    
def get_angle(p0, p1=np.array([0,0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def rotation_transform(theta):
    ''' rotation matrix given theta
    Inputs:
        theta    - theta (in degrees)
    '''
    theta = np.radians(theta)
    A = [[np.math.cos(theta), -np.math.sin(theta)],
         [np.math.sin(theta), np.math.cos(theta)]]
    return np.array(A)

def add_corner_arc(axes, line, radius=.7, color=None, text=None, text_radius=.5, text_rotatation=0, **kwargs):
    ''' display an arc for p0p1p2 angle
    Inputs:
        ax     - axis to add arc to
        line   - MATPLOTLIB line consisting of 3 points of the corner
        radius - radius to add arc
        color  - color of the arc
        text   - text to show on corner
        text_radius     - radius to add text
        text_rotatation - extra rotation for text
        kwargs - other arguments to pass to Arc
    '''

    lxy = line.get_xydata()

    if len(lxy) < 3:
        raise ValueError('at least 3 points in line must be available')

    p0 = lxy[0]
    p1 = lxy[1]
    p2 = lxy[2]

    width = np.ptp([p0[0], p1[0], p2[0]])
    height = np.ptp([p0[1], p1[1], p2[1]])

    n = np.array([width, height]) * 1.0
    p0_ = (p0 - p1) / n
    p1_ = (p1 - p1)
    p2_ = (p2 - p1) / n 

    theta0 = -get_angle(p0_, p1_)
    theta1 = -get_angle(p2_, p1_)

    if color is None:
        # Uses the color line if color parameter is not passed.
        color = line.get_color() 
    arc = axes.add_patch(matplotlib.patches.Arc(p1, width * radius, height * radius, 0, theta0, theta1, color=color, **kwargs))

    if text:
        v = p2_ / np.linalg.norm(p2_)
        if theta0 < 0:
            theta0 = theta0 + 360
        if theta1 < 0:
            theta1 = theta1 + 360
        theta = (theta0 - theta1) / 2 + text_rotatation
        pt = np.dot(rotation_transform(theta), v[:,None]).T * n * text_radius
        pt = pt + p1
        pt = pt.squeeze()
        axes.text(pt[0], pt[1], text,         
                horizontalalignment='left',
                verticalalignment='top',)

    return arc   

# plot_snr_corr(Melections)
# min_correlation(Melections)

load(homepath + 'basic.py')
for elec in Melections.values():
    # elec.myprint()
    # print "yr: ",elec.yr
    if int(elec.yr) >= 1972 and (elec.yr not in Mmmd.keys() or elec.state not in Mmmd[elec.yr]):
        if 2 not in elec.status:
            # ttmp = '_'.join([elec.yr,elec.state,elec.chamber])
            # print ttmp
            # del Melections[ttmp]
            print "%s %s % 3d %s" % (elec.yr,elec.state,elec.Ndists,elec.chamber)

################################################################
# find alpha that minimizes total variation over cycle
################################################################

def min_total_variation(cycle):
    """ see above description
    """
    alphavals = np.linspace(-4,-1,41)
    for c in cycle:
        yrs = [str(int(c.min_year)+2*i) for i in \
            range((int(c.max_year)-int(c.min_year))/2)]
        arr = [[None for j in range(len(yrs))] for i in range(len(alphavals))]
        
        minval = []
        for st in states:
            elecs = filter(lambda x: x.state == st, c.elecs.values())
            tmp = []
            for elec in elecs:
                if elec.yr not in yrs:
                    continue
                idx = yrs.index(elec.yr)
                vals = []
                for k in range(elec.Ndists):
                    vals.append(elec.demfrac[k])
                for i in range(len(alphavals)):
                    arr[i][idx] = compute_alpha_curve(vals,alphavals[i])
            for i in range(len(alphavals)):
                tmp.append(max(arr[i])-min(arr[i]))
            minval.append(alphavals[tmp.index(min(tmp))])
            print c.min_year,st,minval[-1]
    make_histogram('minval.png',minval)

def plot_options(cycle):
    """ see how scatter plots match for various values of alpha
    """
    alphavals = np.linspace(-3,-1,11)
    ans1 = []
    ans2 = []
    for c in cycle:
        for elec in c.elecs.values():
            for i in range(len(alphavals)):
                ans1.append(alphavals[i] + randrange(5)/40)
                ans2.append(compute_alpha_curve(elec.demfrac,alphavals[i]))
    make_scatter('alpha-poss.png',ans1,ans2)
    
def list_alpha(cycle):
    """ see how scatter plots match for various values of alpha
    """
    # alphavals = np.linspace(-3,-1,11)
    alphavals = [-2,0]
    ans1 = []
    ans2 = []
    for c in cycle:
        for elec in c.elecs.values():
            for i in range(len(alphavals)):
                tmp = compute_alpha_curve(elec.demfrac,alphavals[i])
                print "alpha=%.2f %s %s %.3f" % (alphavals[i],elec.yr,elec.state,tmp)

def check_percent(elections):
    """ see if 1-gap can have overall percent vote less than that for 0-gap
    """
    fgap = []
    per = []
    for elec in electionsa.values():
        duh = compute_alpha_curve(elec.demfrac,1)
        d2 = len(filter(lambda x: x >= 0.5, elec.demfrac))*1.0/elec.Ndists
        if int(elec.yr) >= 1972 and abs(duh) < 0.1 and d2 >= 0.5 and elec.Ndists >= 8 :
            fgap.append(compute_alpha_curve(elec.demfrac,1))
            per.append(2*(np.mean(elec.demfrac)-1/2)+(1/2-d2))
            print "%.3f, %.3f %s %s %s" % (fgap[-1],per[-1],elec.yr,elec.state,elec.chamber)
    make_scatter('lt',fgap,per)
        
check_percent(electionsa)
# min_total_variation(cycles11)
# plot_options(cycles11)
# list_alpha(cycles11)

###########################################################################################
# see which value of tau leads to the highest correlation with angle
###########################################################################################
blah = []
blah2 = []
aaa = []
labs = []
xvals = np.linspace(0.35,0.45,11)
for x in []: # xvals:
    blah = []
    blah2 = []
    aaa = []
    labs = []
    for elec in electionsa.values():
        if int(elec.yr) >= 1982 and \
            int(elec.yr)%2 == 0 and \
            (elec.yr not in mmd_dict.keys() or elec.state not in mmd_dict[elec.yr]):
            tmp = find_angle(elec.state,elec.demfrac)
            tmp2 = compute_alpha_curve(elec.demfrac,x)
            if tmp != None and elec.Ndists >= 8 and tmp <= -0.5 and tmp2 <= -0.3:
                blah.append(tmp)
                blah2.append(tmp2)
                aaa.append(elec.Ndists)
                if 1 == 1 or elec.state in ['MI','NC','VA','OH','FL','PA']:
                    labs.append(elec.state + elec.yr[2:])
                else:
                    labs.append('')
                # print "ang: % .3f agap: % .3f %s %s %s" % \
                # (tmp,tmp2,elec.yr,elec.state,elec.chamber)
    print x,stats.pearsonr(blah,blah2)
        # make_histogram('all-hist-angle',blah)


#########################################################################################3
# incorporate the 2016 congressional election data
# for when i added in 2016 data after the fact
load(homepath + 'read_data.py')
load(homepath + 'classes.py')
arr6,yrs6,states6 = read_jacobson_csv()
arr8,yrs8,states8 = read_2012_state_csv('/home/gswarrin/research/gerrymander/data/cong2016.csv','11')
for x in arr6:
    if int(x[0][:4]) >= 2012:
        arr8.append(x)
Melections = make_records(arr8,Melections)
# reset status so that imputed values get reimputed with new model
for elec in Melections.values():
    if int(elec.yr) >= 2012 and elec.chamber == '11':
        for i in range(elec.Ndists):
            if elec.status[i] == 1:
                print "resetting for imputation: ",elec.yr,elec.state,elec.dist[i],elec.demfrac[i]
                elec.status[i] = 0
recent_cong_cycles = create_cycles(Melections,Mmmd,False,True,False)
for c in recent_cong_cycles:
    for k in c.elecs.keys():
        c.elecs[k].compute_overall()       

for elec in Melections.values():
    for i in range(elec.Ndists):
        if elec.demfrac[i] >= 1:
            elec.demfrac[i] = 0.95

######################################################################3
# resetting stuff to be reimputed
elec = Melections['2014_NY_11']
print map(lambda x: type(x), Melections['2016_NY_11'].status)
for elec in []: # Melections.values():
    if int(elec.yr) >= 2012 and elec.chamber == '11':
        for i in range(elec.Ndists):
            print elec.yr,elec.state,elec.dists[i],elec.status[i]
            if elec.status[i] == 1:
                print "resetting for imputation: ",elec.yr,elec.state,elec.dists[i],elec.demfrac[i]
                # elec.status[i] = 0
print arr8[0]
# print Melections['2012_NY_11']            

######################################################################3
# loc_arra,loc_yrs,loc_states,loc_mmd_dict = read_all_data()
# loc_elections = make_records(loc_arra)
# print loc_elections['2006_NY_11'].status

######################################################################3
print stlist.index('NY')
print Melections['2008_NY_11'].demfrac[8] # ,Melections['2006_MA_11'].status
for x in Marr:
    if x[0][:2] == '20' and x[0][5:7] == 'NY' and x[0][-1] == '1' and x[1] == 'NY3209':
        print x

######################################################################
# for looking at time series? Not sure
st = 'NC'
chm = '11'
for yr in ['1982','1984','1986','1988','1990']:
# for yr in ['1972','1974','1976','1978','1980']:
# for yr in ['2002','2004','2006','2008','2010']:
# for yr in ['1992','1994','1996','1998','2000']:
# for yr in ['1983','1985','1987','1989','1991']:
    print Melections[yr + '_' + st + '_' + chm].demfrac
    # print compute_alpha_curve(electionsa[yr + '_' + st + '_' + chm].demfrac,1), find_angle(st,electionsa[yr + '_' + st + '_' + chm].demfrac), compute_egap_directly(electionsa[yr + '_' + st + '_' + chm].demfrac)        


#####################################################################
# seats model
  # Random effects distribution
  # why don't I need random effects for others?
  # u_0k ~ normal(0, sigma_u0k); # expect state variation from global average to be normal around 0
  # v_0l ~ normal(0, sigma_v0l); # expect year variation from global average to be normal around 0
  # delta_D ~ normal(0, sigma_D); # not needed?
  # delta_R ~ normal(0, sigma_R); # not needed?
  # delta_V ~ normal(0, sigma_V); # not needed?

    
