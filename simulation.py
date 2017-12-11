import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import numpy as np
import random

# from classes import *
# from read_data import *

DR_seats = {2016: [194,241], 2012: [201,234], 2008: [257,178], 2004: [201,233],\
            2000: [213,220], 1996: [207,226], 1992: [258,176], 1988: [260,175],\
            1984: [254,181], 1980: [243,192], 1976: [292,143]}


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(xrange(n), r))
    return tuple(pool[i] for i in indices)

def get_declination(st,vals):
    """ Now expressed as a fraction of 90 degrees
    """
    bel = sorted(filter(lambda x: x <=  0.5, vals))
    abo = sorted(filter(lambda x: x > 0.5, vals))
    if len(bel) < 1: # democrats don't lose anything; skewed in their favor
        return -2.0
    if len(abo) < 1: # republicans don't lose anything
        return 2.0

    # print np.mean(bel),np.mean(abo)
    theta = np.arctan((1-2*np.mean(bel))*len(vals)/len(bel))
    gamma = np.arctan((2*np.mean(abo)-1)*len(vals)/len(abo))
    # print "theta: %.3f gamma: %.3f" % (theta,gamma)
    return 2.0*(gamma-theta)/3.1415926535 # len(vals). Hah!

#######################################################################################################
def fit_logistic(legyrs,use_incum=True):
    """ read in who won the district versus mccain share of vote
        fit logistic model using Jacobson data
    """
    statelist = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
      
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    cnt = 0 # how many lines we've seen
    tot = 0
    ans = []
    yrs = []
    states = []
    myids = []
    sq = [[0,0],[0,0]]

    xarr = []
    # yarr = []
    zarr = []

    # di_jit = [] # dem incumbents
    # ri_jit = [] # rep incumbents
    # di_xva = []
    # ri_xva = []

    # for keeping track of who won for a given vote share
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
        incum = 2*int(l[2]) - 1 # so it's symmetric
        # int(l[2]) > 1 and 
        if int(l[0]) in legyrs and l[12] != '' and l[2] != '' and l[3] != '' and \
           int(l[3]) <= 1 and int(l[2]) <= 1:
            # dem's share of major-party presidential vote in most recent election
            xarr.append(float(l[12])/100) 
            # jittered values for who won
            yijit.append(int(l[3]) + np.random.rand(1,1)[0][0]/10-0.05)        # who won

            #############################################################
            # trying to plot something relating to incumbency, not sure what
            # if int(l[2]) == 1: # dem incumbent
            #     di_jit.append(int(l[3]) + np.random.rand(1,1)[0][0]/10)        
            #     # di_xva.append(float(l[12])/100)
            # if int(l[2]) == 0: # dem incumbent
            #     ri_jit.append(int(l[3]) + np.random.rand(1,1)[0][0]/10-0.1)    # who won
            #     # ri_xva.append(float(l[12])/100)

            #############################################################
            # first column is the value we're trying to predict (who won)
            # second column is dem vote share
            # third column is incumbency
            # last column is for intercept (?)
            if use_incum:
                zarr.append([int(l[3]),xarr[-1],incum,1])
            else:
                zarr.append([int(l[3]),xarr[-1],1])

            # for keeping track of possible combinations of incumbents and who won
            # ignores cases in which neither is an incumbent
            if int(l[2]) <= 1:
                sq[int(l[3])][int(l[2])] += 1
            tot += 1

    # print tot
    # print sq

    # make the model
    npz = np.array(zarr)
    logit = sm.Logit(npz[:,0],npz[:,1:])

    # fit the model
    result = logit.fit()
    # print result.params
    # print result.summary()
    params = result.params

    fig = plt.figure(figsize=(8,8))
    xvals = np.linspace(0,1,101)
    # yvals = [1/(1+np.exp(-(0.1711*t-8.4084))) for t in xvals]
    # yvals = [1/(1+np.exp(-(params[0]*t+params[1]+params[2]))) for t in xvals]
    # zvals = [1/(1+np.exp(-(params[0]*t-params[1]+params[2]))) for t in xvals]
    if use_incum:
        wvals = [1/(1+np.exp(-(params[0]*t+params[1]+params[2]))) for t in xvals]
    else:
        wvals = [1/(1+np.exp(-(params[0]*t+params[1]))) for t in xvals]
    # print yvals[:10]
    # plt.plot(xvals,yvals,'b-')
    # plt.plot(xvals,zvals,'r-')
    plt.plot(xvals,wvals,'g-')
    plt.axhline(0.5)
    plt.axvline(0.5)
    plt.grid()

    ###################################################################
    # plot actual cumulative prob function
    xpdf = []
    ypdf = []
    mydel = 0.05
    for t in xvals:
        # get indices of vote fractions <= t
        rel = filter(lambda x: t-mydel <= xarr[x] <= t+mydel, range(len(xarr)))
        # get fraction won by dems
        drel = filter(lambda x: t-mydel <= xarr[x] <= t+mydel and yijit[x] >= 0.5, range(len(xarr)))
        if len(rel) > 0:
            xpdf.append(t)
            ypdf.append(len(drel)*1.0/len(rel))
    plt.plot(xpdf,ypdf,'k-')

    # plt.scatter(di_xva,di_jit,color='blue',s=3)
    # plt.scatter(ri_xva,ri_jit,color='red',s=3)
    plt.scatter(xarr,yijit,color='red',s=3)
    plt.savefig('/home/gswarrin/research/gerrymander/logistic')
    plt.close()
    return result.params
    
#######################################################################################################
def new_fit_logistic(legyrs,trprop,use_incum=True,require_incum=False,avoid_incum=False):
    """ read in who won the district versus mccain share of vote
        fit logistic model using Jacobson data
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
      
    # get a random training set of a particular size
    S = random_combination(range(len(statelist)),int(trprop*len(statelist)))
    print trprop,len(statelist)
    trainlist = [statelist[x] for x in S]
    print "asfd: ",len(trainlist)
    # trainlist = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
    #    'LA','ME','MD','MA','MI','MN','MS','MO']
    # trainlist = statelist[:15]

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

    # for keeping track of who won for a given vote share
    yijit = []

    for line in f:
        cnt += 1 
        if cnt == 1:
            hdrs = line.strip().split(',')
            continue
        l = line.strip().split(',')
        if statelist[int(l[1][:-2])] not in trainlist:
            continue
        #  3 is who won (0=Rep; 1=Dem)
        # 12 dem's share of presidential vote - uses most recent election
        # 2 is incumbency (0=Rep; 1=Dem)
        # let's make incumbency positive for republicans
        incum = 2*int(l[2]) - 1 # so it's symmetric
        # int(l[2]) > 1 and 
        incum_bool = (not require_incum or int(l[2]) <= 1) and (not avoid_incum or int(l[2]) > 1)
        if int(l[0]) in legyrs and l[12] != '' and l[2] != '' and l[3] != '' and \
           int(l[3]) <= 1 and incum_bool:
            # dem's share of major-party presidential vote in most recent election
            xarr.append(float(l[12])/100) 
            # jittered values for who won
            yijit.append(int(l[3]) + np.random.rand(1,1)[0][0]/10-0.05)        # who won

            #############################################################
            # first column is the value we're trying to predict (who won)
            # second column is dem vote share
            # third column is incumbency
            # last column is for intercept (?)
            if use_incum:
                zarr.append([int(l[3]),xarr[-1],incum,1])
            else:
                zarr.append([int(l[3]),xarr[-1],1])

            # for keeping track of possible combinations of incumbents and who won
            # ignores cases in which neither is an incumbent
            if int(l[2]) <= 1:
                sq[int(l[3])][int(l[2])] += 1
            tot += 1

    # make the model
    npz = np.array(zarr)
    logit = sm.Logit(npz[:,0],npz[:,1:],full_output=False)

    # fit the model
    try:
        result = logit.fit()
    except:
        print "Perfect separation"
        return 0,0
    # print result.params
    # print result.summary()

    return tot,result.params

def get_expd(yr,betav,betai,betac):
    """ read in who won the district versus mccain share of vote
    """
    statelist = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
      
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    expd = 0.0
    cnt = 0
    tot = 0
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
        if int(l[0]) == yr and l[12] != '' and int(l[3]) <= 1:
            tot += 1
            xvote = float(l[12])/100
            if int(l[2]) <= 1:
                xincum = 2*int(l[2]) - 1
                # expd += 1/(1+np.exp(-(betav*xvote + betai*xincum + betac)))
            else:
                xincum = 0
                # expd += 1/(1+np.exp(-(18.816*xvote - 9.335)))
            expd += 1/(1+np.exp(-(betav*xvote + betai*xincum + betac)))
    f.close()
    # print yr,tot," expected: ",expd
    return expd

def new_get_expd(yr,betac,betav,betaid,betair):
    """ read in who won the district versus mccain share of vote
    """
    statelist = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
      
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    expd = 0.0
    cnt = 0
    tot = 0
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
        if int(l[0]) == yr and l[12] != '' and int(l[3]) <= 1:
            tot += 1
            xvote = float(l[12])/100
            dincum = 0
            rincum = 0
            if int(l[2]) == 1:
                dincum = 1
            if int(l[2]) == 0:
                rincum = 1
            if int(l[2]) <= 1:
                xincum = 2*int(l[2]) - 1
                # expd += 1/(1+np.exp(-(betav*xvote + betai*xincum + betac)))
            else:
                xincum = 0
                # expd += 1/(1+np.exp(-(18.816*xvote - 9.335)))
            expd += 1/(1+np.exp(-(betac + betav*xvote + betaid*dincum + betair*rincum)))
    f.close()
    # print yr,tot," expected: ",expd
    return expd

def simul_state(fn,betav,betac):
    """ uses Chen & Cottrell simulation data to estimate number of seats
    does not assume anything about incumbency
    only has data for 42 states (not OR or ones with only one district)
    """
    f = open('/home/gswarrin/research/gerrymander/' + fn,'r')
    expd = []
    for line in f:
        l = line.rstrip().split('\t')
        l2 = map(lambda x: float(x), l[1:]) # skip simulation number
        tmp = 0
        for j in range(len(l2)):
            tmp += 1/(1+np.exp(-(betav*(1-l2[j]) + betac))) # l2 appears to have republican percentage of vote
        expd.append(tmp)

    # get expected number of seats in the state from actual district plan
    actual_plan = get_state_actual_plan(2012,fn[:2],betav,0,betac)

    print "%s: % .3f % .3f .. % .3f" % (fn[:2],np.median(expd),actual_plan,np.median(expd)-actual_plan)
    return np.mean(expd),actual_plan

##################################################################################################
##################################################################################################

def total_simul(yr,betav,betai,betac):
    """ read in who won the district versus mccain share of vote
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    nonolist = ['AK','DE','MT','ND','SD','VT','WY','OR']
    expd = 0
    actu = 0
    for st in statelist[1:]:
        if st not in nonolist:
            simulval,actualval = simul_state(st + 'simul.txt',betav,betac)
            expd += simulval
            actu += actualval
            # print "total expected: ",expd
    return expd,actu

def get_state_actual_plan(yr,st,betav,betai,betac):
    """ read in who won the district versus mccain share of vote
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
      
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    expd = 0.0
    cnt = 0
    tot = 0
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
        if int(l[0]) == yr and l[12] != '' and int(l[3]) <= 1 and \
           int(l[1][:-2]) == statelist.index(st):
            tot += 1
            xvote = float(l[12])/100
            if int(l[2]) <= 1:
                xincum = 2*int(l[2]) - 1
            else:
                xincum = 0
            expd += 1/(1+np.exp(-(betav*xvote + betai*xincum + betac)))
    f.close()
    # print yr,tot," expected: ",expd
    return expd

def test_log():
    """
    """
    pf = [0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50]
    hr = [0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]
    co = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    npz = np.array([hr,pf,co]).transpose()
    logit = sm.Logit(npz[:,0],npz[:,1:])
    
    print sm.Logit.__doc__
    print sm.tools.add_constant.__doc__
    result = logit.fit()
    print result.params
    print result.summary()

##################################################################################################
##################################################################################################
# from simulation data
# - compute declination for each state
# from jacobson data
# - compute declination for each state from actual congressional votes
# - **simul_declination: compute declination for each state from presidential vote
# compute # of extra seats for a given state
# compute # of extra seats over all states

def compare_dec_all(yr):
    """
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
    
    nonolist = ['AK','DE','MT','ND','SD','VT','WY','OR']
    tot = 0
    totjac = 0
    elecst = actual_declination_from_elections(yr) # uses congressional races

    simarr = []
    jacarr = []
    for st in statelist[1:]:
        if st not in nonolist:
            act,Ndists = actual_declination(yr,st)
            sim = simul_declination(st) # uses presidential data
            jacact = elecst[st]
            simarr.append(sim)
            jacarr.append(jacact)
            if abs(sim) < 2 and abs(act) < 2 and abs(jacact) < 2:
                tot += ((act-sim)*Ndists/2)
                totjac += ((jacact-sim)*Ndists/2)
                # print "%s % .3f % .3f % .3f (%.3f,%.3f,%d)" % (st,act,elecst[st],sim,((act-sim)*Ndists/2),((jacact-sim)*Ndists/2),Ndists)
    print "tot: %.2f %.2f" % (tot,totjac)
    print np.mean(simarr),np.mean(jacarr)

def actual_declination(yr,st):
    """ read in who won the district versus mccain share of vote
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
      
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    expd = 0.0
    cnt = 0
    tot = 0
    vals = []
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
        if int(l[0]) == yr and l[12] != '' and int(l[3]) <= 1 and \
           int(l[1][:-2]) == statelist.index(st):
            tot += 1
            xvote = float(l[12])/100
            # if int(l[2]) <= 1:
            #     xincum = 2*int(l[2]) - 1
            # else:
            #     xincum = 0
            # expd += 1/(1+np.exp(-(betav*xvote + betai*xincum + betac)))
            vals.append(xvote)
    f.close()
    # print yr,tot," expected: ",expd
    return get_declination('',vals),len(vals)

def actual_declination_from_elections(yr):
    """ read in who won the district versus mccain share of vote
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
      
    yrs,states,cycstates,elecs = read_elections('elec-data.csv')
    d = dict()
    for elec in elecs.values():
        if int(elec.yr) == yr and elec.chamber == '11':
            d[elec.state] = get_declination('',elec.demfrac)
    return d        

def simul_declination(st):
    """ uses Chen & Cottrell simulation data to estimate number of seats
    does not assume anything about incumbency
    only has data for 42 states (not OR or ones with only one district)
    """
    f = open('/home/gswarrin/research/gerrymander/data/ChenCottrell/' + st + 'simul.txt','r')
    expd = []
    for line in f:
        l = line.rstrip().split('\t')
        l2 = map(lambda x: (1-float(x)), l[1:])
        tmp = 0
        tmp = get_declination('',l2)
        # print "%.3f" % (tmp)
        expd.append(tmp)

    # print "%s: % .3f" % (fn[:2],np.median(expd))

    # WARNING: Ignoring these skews answer.
    # If random districting has one side winning everything, then shouldn't just ignore those outcomes.
    # if len(expd) > 0:
    return np.median(expd) # may be equal to \pm 2
    # else:
    #    return 0

def simul_logistic(st,betac,betav):
    """ uses Chen & Cottrell simulation data to estimate number of seats
    05.23.17
    use my logistic fit with incumbency but set incumbency to 0.
    """
    f = open('/home/gswarrin/research/gerrymander/data/ChenCottrell/' + st + 'simul.txt','r')
    expd = []
    for line in f:
        l = line.rstrip().split('\t')
        l2 = map(lambda x: (1-float(x)), l[1:])
        tmp = 0
        for j in range(len(l2)):
            tmp += 1/(1+np.exp(-(betac + l2[j]*betav)))
        expd.append(tmp)

        print l2[:5]
        print st,expd[:3]
    return expd

    # print "%s: % .3f" % (fn[:2],np.median(expd))

    # WARNING: Ignoring these skews answer.
    # If random districting has one side winning everything, then shouldn't just ignore those outcomes.
    # if len(expd) > 0:
    return np.median(expd) # may be equal to \pm 2
    # else:
    #    return 0

def total_log_simul(yr,betac,betav):
    """ read in who won the district versus mccain share of vote
    05.23.17
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    nonolist = ['AK','DE','MT','ND','SD','VT','WY','OR']
    expd = [[] for j in range(len(statelist[1:]))]
    actu = 0
    for i,st in enumerate(statelist[1:]):
        if st not in nonolist:
            simulval = simul_logistic(st,betac,betav)
            expd[i] = simulval
            # print "total expected: ",expd
            # print "%s %.3f" % (st,simulval)

    fig = plt.figure(figsize=(8,8))    
    for i in range(len(statelist[1:])):
        if len(expd[i]) > 0:
            plt.scatter(expd[i],[i*0.2 for j in range(len(expd[i]))],s=1)
    plt.savefig('/home/gswarrin/research/gerrymander/simul-scatter')
    plt.close()
    return expd

def total_dec_simul(yr):
    """ read in who won the district versus mccain share of vote
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    nonolist = ['AK','DE','MT','ND','SD','VT','WY','OR']
    expd = 0
    actu = 0
    for st in statelist[1:]:
        if st not in nonolist:
            simulval = simul_declination(st + 'simul.txt')
            expd += simulval
            # print "total expected: ",expd
            print "%s %.3f" % (st,simulval)
    return expd

def count_all_dec_seats(yr):
    """ compare declination in simulations to real life to determine number of seats
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    nonolist = ['AK','DE','MT','ND','SD','VT','WY','OR']
    expd = 0
    actu = 0
    for st in statelist[1:]:
        if st not in nonolist:
            simulval = count_st_dec_seats(st + 'simul.txt')
            expd += simulval
            # print "total expected: ",expd
            print "%s %.3f" % (st,simulval)
    return expd

# simul_state('FLplans27.txt',12.1167,-6.1843)

# using my numbers
# print total_simul(2012,14.3459,0,-7.2057)
# print total_simul(2012,12.1167,0,-6.1943)
#
# using their fit
# print total_simul(2012,21.8358,0,-11.0607) # coeffs from 2006,8,10,12
# print total_simul(2012,19.578,0,-9.8374)

# get_expd_state(2012,'FL',12.1167,0,-6.1943)

###########################################
# looks at Jacobson congressional data
# uses dem vote fraction to estimate odds of dems winning district
# second parameter comes from fitting logistic regression to incumbency as well
# dems won 201 seats in actuality
#
# get_expd(2012,14.3459,1.8595,-7.2057) # data from 2006,8,10,12; fit using incumbency
# get_expd(2012,14.3459,0,-7.2057)      # data from 2006,8,10,12; fit using incumbency, but ignore that info
# get_expd(2012,21.8358,0,-11.0607)     # data from 2006,8,10,12; fit without incumbency
# 
# get_expd(2004,18.71,0,-9.7575) # data from 2006,8,10,12; fit using incumbency
# get_expd(2006,18.71,0,-9.7575) # data from 2006,8,10,12; fit using incumbency
# get_expd(2008,18.71,0,-9.7575) # data from 2006,8,10,12; fit using incumbency
# get_expd(2010,18.71,0,-9.7575) # data from 2006,8,10,12; fit using incumbency
# get_expd(2012,18.71,0,-9.7575) # data from 2006,8,10,12; fit using incumbency
# get_expd(2012,21.84,0,-11.0765) # data from 2006,8,10,12; fit using incumbency
# get_expd(2012,20.785,0,-11.57) # data from 2006,8,10,12; fit using incumbency
# get_expd(2012,22.89,0,-10.5) # data from 2006,8,10,12; fit using incumbency
#
# Results: 435 seats, dem expected to get (in each case):
# 198.4 - close to actual value of 201
# 223.6 - implies would have won much more without incumbency effects
# 215.4 - pretty far from actual, doesn't seem all that trustworthy
#
# other years
# get_expd(2004,24.577,4.29,-11.46) # 203 predicted, got 202 (only 434 seats)
# get_expd(2006,20.2874,3.4513,-8.2506) # 223 predicted, got 233
# get_expd(2008,12.3402,2.6657,-5.5107) # 259 predicted, got 257
# get_expd(2008,12.3402,0,-5.5107) # 292 predicted!!!, got 257
# get_expd(2010,14.3459,1.8595,-7.2057) # 250 predicted, got 193
# get_expd(2012,40.2833,2.8144,-18.9876) # 207 predicted, got 201
# get_expd(2012,40.2833,0,-18.9876) # 244 predicted, got 201 - predicts lots more without incumbency effects
###########################################

###########################################
# find parameters for logistic regression
# True=fit using incumbency
# 
# yrs = [2006,2008,2010,2012] 
# fit_logistic(yrs,False)     # 14.3459, 1.8544, -7.2057
# fit_logistic(yrs,False)    # 21.8358,      0, -11.0607
#
# yrs = [2004,2006] 
# fit_logistic(yrs,True)     # 20.2874, 3.4513, -8.2506
# fit_logistic(yrs,False)    # 23.2344,      0, -11.0953
#
# yrs = [2004]
# fit_logistic(yrs,True)     # 24.577, 4.29, -11.46
# 
# yrs = [2008]
# print fit_logistic(yrs,True)     # 12.3402, 2.6657, -5.5107; require there be an incumbent
#
# yrs = [1990,1992,1994,1996,1998,2000,2002,2004,2006,2008,2010,2012]
# yrs = [2006,2008,2010,2012]
# fit_logistic(yrs,False)     # 40.2833, 2.8144, -18.9876

from numpy import mean, sqrt, square, arange
# a = arange(10) # For example
# rms = sqrt(mean(square(a)))

def check_estimate(yrs,trprop,use_incum,require_incum,avoid_incum):
    """ see how good the estimates of number of seats are under various scenarios
    """
    diffs = []
    alltot = 0
    for yr in yrs:
        tot,params = new_fit_logistic([yr],trprop,use_incum,require_incum,avoid_incum)
        if type(params) == int:
            continue
        # print params
        alltot += tot
        if use_incum:
            expd = get_expd(yr,params[0],params[1],params[2])
        else:
            expd = get_expd(yr,params[0],0,params[1])
        diffs.append(expd-DR_seats[yr][0])
        print "Yr: %d, %s: Races: %d Expected: %3.3f Actual: %d Diff: %3.1f" % \
        (yr,params,tot,expd,DR_seats[yr][0],expd-DR_seats[yr][0])
    print "All: %4d Use: %s Require: %s Avoid: %s Error: %.3f" % \
        (alltot,use_incum,require_incum,avoid_incum,sqrt(mean(square(np.array(diffs)))))
    return sqrt(mean(square(np.array(diffs))))

def find_training_set_size():
    fig = plt.figure(figsize=(8,8))
    allyrs = [1976,1980,1984,1988,1992,1996,2004,2008,2012]
    for ui in [True,False]:
        for ri in [True,False]:
            for ai in [True,False]:
                if not ri or not ai:
                    tmparr = []
                    ls = np.linspace(0.05,0.95,19)
                    for trprop in ls:
                        try:
                            ce = check_estimate(allyrs,trprop,ui,ri,ai)
                        except:
                            ce = 0
                        tmparr.append(ce)
                    plt.plot(ls,tmparr)
    plt.savefig('training-size')
    plt.close()

def make_guesses(yrs):
    diffs = []
    for yr in yrs:
        expd = new_get_expd(yr,-8.16,15.76,2.90,-2.24)
        diffs.append(expd-DR_seats[yr][0])
        print "Yr: %d Expected: %3.3f Actual: %d Diff: %3.1f" % \
            (yr,expd,DR_seats[yr][0],expd-DR_seats[yr][0])
    print sqrt(mean(square(np.array(diffs))))

# make_guesses([1976])  # ,1980,1984,1988,1992,1996,2004,2008,2012])

# check_estimate(allyrs,False,False,False)
# [1976]: # [1976,1980,1984,1988,1992,1996,2004,2008,2012]:

# test_log()
# simul_declination('CAsimul.txt')
# total_dec_simul(2012)
# compare_dec_all(1992)
# compare_dec_all(1996)
# compare_dec_all(2004)
# compare_dec_all(2008)
# compare_dec_all(2012)

# print get_expd(2012,17.27,0,-9.09)
total_log_simul(2012,-3.86,7.96)

