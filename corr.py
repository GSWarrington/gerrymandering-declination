###################################################################
# Look at correlation function
###################################################################

# run through elections
# compute 0gap, inftygap and alphagap for a range of values
# - so have a big 2darray with rows corresponding to alpha and cols to elections
# - compute correlation coefficient for pairs of rows and plot
# alpha=-2 seems like a pretty good happy medium - depends on data set

def min_correlation(elecs):
    """ see above description
    """
    alphavals = [0,0.5,0.75,1,1.25,1.5,1.75,2,3,4,5,1000] # np.linspace(0,10,21)
    zidx = 0
    while abs(alphavals[zidx]) > 0.25:
        zidx += 1
    # zidx = 1
    ans1 = []
    ans2 = []
    print "zidx",zidx,alphavals[zidx]
    arr = [[] for i in range(len(alphavals))]
    for elec in elecs.values():
        if 2010 >= int(elec.yr) >= 1972 and elec.chamber=='11' and \
        (elec.yr not in Mmmd.keys() or elec.state not in Mmmd[elec.yr]): # and \
            # elec.state in ['NC','PA','FL']: # and elec.Ndists >= 8:
            tmp = get_tau_gap(elec.demfrac,alphavals[zidx])
            if 1 == 1: # abs(tmp) < 0.05:
                for i in range(len(alphavals)):
                    print elec.state,elec.yr
                # tmp = 
                    arr[i].append(get_tau_gap(elec.demfrac,alphavals[i]))
                # print i,alphavals[i],arr[i][-1]
    labs = []
    for i in range(len(alphavals)):
        a1,pv1 = stats.pearsonr(arr[0],arr[i]) # correlation with tau=-10 (total seats)
        a2,pv2 = stats.pearsonr(arr[i],arr[-1]) # correlation with tau=10
        ans1.append(a1)
        ans2.append(a2)
        labs.append(alphavals[i])
        make_scatter('11corrscatinfty' + str(i),arr[0],arr[-1])
        make_scatter('11corrscat0' + str(i),arr[i],arr[zidx])
    make_scatter('11corrends' + str(i),arr[0],arr[-1])
    make_scatter_labels('corr-infty.png',ans1,ans2,labs)

# min_correlation(electionsa)    
