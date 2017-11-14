# how much does dec change for a single pack/crack based on k and k'
# as iteratively continue packing/cracking what does curve of declination look like
# 

import scipy.odr as odr
def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]

# global constants 
# these are from Buzas's high breakdown regression
# gamma0 = 0.09
# gamma1 = 0.83
# 
multfact = 5.0/12
#
maxpct = 0.45

##########################################################################################
def seat_shifts(elecs):
    """ estimate the number of seats that changed hands due to gerrymandering
    """
    for yr in [1972+2*j for j in range(23)]:
        Sdec = 0
        for elec in elecs.values():
            if int(elec.yr) == yr and elec.chamber == '11' and elec.Ndists >= 1: # and \
            # len(filter(lambda x: x <= 0.5, elec.demfrac)) >= 0.2*elec.Ndists and \
            # len(filter(lambda x: x <= 0.5, elec.demfrac)) <= 0.8*elec.Ndists:
                dec = find_angle('',elec.demfrac)
                # if elec.Ndists < 6 and abs(round(dec*elec.Ndists*(3.0/8))) >= 1 and abs(dec) < 2:
                #     print yr,elec.state,dec*elec.Ndists*(3.0/8)
                if abs(dec) < 2:
                    Sdec += round(dec*multfact*elec.Ndists)
        print "%d % .0f" % (yr,Sdec)

##########################################################################################
def distribute_votes(arr,votes,stidx,enidx,maxval,verbose=False):
    """ evenly distribute as many of the votes as possible among the districts
    stidx,stidx+1,...,enidx-1
    returns new array along with amount not distributed
    """
    narr = sorted([x for x in arr])
    amtper = votes*1.0/(enidx-stidx)
    allfit = True
    notfit = 0.0
    if verbose:
        print "In dist: ",votes,stidx,enidx,maxval
    for j in range(stidx,enidx):
        if narr[j]+amtper < maxval:
            narr[j] += amtper
        else:
            allfit = False
            notfit = amtper - (maxval-narr[j])
            narr[j] = maxval
    return allfit,notfit,sorted(narr)

###############################################################################
def even_seats_pack_or_crack(arr,crack=True,verbose=False):
    """ crack or pack if possible
    """
    N = len(arr)
    delx = 1.0/(2*N)
    xvals = np.linspace(delx,1-delx,N)

    # figure out which district is being modified
    narr = sorted([x for x in arr])
    idx = 0
    while idx < N and narr[idx] <= 0.5:
        idx += 1
    if idx == 0 or idx == N:
        if verbose:
            print "One side one everything. Failing"
        return False,narr
    
    # don't allow ones that are too lopsided
    # if ((N-idx)*1.0/N < 0.45 or (N-idx)*1.0/N > 0.55):
    #     return False,narr
    if (crack and (idx < 3 or N-idx-1 < 1)) or ((not crack) and (N-idx-1 < 3 or idx < 1)):
        return False,narr

    # set up parameters for filling things in
    if crack:
        maxval = maxpct
        stidx = 0
        enidx = idx
    else: # pack
        maxval = 1.0
        stidx = idx+1
        enidx = N

    if stidx == enidx:
        if verbose:
            print narr,crack,stidx
        return False,narr

    lr = stats.linregress(xvals[stidx:enidx],narr[stidx:enidx])
    # how much room we have for the votes we're trying to crack
    if crack:
        # room = idx*maxval - sum(narr[stidx:enidx])
        room = idx*maxval - sum([min(maxpct,narr[i]) for i in range(stidx,enidx)])
    else:
        room = (N-idx-1)*maxval - sum(narr[stidx:enidx])
    # new value for district we're cracking
    nval = min(maxpct,lr[1] + lr[0]*delx*(2*idx + 1))
    # amount we're changing that one district
    diff = narr[idx]-nval
    # see if we have enough room to crack the votes
    # could try to not reduce quite as much (ignore linear regression)
    if room < diff:
        nval = maxpct
        diff = narr[idx]-nval
        if room < diff:
            if verbose:
                print "Not enough room to crack votes; returning original"
            return False,arr

    # iteratively move the votes
    narr[idx] = nval
    allfit = False
    if not crack:
        enidx = N-1
        while enidx > stidx+1 and narr[enidx] == maxval:
            enidx -= 1
    else:
        enidx = idx-1
        while enidx > 2 and narr[enidx] == maxval:
            enidx -= 1
    if stidx == enidx:
        return False,arr
    while not allfit:
        if stidx == enidx:
            return False,arr
        allfit,notfit,parr = distribute_votes(narr,diff,stidx,enidx,maxval,verbose)
        narr = parr
        diff = notfit
        if not allfit:
            while narr[enidx-1] == maxval:
                enidx -= 1
        # print "blah: ",narr
    return True,narr

###############################################################################
def seats_pack_or_crack(arr,crack=True,verbose=False):
    """ crack or pack if possible. shove as far away from middle as possible.
    """
    N = len(arr)
    delx = 1.0/(2*N)
    xvals = np.linspace(delx,1-delx,N)

    # figure out which district is being modified
    narr = sorted([x for x in arr])
    idx = 0
    while idx < N and narr[idx] <= 0.5:
        idx += 1
    if idx == 0 or idx == N:
        if verbose:
            print "One side one everything. Failing"
        return False,narr
    
    # don't allow ones that are too lopsided
    # if ((N-idx)*1.0/N < 0.45 or (N-idx)*1.0/N > 0.55):
    #     return False,narr
    if (crack and (idx < 3 or N-idx-1 < 1)) or ((not crack) and (N-idx-1 < 3 or idx < 1)):
        return False,narr

    # set up parameters for filling things in
    if crack:
        maxval = maxpct
        stidx = 0
        enidx = idx
    else: # pack
        maxval = 1.0
        stidx = idx+1
        enidx = N

    if stidx == enidx:
        if verbose:
            print narr,crack,stidx
        return False,narr

    lr = stats.linregress(xvals[stidx:enidx],narr[stidx:enidx])
    # how much room we have for the votes we're trying to crack
    if crack:
        # room = idx*maxval - sum(narr[stidx:enidx])
        room = idx*maxval - sum([min(maxpct,narr[i]) for i in range(stidx,enidx)])
    else:
        room = (N-idx-1)*maxval - sum(narr[stidx:enidx])
    # new value for district we're cracking
    nval = min(maxpct,lr[1] + lr[0]*delx*(2*idx + 1))
    # amount we're changing that one district
    diff = narr[idx]-nval
    # see if we have enough room to crack the votes
    # could try to not reduce quite as much (ignore linear regression)
    if room < diff:
        nval = maxpct
        diff = narr[idx]-nval
        if room < diff:
            if verbose:
                print "Not enough room to crack votes; returning original"
            return False,arr

    narr[idx] = nval
    # if idx == N-1 and pack:
    #     return False,narr
    # if idx == 0 and crack:
    #     return False,narr
    while diff > 0.001:
        if crack:
            val, nidx = min((val, nidx) for (nidx, val) in enumerate(narr[:idx]))
        else:
            val, nidx = max((val, nidx) for (nidx, val) in enumerate(narr[idx+1:]))
            nidx += (idx+1)
        # print diff,val,nidx,narr
        amt = min(0.01,diff)
        narr[nidx] += amt
        diff -= amt
    return True,narr

###############################################################################
def pandc_four_options(st,yr,arr):
    """ try all four possibilities of packing/cracking for each party
    """
    ansdem = []
    ansrep = []
    N = len(arr)*1.0
    
    # seats_pack_or_crack by default does a pro-Rep gerrymander
    # so variable names just be read as which party *loses* the seat
    # I figure I'm more likely to make a mistake by renaming everything.

    carr = sorted([x for x in arr])
    # pro-Republican crack
    isokdc,narrdc = seats_pack_or_crack(carr,True)
    # pro-Republican crack
    isokdp,narrdp = seats_pack_or_crack(carr,False)

    tarr = sorted([1-x for x in arr])
    # pro-Democratic crack
    isokrc,narrrc = seats_pack_or_crack(tarr,True)
    # pro-Democratic crack
    isokrp,narrrp = seats_pack_or_crack(tarr,False)

    dec = find_angle('',arr)*multfact*N
    deci = find_angle('',tarr)*multfact*N
    if isokdc:
        ansdem.append([N,find_angle('',narrdc)*multfact*N-dec])
        if abs(ansdem[-1][1]) >= 2:
            print st,yr,"demc",ansdem[-1],narrdc
    if isokdp:
        ansdem.append([N,find_angle('',narrdp)*multfact*N-dec])
        if abs(ansdem[-1][1]) >= 2:
            print st,yr,"demp",ansdem[-1],narrdp
    if isokrc:
        ansrep.append([N,deci-find_angle('',narrrc)*multfact*N])
        if abs(ansrep[-1][1]) >= 2:
            print st,yr,"repc",ansrep[-1],narrrc
    if isokrp:
        ansrep.append([N,deci-find_angle('',narrrp)*multfact*N])
        if abs(ansrep[-1][1]) >= 2:
            print st,yr,"repp",ansrep[-1],narrrp
    return ansdem,ansrep

###############################################################################
def create_mpandc_pic(fn,elecs,mmd,chamber,verbose = False):
    """ run through all elections, try to pack and crack most competitive district each side wins in all possible ways. see effect on declination. I named things wrong. ansdem corresponds to dems *losing* a seat.

    """
    ansdem = []
    ansrep = []
    totposs = 0
    totok = 0
    for elec in elecs.values():
        if int(elec.yr) >= 1972 and elec.Ndists >= 5 and elec.chamber == chamber and \
           (int(elec.yr)%4 == 0) and \
           len(filter(lambda x: x > 0.5, elec.demfrac)) > 0 and \
           len(filter(lambda x: x > 0.5, elec.demfrac)) < elec.Ndists and \
           (elec.chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
            # print elec.state,elec.yr,elec.Ndists
            adem,arep = pandc_four_options(elec.state,elec.yr,elec.demfrac)
            totposs += 4
            totok += (len(adem)+len(arep))
            if adem != []:
                ansdem.extend(adem)
            if arep != []:
                ansrep.extend(arep)
            if verbose:
                print adem,arep

    print "numbers: ",totposs,totok

    for x in ansdem:
        if len(x) < 2:
            print x
    print "----------------"
    for x in ansrep:
        if len(x) < 2:
            print x
        
    lrd = stats.linregress([x[0] for x in ansdem],[x[1] for x in ansdem])
    print "red squares",lrd
    lrr = stats.linregress([x[0] for x in ansrep],[x[1] for x in ansrep])
    print "blue dots",lrr

    # find 95% CI's
    redlist = sorted([x[1] for x in ansdem])
    bluelist = sorted([x[1] for x in ansrep])
    print redlist[int(0.05*len(redlist))],redlist[int(0.95*len(redlist))]
    print bluelist[int(0.05*len(bluelist))],bluelist[int(0.95*len(bluelist))]

    print len(ansdem),len(ansrep)
    plt.figure(figsize=(8,4))
    tmpd = plt.scatter([x[0]-0.25 + np.random.uniform(-0.5,0.5) for x in ansdem],\
                       [x[1] for x in ansdem],color='red',marker="s",s=8)
    tmpr = plt.scatter([x[0]-0.25 + np.random.uniform(-0.5,0.5)  for x in ansrep],\
                       [x[1] for x in ansrep],color='blue',s=8)
    legs = []
    legs.append(tmpd)
    legs.append(tmpr)
    
    # plot linear regression lines
    xmax = 55
    plt.plot([0,xmax],[lrd[1],lrd[1] + lrd[0]*xmax],'black')
    plt.plot([0,xmax],[lrr[1],lrr[1] + lrr[0]*xmax],'black')

    # compute rmse
    predd = [lrd[1] + lrd[0]*x[0] for x in ansdem]
    # predd = [x[0] for x in ansdem]
    actud = [x[1] for x in ansdem]

    predr = [lrr[1] + lrr[0]*x[0] for x in ansrep]
    # predr = [x[0] for x in ansrep]
    actur = [x[1] for x in ansrep]

    print "rmse pro-r: ",math.sqrt(np.mean([(predd[i] - actud[i])**2 for i in range(len(predd))]))
    print "rmse pro-d: ",math.sqrt(np.mean([(predr[i] - actur[i])**2 for i in range(len(predr))]))

    # labels and formatting
    plt.gca().set_xlabel('Number of districts, $N$')
    plt.gca().set_ylabel('Change in $S$-declination')
    if chamber == '11':
        plt.gca().set_xlim(0,60)

    plt.legend((tmpd,tmpr),('Pro-Republican pack/crack',\
                            'Pro-Democratic pack/crack'),\
                            loc='upper right')
    plt.tight_layout()

    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn + chamber)
    plt.close()

###############################################################################
###############################################################################
# code for modeling linera regression of chen-cottrell paper

def seats_get_leg_vote(yrmin,yrmax):
    """ get legislative votes
    """
    # so we can try to match things up....
    statelist = ['b','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    ##################################################################################################
    # get districts for given year,state from Jacobson file (this is what we have for pres'l vote)
    ##################################################################################################
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    yrs = range(yrmin,yrmax+1,4)
    d = dict()
    for yr in yrs:
        d[yr] = dict()
    cnt = 0
    for line in f:
        cnt += 1 
        if cnt == 1:
            hdrs = line.strip().split(',')
            continue
        l = line.strip().split(',')

        # only keep certain years
        if int(l[0]) not in yrs:
            continue

        yr = int(l[0])
        dist = l[1][-2:]
        st = int(l[1][:-2])
        # print " asdf: ",l[3]
        if l[4] == '' or l[4] == ' ':
            continue
        lvote = float(l[4])/100
        pvote = float(l[12])/100
        # print st,yr
        
        # ky = '_'.join([statelist[st],dist])
        if statelist[st] in d[yr].keys():
            d[yr][statelist[st]].append([lvote,pvote])
        else:
            d[yr][statelist[st]] = [[lvote,pvote]]

    return d

def get_pl_vote(yrmin,yrmax):
    """ get legislative votes
    """
    # so we can try to match things up....
    statelist = ['b','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    ##################################################################################################
    # get districts for given year,state from Jacobson file (this is what we have for pres'l vote)
    ##################################################################################################
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    yrs = range(yrmin,yrmax+1,4)
    d = dict()
    for yr in yrs:
        d[yr] = dict()
        for st in statelist[1:]:
            d[yr][st] = dict()

    cnt = 0
    for line in f:
        cnt += 1 
        if cnt == 1:
            hdrs = line.strip().split(',')
            continue
        l = line.strip().split(',')

        # only keep certain years
        if int(l[0]) not in yrs:
            continue

        yr = int(l[0])
        dist = l[1][-2:]
        st = int(l[1][:-2])
        # print " asdf: ",l[3]
        if l[4] == '' or l[4] == ' ':
            continue
        lvote = float(l[4])/100
        pvote = float(l[12])/100
        # print st,yr
        
        # ky = '_'.join([statelist[st],dist])
        d[yr][statelist[st]][dist] = [lvote,pvote]

    return d

###############################################################################
def get_log_estimate(arr,yr,betac,betav,gamma0,gamma1):
    """
    """
    # convert back to presidential vote
    origp = map(lambda i: (gamma0[yr] + gamma1[yr]*arr[i]), range(len(arr)))
    # estimate number of seats using logistic regression
    return sum([1/(1+np.exp(-(betac[yr]+betav[yr]*t))) for t in origp])

###############################################################################
def seats_pandc_four_options(st,yr,arr,betac,betav,gamma0,gamma1,verbose=True):
    """ try all four possibilities of packing/cracking for each party
    """
    ansdem = []
    ansrep = []
    N = len(arr)*1.0
    
    # same issue as for declination simulation
    # variable labels indicate party being *hurt*

    carr = sorted([x for x in arr])
    # pro-Republican crack
    isokdc,narrdc = seats_pack_or_crack(carr,True)
    # pro-Republican crack
    isokdp,narrdp = seats_pack_or_crack(carr,False)

    tarr = sorted([1-x for x in arr])
    # pro-Democratic crack
    isokrc,narrrc = seats_pack_or_crack(tarr,True)
    # pro-Democratic crack
    isokrp,narrrp = seats_pack_or_crack(tarr,False)

    est_orig = get_log_estimate(carr,yr,betac,betav,gamma0,gamma1)
    est_origi = get_log_estimate(tarr,yr,betac,betav,gamma0,gamma1)

    if isokdc:
        ansdem.append([N,est_orig-get_log_estimate(narrdc,yr,betac,betav,gamma0,gamma1)])
        if abs(ansdem[-1][1]) >= 2 or verbose:
            print st,yr,"demc",ansdem[-1],narrdc
    if isokdp:
        ansdem.append([N,est_orig-get_log_estimate(narrdp,yr,betac,betav,gamma0,gamma1)])
        if abs(ansdem[-1][1]) >= 2 or verbose:
            print st,yr,"demp",ansdem[-1],narrdp
    if isokrc:
        ansrep.append([N,get_log_estimate(narrrc,yr,betac,betav,gamma0,gamma1)-est_origi])
        if abs(ansrep[-1][1]) >= 2 or verbose:
            print st,yr,"repc",ansrep[-1],narrrc
    if isokrp:
        ansrep.append([N,get_log_estimate(narrrp,yr,betac,betav,gamma0,gamma1)-est_origi])
        if abs(ansrep[-1][1]) >= 2 or verbose:
            print st,yr,"repp",ansrep[-1],narrrp
    return ansdem,ansrep

###############################################################################
def eg_pandc_four_options(st,yr,arr,betac,betav,verbose=True):
    """ try all four possibilities of packing/cracking for each party
    """
    ansdem = []
    ansrep = []
    N = len(arr)*1.0
    
    # same issue as for declination simulation
    # variable labels indicate party being *hurt*

    carr = sorted([x for x in arr])
    # pro-Republican crack
    isokdc,narrdc = seats_pack_or_crack(carr,True)
    # pro-Republican crack
    isokdp,narrdp = seats_pack_or_crack(carr,False)

    tarr = sorted([1-x for x in arr])
    # pro-Democratic crack
    isokrc,narrrc = seats_pack_or_crack(tarr,True)
    # pro-Democratic crack
    isokrp,narrrp = seats_pack_or_crack(tarr,False)

    est_orig = compute_alpha_curve(carr,0)*N
    est_origi = compute_alpha_curve(tarr,0)*N

    if isokdc:
        ansdem.append([N,-est_orig+compute_alpha_curve(narrdc,0)*N])
        if abs(ansdem[-1][1]) >= 2 or verbose:
            print st,yr,"demc",ansdem[-1],narrdc
    if isokdp:
        ansdem.append([N,-est_orig+compute_alpha_curve(narrdp,0)*N])
        if abs(ansdem[-1][1]) >= 2 or verbose:
            print st,yr,"demp",ansdem[-1],narrdp
    if isokrc:
        ansrep.append([N,-compute_alpha_curve(narrrc,0)*N+est_origi])
        if abs(ansrep[-1][1]) >= 2 or verbose:
            print st,yr,"repc",ansrep[-1],narrrc
    if isokrp:
        ansrep.append([N,-compute_alpha_curve(narrrp,0)*N+est_origi])
        if abs(ansrep[-1][1]) >= 2 or verbose:
            print st,yr,"repp",ansrep[-1],narrrp
    return ansdem,ansrep

###############################################################################
def eg_create_mpandc_pic(fn,betac,betav,verbose = False):
    """ run through all elections, try to pack and crack most competitive district 
    each side wins in all possible ways. see effect on declination
    """
    d = seats_get_leg_vote(min(betac.keys()),max(betac.keys()))

    ansdem = []
    ansrep = []
    for yr in d.keys():
        for st in d[yr].keys():
            vote = d[yr][st]
            # convert to presidential vote
            origl = sorted([x[0] for x in vote])
            origp = map(lambda i: (gamma0 + gamma1*origl[i]), range(len(vote)))
            if int(yr) >= 1972 and len(origl) >= 3 and \
               len(filter(lambda x: x > 0.5, origl)) > 0 and \
               len(filter(lambda x: x > 0.5, origl)) < len(vote):

                # from above, adem lists pro-repub

                adem,arep = eg_pandc_four_options(st,yr,origl,betac,betav,False)
                if adem != []:
                    ansdem.extend(adem)
                if arep != []:
                    ansrep.extend(arep)
                if verbose:
                    print adem,arep

    for x in ansdem:
        if len(x) < 2:
            print x
    print "----------------"
    for x in ansrep:
        if len(x) < 2:
            print x
        
    lrd = stats.linregress([x[0] for x in ansdem],[x[1] for x in ansdem])
    print "red squares",lrd
    lrr = stats.linregress([x[0] for x in ansrep],[x[1] for x in ansrep])
    print "blue dots",lrr

    # find 95% CI's
    redlist = sorted([x[1] for x in ansdem])
    bluelist = sorted([x[1] for x in ansrep])
    print redlist[int(0.05*len(redlist))],redlist[int(0.95*len(redlist))]
    print bluelist[int(0.05*len(bluelist))],bluelist[int(0.95*len(bluelist))]

    print len(ansdem),len(ansrep)
    plt.figure(figsize=(8,4))
    tmpd = plt.scatter([x[0] -0.25 + np.random.uniform(-0.5,0.5) for x in ansdem],\
                       [x[1] for x in ansdem],color='red',marker="s",s=8)
    tmpr = plt.scatter([x[0] -0.25 + np.random.uniform(-0.5,0.5) for x in ansrep],\
                       [x[1] for x in ansrep],color='blue',s=8)
    legs = []
    legs.append(tmpd)
    legs.append(tmpr)
    
    # plot linear regression lines
    xmax = 55
    plt.plot([0,xmax],[lrd[1],lrd[1] + lrd[0]*xmax],'black')
    plt.plot([0,xmax],[lrr[1],lrr[1] + lrr[0]*xmax],'black')

    # compute rmse
    predd = [lrd[1] + lrd[0]*x[0] for x in ansdem]
    actud = [x[1] for x in ansdem]

    predr = [lrr[1] + lrr[0]*x[0] for x in ansrep]
    actur = [x[1] for x in ansrep]

    print "rmse pro-r: ",math.sqrt(np.mean([(predd[i] - actud[i])**2 for i in range(len(predd))]))
    print "rmse pro-d: ",math.sqrt(np.mean([(predr[i] - actur[i])**2 for i in range(len(predr))]))

    # labels and formatting
    plt.gca().set_xlabel('Number of districts, $N$')
    plt.gca().set_ylabel('Change in estimate, $E(g(\ell^\circ)-E(g(\ell^*))$')
    plt.gca().set_xlim(0,60)

    plt.legend((tmpd,tmpr),('Pro-Republican pack/crack',\
                            'Pro-Democratic pack/crack'),\
                            loc='upper right')
    plt.tight_layout()

    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    plt.close()

    betac = {2008: -7.36, 2012: -7.6}
    # betac = {2008: -7.36, 2012: -7.6}
    # betac = {2008: -8.4, 2012: -8.4}
    betav = 15.79

###############################################################################
def cottrell_create_mpandc_pic(fn,betac,betav,gamma0,gamma1,verbose = False):
    """ run through all elections, try to pack and crack most competitive district 
    each side wins in all possible ways. see effect on declination
    """
    d = seats_get_leg_vote(min(betac.keys()),max(betac.keys()))

    ansdem = []
    ansrep = []
    ccc = []
    for yr in d.keys():
        for st in d[yr].keys():
            vote = d[yr][st]
            # convert to presidential vote
            origl = sorted([x[0] for x in vote])
            origp = map(lambda i: (gamma0[yr] + gamma1[yr]*origl[i]), range(len(vote)))
            if int(yr) >= 1972 and len(origl) >= 3 and \
               len(filter(lambda x: x > 0.5, origl)) > 0 and \
               len(filter(lambda x: x > 0.5, origl)) < len(vote):

                # from above, adem lists pro-repub

                adem,arep = seats_pandc_four_options(st,yr,origl,betac,betav,gamma0,gamma1,False)
                if adem != []:
                    ansdem.extend(adem)
                if arep != []:
                    ansrep.extend(arep)
                if verbose:
                    print adem,arep
                if yr == 2012:
                    for x in adem:
                        # print 'dem',st,"%.2f %.2f" % (x[0],x[1])
                        ccc.append(abs(x[1]))
                    for x in arep:
                        # print 'rep',st,"%.2f %.2f" % (x[0],x[1])
                        ccc.append(abs(x[1]))
    print "cccccc: ",np.mean(ccc),np.median(ccc)

    for x in ansdem:
        if len(x) < 2:
            print x
    print "----------------"
    for x in ansrep:
        if len(x) < 2:
            print x
        
    lrd = stats.linregress([x[0] for x in ansdem],[x[1] for x in ansdem])
    print "red squares",lrd
    lrr = stats.linregress([x[0] for x in ansrep],[x[1] for x in ansrep])
    print "blue dots",lrr

    # find 95% CI's
    redlist = sorted([x[1] for x in ansdem])
    bluelist = sorted([x[1] for x in ansrep])
    print redlist[int(0.05*len(redlist))],redlist[int(0.95*len(redlist))]
    print bluelist[int(0.05*len(bluelist))],bluelist[int(0.95*len(bluelist))]

    print len(ansdem),len(ansrep)
    plt.figure(figsize=(8,4))
    tmpd = plt.scatter([x[0] -0.25 + np.random.uniform(-0.5,0.5) for x in ansdem],\
                       [x[1] for x in ansdem],color='red',marker="s",s=8)
    tmpr = plt.scatter([x[0] -0.25 + np.random.uniform(-0.5,0.5) for x in ansrep],\
                       [x[1] for x in ansrep],color='blue',s=8)
    legs = []
    legs.append(tmpd)
    legs.append(tmpr)
    
    # plot linear regression lines
    xmax = 55
    plt.plot([0,xmax],[lrd[1],lrd[1] + lrd[0]*xmax],'black')
    plt.plot([0,xmax],[lrr[1],lrr[1] + lrr[0]*xmax],'black')

    # compute rmse
    predd = [lrd[1] + lrd[0]*x[0] for x in ansdem]
    actud = [x[1] for x in ansdem]

    predr = [lrr[1] + lrr[0]*x[0] for x in ansrep]
    actur = [x[1] for x in ansrep]

    print "rmse pro-r: ",math.sqrt(np.mean([(predd[i] - actud[i])**2 for i in range(len(predd))]))
    print "rmse pro-d: ",math.sqrt(np.mean([(predr[i] - actur[i])**2 for i in range(len(predr))]))

    # labels and formatting
    plt.gca().set_xlabel('Number of districts, $N$')
    plt.gca().set_ylabel('Change in estimate, $E(g(\ell^\circ))-E(g(\ell^*))$')
    plt.gca().set_xlim(0,60)

    plt.legend((tmpd,tmpr),('Pro-Republican pack/crack',\
                            'Pro-Democratic pack/crack'),\
                            loc='upper right')
    plt.tight_layout()

    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    plt.close()

    betac = {2008: -7.36, 2012: -7.6}
    # betac = {2008: -7.36, 2012: -7.6}
    # betac = {2008: -8.4, 2012: -8.4}
    betav = 15.79

def find_change(arr):
    """ plug into ugly formula to figure out what is going wrong
    """
    narr = sorted([x for x in arr])
    N = len(narr)
    repw = filter(lambda x: x <= 0.5, narr)
    demw = filter(lambda x: x >  0.5, narr)
    k = len(repw)*1.0
    l = len(demw)*1.0
    ybar = np.mean(repw)
    zbar = np.mean(demw)
    
    if k < 3 or l < 3:
        return False,0,0
    
    p = min(demw)
    theta = np.arctan2(1-2*ybar,k/N)
    gamma = np.arctan2(2*zbar-1,l/N)
    thetap = np.arctan((k/(k+1))*((1-2*ybar)/(k/N)) + (2.0*N/((k+1)*(k+1)))*(ybar-p))
    gammap = np.arctan((l/(l-1))*((2*zbar-1)/(l/N)) + (2.0*N/((l-1)*(l-1)))*(zbar-p))

    ans = (N/3.14)*((gammap-gamma)-(thetap-theta))
    return True,ans,p

    # diff = (N*N/(3.14))*(1/(l-1))*

def find_cubic(arr,islinea=False):
    """ plug into ugly formula to figure out what is going wrong
    """
    narr = sorted([x for x in arr])
    N = len(narr)
    repw = filter(lambda x: x <= 0.5, narr)
    demw = filter(lambda x: x >  0.5, narr)
    k = len(repw)*1.0
    l = len(demw)*1.0
    ybar = np.mean(repw)
    zbar = np.mean(demw)
    
    if k < 3 or l < 3:
        return False,0,0
    
    p = min(demw)
    if islinea:
        ta = (zbar-ybar)
    else:
        # print "%.2f %.2f" % ((zbar-p)*(4*zbar-2)*(4*zbar-2), (ybar-p)*(2-4*ybar)*(2-4*ybar))
        ta = (zbar-ybar) - (zbar-p)*(4*zbar-2)*(4*zbar-2) + (ybar-p)*(2-4*ybar)*(2-4*ybar) + \
             (zbar-p)*math.pow(4*zbar-2,4) - (ybar-p)*math.pow(2-4*ybar,4)
        
    ans = (8/3.14)*ta
    return True,ans,p

    # diff = (N*N/(3.14))*(1/(l-1))*

def find_diff(elecs):
    """
    """
    arr = []
    ar2 = []
    ar3 = []
    arh = []
    ari = []
    for elec in elecs.values():
        if int(elec.yr) >= 1972 and elec.Ndists >= 3 and elec.chamber == '11' and \
           elec.Ndists >= 10:
            zarr = filter(lambda x: x > 0.5, elec.demfrac)
            yarr = filter(lambda x: x <= 0.5, elec.demfrac)
            if len(zarr) > elec.Ndists/3 and len(yarr) > elec.Ndists/3:
                arr.append(np.mean(yarr))
                ar2.append(np.mean(zarr)) # 16.0*(np.mean(zarr)-np.mean(yarr))/3.14)
                ar3.append(8.0*(np.mean(zarr)-np.mean(yarr))/3.14)
                mybo,tmp_true,p = find_change(elec.demfrac)
                mybo,tmp,p = find_cubic(elec.demfrac,True)
                mybo2,tmp2,p2 = find_cubic(elec.demfrac)
                if mybo:
                    arh.append(tmp_true)
                    ari.append(tmp2)
    make_scatter('duhcmp',arh,ari)
    make_histogram('duhdiff',arh)
    make_histogram('duhdiff-8',ar3)

# cottrell_create_mpandc_pic({2012: -7.6}, 15.79)
    
#################################################################################

def pres_v_leg(fn,yrs,ax):
    """ plot leg vote versus presidential along with regression line
    """
    xarr = []
    yarr = []
    cnt = 0
    d = seats_get_leg_vote(min(yrs),max(yrs))
    for yr in yrs:
        for ys in d[yr].values():
            # print ys
            for i in range(len(ys)):
                xarr.append(float(ys[i][0]))
                yarr.append(float(ys[i][1]))
                cnt += 1

    print "Total districts: ",cnt

    # plt.figure(figsize=(6,6))
    ax.scatter(xarr,yarr,color='green')
    
    lr = stats.linregress(xarr,yarr)
    print "from pic: ",lr

    linear = odr.Model(f)
    mydata = odr.Data(xarr,yarr)
    myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
    myoutput = myodr.run()
    myoutput.pprint()
    print type(myoutput)
    # print stats.pearsonr([x[0] for x in stabe],[x[1] for x in stabe])

    print lr
    # OLS line
    # ax.plot([0,1],[0.12,0.77],color='black')
    # ODR line
    # From Jeff
    ax.plot([0,1],[lr[1],lr[0] + lr[1]],color='black')

    # labels and formatting
    ax.set_xlabel('Dem. legislative vote')
    ax.set_ylabel('Dem. presidential vote')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    return ax

def cottrell_plot_logit_curves(ax,pltcum=False):
    """ so we can compare different fits
    """
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
        ax.plot(np.linspace(xmin,xmax,xunits),duh,color=cols[i],linestyle='dashed')

    ryvals = [x[1] + 0.05 + np.random.rand(1,1)[0][0]/10-0.05 for x in rinc]
    dyvals = [x[1] - 0.05 + np.random.rand(1,1)[0][0]/10-0.05 for x in dinc]
    nyvals = [x[1] + np.random.rand(1,1)[0][0]/10-0.05 for x in ninc]

    ax.scatter([x[0] for x in rinc],[x for x in ryvals],color='red',s=2)
    ax.scatter([x[0] for x in dinc],[x for x in dyvals],color='blue',s=2)
    ax.scatter([x[0] for x in ninc],[x for x in nyvals],color='green',s=2)

    rvals = [1/(1+np.exp(-(beta0 + delta_R + delta_V*t))) for t in xvals]
    ax.plot(xvals,rvals,'r-')
    dvals = [1/(1+np.exp(-(beta0 + delta_D + delta_V*t))) for t in xvals]
    ax.plot(xvals,dvals,'b-')
    nvals = [1/(1+np.exp(-(beta0 + delta_V*t))) for t in xvals]
    ax.plot(xvals,nvals,'g-')

    # where does this come from
    beta0 = -10.93
    delta_V = 21.08
    ovals = [1/(1+np.exp(-(beta0 + delta_V*t))) for t in xvals]
    ax.plot(xvals,ovals,color='black')

    beta0 = -5.44
    delta_V = 11.31
    pvals = [1/(1+np.exp(-(beta0 + delta_V*t))) for t in xvals]
    ax.plot(xvals,pvals,color='grey')

    # print yvals[:10]
    # plt.plot(xvals,yvals,'b-')
    # plt.plot(xvals,zvals,'r-')
    ax.axhline(0.5)
    ax.axvline(0.5)
    ax.grid()
    ax.savefig('/home/gswarrin/research/gerrymander/logit-curves')

    plt.close()

def get_data_points(yrmin,yrmax,Rincum,Dincum):
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
        if int(l[0]) >= yrmin and int(l[0]) <= yrmax and l[12] != '' and l[3] != '' and \
           int(l[3]) <= 1:

            if (Rincum and int(l[2]) == 0) or \
               (Dincum and int(l[2]) == 1) or \
               (not Rincum and not Dincum and int(l[2]) > 1):
                ans.append([float(l[12])/100,int(l[3])])
    f.close()
    return ans

def get_linear_points(yrmin,yrmax,Rincum,Dincum):
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
        if int(l[0]) >= yrmin and int(l[0]) <= yrmax and l[12] != '' and l[4] != '' and \
           l[4] != ' ' and int(l[3]) <= 1:

            if (Rincum and int(l[2]) == 0) or \
               (Dincum and int(l[2]) == 1) or \
               (not Rincum and not Dincum and int(l[2]) > 1):
                ans.append([float(l[12])/100,float(l[4])/100])
    f.close()
    return ans

def seats_plot_logit_curves(ax,pltcum=False):
    """ so we can compare different fits
    """
    xvals = np.linspace(0,1,101)

    beta0 = -9.38
    # delta_D = 3.02
    # delta_R = -2.58
    delta_V = 18.87

    rinc = get_data_points(2008,2008,True,False)
    dinc = get_data_points(2008,2008,False,True)
    ninc = get_data_points(2008,2008,False,False)
    allinc = rinc + dinc + ninc
    print "No. pts",len(rinc)+len(dinc)+len(ninc),len(rinc),len(dinc),len(ninc)

    xmin = 0.3
    xmax = 0.7
    xunits = (xmax-xmin)*100+1
    dotty = []
    # for i,arr in enumerate([rinc,dinc,ninc,allinc]):
    #     tmp = []
    #     for cp in np.linspace(xmin,xmax,xunits):
    #         toleft = filter(lambda x: cp-0.05 <= x[0] <= cp+0.05, arr)
    #         numer = filter(lambda x: x[1] == 1, toleft)
    #        if len(toleft) > 0:
    #             tmp.append(len(numer)*1.0/len(toleft))
    #         else:
    #             tmp.append(0)
    #     dotty.append(tmp)
    # cols = ['red','blue','green','grey']
    # for i,duh in enumerate(dotty):
    #     ax.plot(np.linspace(xmin,xmax,xunits),duh,color=cols[i],linestyle='dashed')

    ryvals = [x[1] + np.random.rand(1,1)[0][0]/10-0.05 for x in rinc]
    dyvals = [x[1] + np.random.rand(1,1)[0][0]/10-0.05 for x in dinc]
    nyvals = [x[1] + np.random.rand(1,1)[0][0]/10-0.05 for x in ninc]

    ax.scatter([x[0] for x in rinc],[x for x in ryvals],color='green',s=2)
    ax.scatter([x[0] for x in dinc],[x for x in dyvals],color='green',s=2)
    ax.scatter([x[0] for x in ninc],[x for x in nyvals],color='green',s=2)

    # rvals = [1/(1+np.exp(-(beta0 + delta_R + delta_V*t))) for t in xvals]
    # ax.plot(xvals,rvals,'r-')
    # dvals = [1/(1+np.exp(-(beta0 + delta_D + delta_V*t))) for t in xvals]
    # ax.plot(xvals,dvals,'b-')
    # nvals = [1/(1+np.exp(-(beta0 + delta_V*t))) for t in xvals]
    # ax.plot(xvals,nvals,'g-')

    # from 2008, 2012 fit
    beta0 = -11.92
    delta_V = 23.58
    ovals = [1/(1+np.exp(-(beta0 + delta_V*t))) for t in xvals]
    ax.plot(xvals,ovals,color='black')

    # beta0 = -5.44
    # delta_V = 11.31
    # pvals = [1/(1+np.exp(-(beta0 + delta_V*t))) for t in xvals]
    # ax.plot(xvals,pvals,color='grey')

    # print yvals[:10]
    # plt.plot(xvals,yvals,'b-')
    # plt.plot(xvals,zvals,'r-')
    ax.axhline(0.5,ls='dotted',lw=1)
    ax.axvline(0.5,ls='dotted',lw=1)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    mylabels = ['Rep. seat','Dem. seat']
    ax.yaxis.set_ticks([0,1]) # ,labels=mylabels)
    ax.yaxis.set_ticklabels(mylabels)
    ax.set_xlabel('Democratic presidential vote fraction')
    # set_ylabel('')

    # ax.grid()
    # plt.savefig('/home/gswarrin/research/gerrymander/new-seats-curves')

def dbl_plot(fn,yrs):
    """ plot leg v pres data and logistic fits
    """
    fig = plt.figure(figsize=(6,2)) # plt.subplots(1,2,figsize=(8,4))

    # pres_v_leg(fn,yrs,axes[0])
    seats_plot_logit_curves(fig.gca())

    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    plt.close()

def new_look_at_linearity(ax,regline=False,yrmin=1972,yrmax=2012):
    """
    """
    rinc = get_linear_points(yrmin,yrmax,True,False)
    dinc = get_linear_points(yrmin,yrmax,False,True)
    ninc = get_linear_points(yrmin,yrmax,False,False)
    allinc = rinc+dinc+ninc

    alr = stats.linregress([x[1] for x in allinc],[x[0] for x in allinc])
    ax.scatter([x[1] for x in rinc],[x[0] for x in rinc],color='red',s=5)
    ax.scatter([x[1] for x in dinc],[x[0] for x in dinc],color='blue',s=5)
    ax.scatter([x[1] for x in ninc],[x[0] for x in ninc],color='lightgrey',s=5)

    if regline:
        ax.plot([0,1],[alr[1],alr[0]+alr[1]],color='black')
        print alr

    ax.set_xlabel('$\ell^\circ$')
    ax.set_ylabel('$p^\circ$')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
def plot_diff(yrmin,yrmax):
    """
    """
    statelist = ['b','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    xarr = []
    yarr = []
    d = get_pl_vote(yrmin,yrmax)
    for y in d.keys():
        for st in d[y].keys():
            v = d[y][st]
            newy = int(y+4)
            if newy in d.keys():
                if st in d[newy].keys():
                    newv = d[newy][st]
                    for dist in d[y][st].keys():
                        if dist in d[newy][st].keys():
                            xarr.append(d[newy][st][dist][0]-d[y][st][dist][0])
                            yarr.append(d[newy][st][dist][1]-d[y][st][dist][1])
                    
    print "total pts: ",len(xarr)
    make_scatter('darr',xarr,yarr)
    lr = stats.linregress(xarr,yarr)

    linear = odr.Model(f)
    mydata = odr.Data(xarr,yarr)
    myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
    myoutput = myodr.run()
    myoutput.pprint()
    print type(myoutput)

    print lr

def triple_plot(fn):
    """ plot leg v pres data and logistic fits
    """
    fig,axes = plt.subplots(1,3,figsize=(9,3))

    # p v l all years
    # p v l 2008
    # p v l 2012
    # pres_v_leg(fn,[1972 + 4*j for j in range(11)],axes[0])
    # pres_v_leg(fn,[2008],axes[1])
    # pres_v_leg(fn,[2012],axes[2])

    new_look_at_linearity(axes[0],True,1972,2012)
    new_look_at_linearity(axes[1],True,1980,1980)
    new_look_at_linearity(axes[2],True,2012,2012)

    axes[0].text(0.01,.94,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
    axes[1].text(0.33,.94,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')
    axes[2].text(0.66,.94,'C',fontsize=16, transform=fig.transFigure, fontweight='bold')


    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    plt.close()
    
def get_ptol_regress(d):
    """ so we can convert pres vote to leg vote
    """
    lrd = dict()
    for yr in d.keys():
        arr = []
        for st in d[yr].keys():
            vote = d[yr][st]
            if len(vote) < 5:
                continue
            # print yr,st,vote
            # lvote is in first index; pvote in second
            # convert to presidential vote
            arr.extend(vote)
            # origl = sorted([x[0] for x in vote])
            # origp = sorted([x[1] for x in vote])
        lr = stats.linregress([x[1] for x in arr],[x[0] for x in arr])
        print yr,lr[1],lr[0]
        lrd[yr] = [lr[1],lr[0]]
    return lrd

def compare_dec(gammaint,gammaslo):
    """ so we can compare different fits
    """
    fig = plt.figure(figsize=(6,6)) # plt.subplots(1,2,figsize=(8,4))

    yrmin = 1972
    yrmax = 2012
    d = seats_get_leg_vote(yrmin,yrmax)

    lrd = get_ptol_regress(d)

    decp = []
    decl = []
    for yr in d.keys():
        curyrp = 0
        curyrl = 0
        for st in d[yr].keys():
            # if yr not in [2000,'2000'] or st != 'FL':
            #     continue
            vote = d[yr][st]
            if len(vote) < 2:
                continue
            # print yr,st,vote
            # lvote is in first index; pvote in second
            # convert to presidential vote
            origl = sorted([x[0] for x in vote])
            origp = sorted([lrd[yr][0] + lrd[yr][1]*x[1] for x in vote])
            # lr = stats.linregress(origp,origl)
            # print lr

            # print origl,origp
            pans = find_angle('',origp)
            lans = find_angle('',origl)
            if abs(pans) < 2 and abs(lans) < 2:
                decp.append(pans*multfact*len(vote))
                decl.append(lans*multfact*len(vote))
            curyrp += round(pans*multfact*len(vote))
            curyrl += round(lans*multfact*len(vote))
        print yr," %.2f %.2f" % (curyrp,curyrl)
    print stats.pearsonr(decp,decl)
    plt.scatter(decp,decl)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/declp')
    plt.close()

def elec_get_ptol_regress(elecs,yrs):
    """ so we can convert pres vote to leg vote
    """
    lrd = dict()
    for yr in yrs:
        arrl = []
        arrp = []
        for elec in elecs.values():
            if elec.yr == yr and elec.chamber == '11':
                lvote = elec.demfrac
                pvote = elec.pvote
                if len(lvote) < 5:
                    continue
                # print yr,st,vote
                # lvote is in first index; pvote in second
                # convert to presidential vote
                arrl.extend(lvote)
                arrp.extend(pvote)
                # origl = sorted([x[0] for x in vote])
                # origp = sorted([x[1] for x in vote])
        lr = stats.linregress(arrp,arrl)
        print yr,lr[1],lr[0]
        lrd[yr] = [lr[1],lr[0]]
    return lrd

# in 2012, leg vote and leg vote regressed from pres vote yield very
# different answers for number of seats. so if I can put in simulation
# data instead should get something reasonable.
def compare_dec(elecs):
    """ so we can compare different fits
    """
    fig = plt.figure(figsize=(6,6)) # plt.subplots(1,2,figsize=(8,4))

    yrmin = 1972
    yrmax = 2012
    yrs = [str(yrmin+4*j) for j in range(11)]

    lrd = elec_get_ptol_regress(elecs,yrs)

    decp = []
    decl = []
    for yr in yrs:
        curyrp = 0
        curyrl = 0
        for elec in elecs.values():
            if elec.chamber == '11' and elec.yr == yr:
                lvote = elec.demfrac
                pvote = elec.pvote
                if len(lvote) < 2 or '' in pvote:
                    continue
                # print yr,st,vote
                # lvote is in first index; pvote in second
                # convert to presidential vote
                origl = sorted(lvote)
                origp = sorted([lrd[yr][0] + lrd[yr][1]*x for x in pvote])

                # print origl,origp
                lans = find_angle('',elec.demfrac) # origl)
                pans = find_angle('',origp)
                if abs(lans) < 2 and abs(lans) < 2:
                    decl.append(lans) # *multfact*len(lvote))
                    decp.append(pans) # *multfact*len(pvote))
                    curyrl += round(lans*multfact*elec.Ndists)
                    curyrp += round(pans*multfact*len(pvote))
        print yr," ... %.2f %.2f" % (curyrl,curyrp)
        # print yr," ... %.2f" % (curyrl)
    # print stats.pearsonr(decp,decl)
    # plt.scatter(decp,decl)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/declp')
    plt.close()

def cottrell_total_seats(betav,betac):
    """ try to figure out how many total seats under simulated plans
    """
    act = []
    sim = []
    for st in statelist:
        fn = '/home/gswarrin/research/gerrymander/data/ChenCottrell/' + st + 'simul.txt'
        if os.path.isfile(fn):



            f = open(fn,'r')
            l = cottrell_fig2('',st,betav,betac)
            act.append([st,l[0]])
            sim.append([st,l[1]])
    diff = [sim[i][1] - act[i][1] for i in range(len(act))]
    diffr = map(lambda x: floor(x) if x > 0 else floor(x)+1, diff)
    # simr = map(lambda x: floor(x) if x > 0 else floor(x)+1, [x[1] for x in sim])
    # actr = map(lambda x: floor(x) if x > 0 else floor(x)+1, [x[1] for x in act])
    print "sim expd: ",np.sum([x[1] for x in sim])
    print "act expd: ",np.sum([x[1] for x in act])
    # rounded version
    print "dif expd: ",np.sum(diffr)

    fn = st + 'simul.txt'
    f = open('/home/gswarrin/research/gerrymander/data/ChenCottrell/' + fn,'r')
    simul_votes = [[] for j in range(205)] # list of vote fractions for each of 200 simulations in the state
    expd_seats = [[] for j in range(205)]  # corresponding expected probability of electing democrat
    for i,line in enumerate(f):
        l = line.rstrip().split('\t')
        l2 = map(lambda x: float(x), l[1:]) # skip simulation number
        tmp = 0
        for j in range(len(l2)):
            simul_votes[i].append(1-l2[j])

        expd_seats[i] = map(lambda x: 1/(1+np.exp(-(betav*x + betac))), simul_votes[i])

        vals = sorted(simul_votes[i])
        evals = sorted(expd_seats[i])
        xvals = np.linspace(0+1.0/(2*len(vals)),1-1.0/(2*len(vals)),len(vals))
        xvals = map(lambda x: x + np.random.rand(1,1)[0][0]/50-0.01, xvals)
        if makepic:
            axes[0].scatter(xvals,vals,s=1,color='grey')
            axes[1].scatter(xvals,evals,s=1,color='grey')

    if makepic:
        axes[0].axhline(0.5,color='black')
        axes[1].axhline(0.5,color='black')

    # get expected number of seats in the state from actual district plan
    actual_plan = sorted(get_state_actual_probs(2012,fn[:2]))
    actual_prob = map(lambda x: 1/(1+np.exp(-(betav*x + betac))), actual_plan)
    
