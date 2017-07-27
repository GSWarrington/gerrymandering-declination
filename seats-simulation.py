###########################################
# code to manually pack and crack districts
###########################################

# find best fit line between pres vote and leg vote
# start with leg vote
# - convert to pres vote (add in random error) & compute expected
# - pack/crack -> convert to pres vote -> compute expected
# see how different the answers are.

# def check_logistic_number_seats():
#     """ run through elections, manually pack and/or crack, see effect on expected # seats
#     """

import os.path
import scipy.odr as odr
import numpy as np

def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]
linear = odr.Model(f)

def get_leg_vote(yrmin,yrmax):
    """ get legislative votes
    """
    # so we can try to match things up....
    statelist = ['b','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    ####################################################################################################
    # get districts for given year,state from Jacobson file (this is what we have for presidential vote)
    ####################################################################################################
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

def plot_seat_change(d,betav,betac,p):
    """ see how packing and cracking affect estimated number of seats
    """
    # fig,axes = plt.figure(figsize=(18,6))

    # add some random error back in 
    rerr = list(0.057 * np.random.randn(1000))
    rcp = list(np.random.uniform(0,1,1000))
    sarr = []
    est = []
    nest = []
    npack = 0

    allvote = []
    for yr in d.keys():
        for st in d[yr].keys():
            vote = d[yr][st]
            allvote.extend(vote)

            # convert to presidential vote
            origl = sorted([x[0] for x in vote])
            origp = map(lambda i: (0.13 + 0.8*origl[i])+rerr.pop(), range(len(vote)))

            # try to pack/crack
            rvote = sorted([x[1] for x in vote])
            npack += 1
            if len(filter(lambda x: x < 0.5,rvote)) >= 2 and \
               len(filter(lambda x: x > 0.5,rvote)) >= 2 and \
               rcp.pop() < p:
                isok,newl = pack_or_crack(origl,crack=(rcp.pop() < 1))
                if isok:
                    newp = map(lambda i: (0.13 + 0.8*newl[i])+rerr.pop(), range(len(newl)))
                    ea = [1/(1+np.exp(-(betac[yr]+betav*t))) for t in origp]
                    nea = [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newp]

                    # for i in range(len(origp)):
                    #     print origp[i],newp[i],ea[i],nea[i]
                        # print [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newy]
                    # print [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newz]
                    # get new estimate
                    delx = 1.0/(2*len(newp))
                    xvals = np.linspace(delx,1-delx,len(newp))
                    est.append(sum([1/(1+np.exp(-(betac[yr]+betav*t))) for t in origp]))
                    nest.append(sum([1/(1+np.exp(-(betac[yr]+betav*t))) for t in newp]))
                    sarr.append(len(filter(lambda x: x > 0.5, [x[0] for x in vote])) + np.random.randn(1)[0]*0.1)
                    # axes[0].scatter(xvals,sorted(origp),color='blue',s=20)
                    # axes[0].scatter(xvals,sorted(newp),color='purple',s=20)
                    # axes[1].scatter(xvals,sorted(ea),color='orange',s=20)
                    # axes[1].scatter(xvals,sorted(nea),color='red',s=20)
                    # axes[2].scatter(xvals,sorted(origl),color='green',s=20)
                    # axes[2].scatter(xvals,sorted(newl),color='brown',s=20)
                    # axes[0].set_title('pres vote')
                    # axes[1].set_title('prob seat')
                    # axes[2].set_title('legi vote')

    # print stats.pearsonr([x[0] for x in allvote],[x[1] for x in allvote])
    # print stats.linregress([x[0] for x in allvote],[x[1] for x in allvote])
    # plt.plot([0,1],[0.1,0.1+0.86],color='black')
    # newy = map(lambda i: allvote[i][1]-(0.13 + 0.8*allvote[i][0]), range(len(allvote)))
    # plt.scatter([x[0] for x in allvote],newy,color='green')
    # print "stddev: ",np.std(newy)
    # plt.plot([0,1],[0.13,0.13+0.8],color='black')

    # plt.scatter(sarr,est,s=5)
    # print "total estd: ",sum(est),len(allvote)
    # print "new estd: ",sum(nest),len(allvote)

    # this may be appropriate viewpoint, but I am treating leg vote as indep variable....
    # mydata = odr.Data([x[0] for x in allvote], [x[1] for x in allvote])
    # myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
    # myoutput = myodr.run()
    # myoutput.pprint()
    # print myoutput.sum_square

    # plt.savefig('/home/gswarrin/research/gerrymander/pack-move')
    # plt.close()
    return npack,sum(est),sum(nest)

def plot_seat_change_new(d,betav,betac):
    """ see how packing and cracking affect estimated number of seats
    new version analogous to declination version
    """
    # fig,axes = plt.figure(figsize=(18,6))

    # add some random error back in 
    rerr = list(0.057 * np.random.randn(1000))
    rcp = list(np.random.uniform(0,1,1000))
    sarr = []
    est = []
    nest = []
    npack = 0

    allvote = []
    for yr in d.keys():
        for st in d[yr].keys():
            vote = d[yr][st]
            allvote.extend(vote)

            # convert to presidential vote
            origl = sorted([x[0] for x in vote])
            origp = map(lambda i: (0.13 + 0.8*origl[i]), range(len(vote)))

        # if int(elec.yr) >= 1972 and elec.Ndists >= 3 and elec.chamber == chamber and \
        #    len(filter(lambda x: x > 0.5, elec.demfrac)) > 0 and \
        #    len(filter(lambda x: x > 0.5, elec.demfrac)) < elec.Ndists and \
        #    (elec.chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
        #     adem,arep = pandc_four_options(elec.state,elec.yr,elec.demfrac)

            # try to pack/crack
            rvote = sorted([x[1] for x in vote])
            npack += 1
            if len(filter(lambda x: x < 0.5,rvote)) >= 2 and \
               len(filter(lambda x: x > 0.5,rvote)) >= 2 and \
               rcp.pop() < p:
                isok,newl = pack_or_crack(origl,crack=(rcp.pop() < 1))
                if isok:
                    newp = map(lambda i: (0.13 + 0.8*newl[i])+rerr.pop(), range(len(newl)))
                    ea = [1/(1+np.exp(-(betac[yr]+betav*t))) for t in origp]
                    nea = [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newp]

                    # for i in range(len(origp)):
                    #     print origp[i],newp[i],ea[i],nea[i]
                        # print [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newy]
                    # print [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newz]
                    # get new estimate
                    delx = 1.0/(2*len(newp))
                    xvals = np.linspace(delx,1-delx,len(newp))
                    est.append(sum([1/(1+np.exp(-(betac[yr]+betav*t))) for t in origp]))
                    nest.append(sum([1/(1+np.exp(-(betac[yr]+betav*t))) for t in newp]))
                    sarr.append(len(filter(lambda x: x > 0.5, [x[0] for x in vote])) + np.random.randn(1)[0]*0.1)

    # plt.savefig('/home/gswarrin/research/gerrymander/pack-move')
    # plt.close()
    return npack,sum(est),sum(nest)

def test_seat_change(d,betav,betac,p):
    """ see how packing and cracking affect estimated number of seats
    """
    fig,axes = plt.subplots(1,3,figsize=(18,6))

    # add some random error back in 
    rerr = list(0.057 * np.random.randn(10000))
    rcp = list(np.random.uniform(0,1,10000))
    sarr = []
    est = []
    nest = []
    npack = 0

    allvote = []
    for yr in d.keys():
        for st in ['CO']: # d[yr].keys():
            vote = d[yr][st]
            # if len(vote) < 20:
            #     continue
            allvote.extend(vote)

            # convert to presidential vote
            origl = sorted([x[0] for x in vote])
            origp = map(lambda i: (0.13 + 0.8*origl[i]), range(len(vote)))

            # try to pack/crack
            rvote = sorted([x[1] for x in vote])
            if len(filter(lambda x: x < 0.5,rvote)) >= 2 and \
               len(filter(lambda x: x > 0.5,rvote)) >= 2 and \
               rcp.pop() < p:
                isok,newl = pack_or_crack(origl,crack=False) # (rcp.pop() < 1))
                if isok:
                    newp = map(lambda i: (0.13 + 0.8*newl[i]), range(len(newl)))
                    ea = [1/(1+np.exp(-(betac[yr]+betav*t))) for t in origp]
                    nea = [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newp]

                    for i in range(len(origp)):
                        print origp[i],newp[i],ea[i],nea[i]
                        # print [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newy]
                    # print [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newz]
                    # get new estimate
                    delx = 1.0/(2*len(newp))
                    xvals = np.linspace(delx,1-delx,len(newp))
                    est.append(sum([1/(1+np.exp(-(betac[yr]+betav*t))) for t in origp]))
                    nest.append(sum([1/(1+np.exp(-(betac[yr]+betav*t))) for t in newp]))
                    sarr.append(len(filter(lambda x: x > 0.5, [x[0] for x in vote])) + np.random.randn(1)[0]*0.1)
                    axes[0].scatter(xvals,sorted(origp),color='blue',s=20)
                    axes[0].scatter(xvals,sorted(newp),color='purple',s=20)
                    axes[1].scatter(xvals,sorted(ea),color='orange',s=20)
                    axes[1].scatter(xvals,sorted(nea),color='red',s=20)
                    axes[2].scatter(xvals,sorted(origl),color='green',s=20)
                    axes[2].scatter(xvals,sorted(newl),color='brown',s=20)
                    axes[0].set_title('pres vote')
                    axes[1].set_title('prob seat')
                    axes[2].set_title('legi vote')

    # print stats.pearsonr([x[0] for x in allvote],[x[1] for x in allvote])
    # print stats.linregress([x[0] for x in allvote],[x[1] for x in allvote])
    # plt.plot([0,1],[0.1,0.1+0.86],color='black')
    newy = map(lambda i: allvote[i][1]-(0.13 + 0.8*allvote[i][0]), range(len(allvote)))
    # plt.scatter([x[0] for x in allvote],newy,color='green')
    # print "stddev: ",np.std(newy)
    # plt.plot([0,1],[0.13,0.13+0.8],color='black')

    # plt.scatter(sarr,est,s=5)
    print "total estd: ",sum(est),len(allvote)
    print "new estd: ",sum(nest),len(allvote)

    # this may be appropriate viewpoint, but I am treating leg vote as indep variable....
    # mydata = odr.Data([x[0] for x in allvote], [x[1] for x in allvote])
    # myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
    # myoutput = myodr.run()
    # myoutput.pprint()
    # print myoutput.sum_square

    plt.savefig('/home/gswarrin/research/gerrymander/pack-move')
    plt.close()
    return npack,sum(est),sum(nest)

def try_lots_of_packing(N,p=1):
    """ p is prob of trying to pack/crack
    """
    xarr = []
    yarr = []
    betac = {2012: -7.6}
    # betac = {2008: -7.36, 2012: -7.6}
    # betac = {2008: -8.4, 2012: -8.4}
    betav = 15.79
    d = get_leg_vote(2012,2012)

    for i in range(N):
        npack,sest,snest = plot_seat_change(d,betav,betac,p)
        print npack
        xarr.append(npack)
        yarr.append(snest-sest)
        
    fig = plt.figure(figsize=(8,8))
    plt.scatter(xarr,yarr)
    plt.savefig('/home/gswarrin/research/gerrymander/random_pc-new')
    plt.close()
    
# get_seat_change(17.27,-9.09)
# get_seat_change(21.08,-10.93)
try_lots_of_packing(1,.8)

######################################################################################################

def list_districts(elecs):
    """
    """
    # so we can try to match things up....
    statelist = ['b','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    ####################################################################################################
    # get districts for given year,state from Jacobson file (this is what we have for presidential vote)
    ####################################################################################################
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    yrs = range(2010,2014,2)
    d = [dict() for j in yrs]
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
        yridx = yrs.index(yr)
        dist = l[1][-2:]
        st = int(l[1][:-2])
        # print st,yr
        
        # ky = '_'.join([statelist[st],dist])
        if statelist[st] in d[yridx].keys():
            d[yridx][statelist[st]].append(dist)
        else:
            d[yridx][statelist[st]] = [dist]

    ############################################
    # now look at districts that appear in elecs
    ############################################
    nd = [dict() for j in yrs]
    for elec in elecs.values():
        if int(elec.yr) in yrs:
            yridx = yrs.index(int(elec.yr))
            nd[yridx][elec.state] = sorted(elec.dists)

    for yr in yrs:
        for k in nd[yrs.index(yr)].keys():
            print yr,k,sorted(nd[yrs.index(yr)][k])

    ####################################################################################################
    # get districts for given year,state from Jacobson file (this is what we have for presidential vote)
    ####################################################################################################
    
    

def find_pres_leg_linregress():
    """
    """
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    expd = 0.0
    cnt = 0
    tot = 0
    ans = []
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
            ans.append(float(l[12])/100)
    
# list_districts(Nelections)

def jeff_data():
    """ June 01 - spit out data in nice form for Jeff to use
    """
    # so we can try to match things up....
    statelist = ['b','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

    #################################################
    # get districts for given year,state from Jacobson file (this is what we have for presidential vote)
    #################################################
    outf = open('/home/gswarrin/research/gerrymander/data/jeffdata.csv','w')
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')

    outf.write('Year,State,District,Winner,PresVote,LegVote,Inc\n')
    cnt = 0
    for line in f:
        cnt += 1 
        if cnt == 1:
            hdrs = line.strip().split(',')
            continue
        l = line.strip().split(',')

        # only keep certain years
        if int(l[0]) < 1972:
            continue

        # skip line if we don't have the legislative vote
        if l[4] == '' or l[4] == ' ':
            lv = 'NA'
        else:
            lv = float(l[4])/100
        
        # incumbency
        if int(l[2]) == 0:
            inc = 'R'
        elif int(l[2]) == 1:
            inc = 'D'
        else:
            inc = 'N'

        if l[12] == '' or l[12] == ' ':
            pv = 'NA'
        else:
            pv = float(l[12])/100
            
        if l[3] == '' or l[3] == ' ':
            winner = 'NA'
        else:
            winner = int(l[3])

        yr = int(l[0])
        dist = l[1][-2:]
        st = statelist[int(l[1][:-2])]

        mystr = ','.join([str(yr),st,dist,str(winner),str(pv),str(lv),str(inc)])
        outf.write(mystr + '\n')
    f.close()
    outf.close()

# jeff_data()

