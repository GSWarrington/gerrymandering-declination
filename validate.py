##############################################################################
# cross-validate the imputed values

def validate_get_model_input_data(cycle):
    """
    """
    ild = [] # incumbent list democrats
    ilr = [] # incumbent list republicans
    pv = []
    sts = []
    dists = []
    yrs = []
    wind = []
    winr = []
    # print "duh: ",len(curelections.keys())
    for elec in cycle.elecs.values():
        # print elec.Ndists,elec.yr
        for i in range(elec.Ndists):
            if elec.status[i] == 2: # doesn't need to be imputed
                # actual_votes,dv,rv = elec.compute_district_vote(i)
                # if actual_votes:
                # print actual_votes,dv,rv
                # 02.08.17
                # for testing "winner" flag instead of "incumbency" flag
                if elec.dcands[i].winner:
                    wind.append(1)
                else:
                    wind.append(0)
                if elec.rcands[i].winner:
                    winr.append(1)
                else:
                    winr.append(0)
                    
                # usual use of incumbency
                if elec.dcands[i].incum:
                    ild.append(1)
                else:
                    ild.append(0)
                if elec.rcands[i].incum:
                    ilr.append(1)
                else:
                    ilr.append(0)

                dists.append(elec.dists[i])
                # pv.append(dv*1.0/(dv+rv))
                pv.append(elec.demfrac[i])
                # REAPP
                sts.append(elec.cyc_state)
                yrs.append(elec.yr)
    # print len(il),len(pv),len(sts)
    return wind,winr,ild,ilr,pv,sts,dists,yrs

def cycle_district_known(cycle,st,chm,dist):
    """ how many races in this district in the cycle are known
    """
    ans = 0
    tot = 0
    for elec in cycle.elecs.values():
        if elec.state == st and elec.chamber == chm and \
           dist in elec.dists:
            didx = elec.dists.index(dist)
            if elec.status[didx] >= 2: # didn't need to be imputed
                ans += 1
            tot += 1
    return ans,tot

def cross_validate(cycle,num=10):
    """
    """
    winD_Ni,winR_Ni,incumbentsD_Ni,incumbentsR_Ni,demvote_Ni,states_Ni,districts_Ni,years_Ni = validate_get_model_input_data(cycle)
    
    j_winD_Ni = []
    j_winR_Ni = []
    j_incumbentsD_Ni = []
    j_incumbentsR_Ni = []
    j_demvote_Ni = []
    j_states_Ni = []
    j_districts_Ni = []
    j_years_Ni = []

    ans = 0.0
    for lll in range(num):
        # don't want to remove the last election in a particular district
        isokay = False
        j = 0
        while not isokay:
            j = randrange(len(years_Ni))
            j_incumbentsD_Ni = incumbentsD_Ni[:j] + incumbentsD_Ni[j+1:]
            j_incumbentsR_Ni = incumbentsR_Ni[:j] + incumbentsR_Ni[j+1:]
            j_winD_Ni = winD_Ni[:j] + winD_Ni[j+1:]
            j_winR_Ni = winR_Ni[:j] + winR_Ni[j+1:]
            j_demvote_Ni = demvote_Ni[:j] + demvote_Ni[j+1:]
            j_states_Ni = states_Ni[:j] + states_Ni[j+1:]
            j_districts_Ni = districts_Ni[:j] + districts_Ni[j+1:]
            j_years_Ni = years_Ni[:j] + years_Ni[j+1:]
            if districts_Ni[j] in j_districts_Ni:
                isokay = True
            else:
                print "Failed on %s %s %s %d" % (years_Ni[j],states_Ni[j],districts_Ni[j],j)
        
        # get list of all states seen
        setstates = sorted(list(set(j_states_Ni)))
        # put them into a dictionary so can assign a number to each
        state_lookup = dict(zip(setstates, range(len(setstates))))
        # replace list of states with indices of them
        statenums = map(lambda x: state_lookup[x]+1, j_states_Ni) # srrs_mn.county.replace(county_lookup).values
        
        # get list of all years seen
        setyears = sorted(list(set(j_years_Ni)))
        # put them into a dictionary so can assign a number to each
        year_lookup = dict(zip(setyears, range(len(setyears))))
        # replace list of yearss with indices of them
        yearnums = map(lambda x: year_lookup[x]+1, j_years_Ni) # srrs_mn.county.replace(county_lookup).values
        
        # get list of all districts seen
        setdists = sorted(list(set(j_districts_Ni)))
        # put them into a dictionary so can assign a number to each
        dist_lookup = dict(zip(setdists, range(len(setdists))))
        # replace list of districts with indices of them
        distnums = map(lambda x: dist_lookup[x]+1, j_districts_Ni) # srrs_mn.county.replace(county_lookup).values
        # print "distnums: ",distnums
        
        # need a lookup that tells us which state number corresponds to a given district number
        d = dict()
        for i in range(len(j_districts_Ni)):
            if j_districts_Ni[i] not in d:
                d[j_districts_Ni[i]] = j_states_Ni[i]
        statenum_lookup = []
        for i in range(len(dist_lookup)):
            statenum_lookup.append(state_lookup[d[setdists[i]]]+1)
        
        leveliii_data = {'Ni': len(j_demvote_Ni),
                       'Nj': len(setdists),
                       'Nk': len(setstates),
                       'Nl': len(setyears),
                       'district_id': distnums,
                       'state_id': statenums,
                       'year': yearnums,
                       'state_Lookup': statenum_lookup,
                       'y': j_demvote_Ni,
                       'winD': j_winD_Ni,
                       'winR': j_winR_Ni,
                       'IncD': j_incumbentsD_Ni,
                       'IncR': j_incumbentsR_Ni}
        cycle.leveliii_data = leveliii_data
        cycle.state_lookup = state_lookup
        cycle.year_lookup = year_lookup
        cycle.dist_lookup = dist_lookup
           
        if int(cycle.max_year)-int(cycle.min_year) <= 4:
            niter = 4000
            nchains = 4
        else:
            niter = 2000
            nchains = 2
        if lll == 0:
            cycle.leveliii_fit = pystan.stan(model_code=leveliii_model, data=leveliii_data, 
                                             iter=niter, chains=nchains)
        else:
            cycle.leveliii_fit = pystan.stan(fit = cycle.leveliii_fit, data=leveliii_data)
        predd = cycle.leveliii_fit.extract()
        
        # return years_Ni[j],states_Ni[j],districts_Ni[j],j_demvote_Ni[j],j
        elec = cycle.elecs['_'.join([years_Ni[j],states_Ni[j][:2],cycle.chamber])]
        didx = elec.dists.index(districts_Ni[j])
        val = elec.demfrac[didx]
        u0jk = mean(predd['u_0jk'][:,cycle.dist_lookup[districts_Ni[j]]])
        winD = mean(predd['win_D'])
        winR = mean(predd['win_R'])
        deltaD = mean(predd['delta_D'])
        deltaR = mean(predd['delta_R'])
        # beta0 = mean(predd['beta_0'])
        beta0 = 0.5
        u0k = mean(predd['u_0k'][:,cycle.state_lookup[elec.cyc_state]])
        v0l = mean(predd['v_0l'][:,cycle.year_lookup[elec.yr]])
        
        # modify based on winner
        winnermod = 0
        if elec.dcands[didx] != None and elec.dcands[didx].winner:
            winnermod += winD
        if elec.rcands[didx] != None and elec.rcands[didx].winner:
            winnermod += winR

        # modify based on incumbency of candidates
        incummod = 0
        if elec.dcands[didx] != None and elec.dcands[didx].incum:
            incummod += deltaD
        if elec.rcands[didx] != None and elec.rcands[didx].incum:
            incummod += deltaR
        
        newdv = beta0 + u0k + u0jk + v0l + incummod + winnermod
        print "lll: %d beta: %.3f u0k: %.3f u0jk: %.3f v0l: %.3f inc: %.3f win: %.3f" % (lll,beta0,u0k,u0jk,v0l,incummod,winnermod)
        actv = elec.demfrac[didx]
        ans += (newdv-actv)*(newdv-actv)
        a,b = cycle_district_known(cycle,states_Ni[j][:2],cycle.chamber,districts_Ni[j])
        print "known: %d / %d dist: %s %s %s act: %.3f est: %.3f" % (a,b,districts_Ni[j],years_Ni[j],states_Ni[j],actv,newdv)
    print "final ans: ",pow(ans/num,0.5)

def cross_validate_lots(cycles,tot=10):
    """ cross_validate through lots of cycles
    """
    for c in cycles:
        if c.chamber == '11' and int(c.min_year) == 2002:
            cross_validate(c,tot)
        
def dumb_validate(elections,years,states,tot=20,winper=0.65):
    """ just compare to 75-25
    """
    cnt = 0
    ans = 0
    while cnt < tot:
        ryr = randrange(len(years))
        rst = randrange(len(states))
        myid = '_'.join([years[ryr],states[rst],'11'])
        if int(years[ryr]) >= 1972 and myid in elections.keys():
            elec = elections[myid]
            rdi = randrange(elec.Ndists)
            if elec.status[rdi] == 2:
                if elec.dcands[rdi].winner:
                    ans += pow(elec.demfrac[rdi]-winper,2)
                else:
                    ans += pow(elec.demfrac[rdi]-(1-winper),2)
                cnt += 1
    return pow(ans*1.0/tot,0.5)

#############################################################
# TODO: Thur
# def scatter_district_effect(cycle):
#     """ see what the correlation is between frequency of uncontested races and district effect
#     """


def sensitivity(elections,chm,amt):
    """ check sensitivity to imputation values
    """
    for dwinup in [-1,1]:
        for dloseup in [-1,1]:
            angdiff = []
            ogapdiff = []
            sz = []
            for elec in elections.values():
                if int(elec.yr) >= 1972 and elec.chamber == chm and 1 in elec.status and \
                    1 <= len(filter(lambda x: x > 0.5, elec.demfrac)) < elec.Ndists:
                    vals = []
                    for j in range(len(elec.demfrac)):
                        if elec.status[j] == 2:
                            vals.append(elec.demfrac[j])
                        else:
                            if elec.demfrac[j] > 0.5:
                                vals.append(max(0.51,elec.demfrac[j]+dwinup*amt))
                            else:
                                vals.append(min(0.49,elec.demfrac[j]+dloseup*amt))
                    ang = get_declination(elec.state,elec.demfrac)
                    newang = get_declination(elec.state,vals)
                    ogap = get_tau_gap(elec.demfrac,1)
                    newogap = get_tau_gap(vals,1)
                    sz.append(len(filter(lambda x: x == 1, elec.status))*1.0/elec.Ndists)
                    angdiff.append(ang-newang)
                    ogapdiff.append(ogap-newogap)
                    # if angdiff[-1] == 0:
                    #    print elec.yr,elec.state,elec.demfrac,elec.status
            print "Dwin adj: %d Dlose adj: %d yields mean ang diff: %.2f mean ogap diff %.2f" % \
                (dwinup,dloseup,np.mean(angdiff),np.mean(ogapdiff))
            make_scatter("ang-systemic_%d_%d" % (dwinup,dloseup),sz,angdiff)
            make_scatter("ogap-systemic_%d_%d" % (dwinup,dloseup),sz,ogapdiff)            

            slope, intercept, r_value, p_value, std_err = stats.linregress(sz,angdiff)
            print "Ang Dwin adj: %d Dlose adj: %d slope: %.2f %.2f" % (dwinup,dloseup,slope,r_value*r_value)
            slope, intercept, r_value, p_value, std_err = stats.linregress(sz,ogapdiff)
            print "Zgap Dwin adj: %d Dlose adj: %d slope: %.2f %.2f" % (dwinup,dloseup,slope,r_value*r_value)


def fraction_imputed(elections,chm):
    """ check sensitivity to imputation values
    """
    tenper = 0
    tweper = 0
    tot = 0
    arr = []
    for elec in elections.values():
        if int(elec.yr) >= 1972 and elec.chamber == chm and elec.Ndists >= 8 and \
           1 <= len(filter(lambda x: x > 0.5, elec.demfrac)) < elec.Ndists:
            fracimp = 1.0*len(filter(lambda x: x == 1, elec.status))/elec.Ndists
            arr.append(fracimp)
            if fracimp < 0.10:
                tenper += 1
            if fracimp < 0.65:
                tweper += 1
#            else:
#                print elec.yr,elec.state
            tot += 1
    make_histogram('fracimp',arr)
    print "fraction < 10 percent imputed: %.2f" % (tenper*1.0/tot)
    print "fraction < 60 percent imputed: %.2f" % (tweper*1.0/tot)    

def race_data(elections,mmd,chm):
    """ check sensitivity to imputation values
    """
    imputed = 0
    tot = 0
    totelec = 0
    for elec in elections.values():
        if int(elec.yr) >= 1972 and elec.chamber == chm and \
           (elec.chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
            for j in range(elec.Ndists):
                if elec.status[j] == 1:
                    imputed += 1
                tot += 1
            totelec += 1
    print "%d %d %d" % (imputed,tot,totelec)
    

    
