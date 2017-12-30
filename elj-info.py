def count_uncontested(elecs,mmd,chamber,verbose=True):
    """ Count uncontested races for 3.1 of paper
    """
    tot_elec = 0  # number of elections
    tot_unc = 0   # number of uncontested races
    tot_race = 0  # total number of races
    for elec in elecs.values():
        if elec.chamber != chamber:
            continue
        if chamber == '11' or (elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
            for i in range(elec.Ndists):
                if elec.status[i] < 2:
                    tot_unc += 1
                tot_race += 1
            tot_elec += 1
            print '_'.join([elec.yr,elec.state,elec.chamber])
    if verbose:
        print "Section 3.1:"
    print "   Chamber %s: %d elections, %d races, %d uncontested" % (chamber,tot_elec, tot_race, tot_unc)

def get_frac_uncontested(elec):
    """ return the fraction of races in the election that were uncontested
    """
    return len(filter(lambda i: elec.status[i] < 2, range(elec.Ndists)))*1.0/elec.Ndists

def characterize_uncontested(elecs,mmd,chamber,minN=0,verbose=True):
    """ Characterize number of elections with various amounts of uncontestedness
    """
    frac = []     # stores fraction of races uncontested in given election
    for elec in elecs.values():
        if elec.chamber != chamber or elec.Ndists < minN:
            continue
        if chamber == '11' or (elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
            frac.append(get_frac_uncontested(elec))
    sfrac = sorted(frac)
    idx = 0
    jdx = 0
    for i in range(len(sfrac)):
        if sfrac[i] <= 0.10:
            idx += 1
        if sfrac[i] <= 0.50:
            jdx += 1
    print "   Chamber %s: at most 10percent: %.2f at most 50percent: %.2f" % \
        (chamber,idx*1.0/len(sfrac),jdx*1.0/len(sfrac))
    
def fl_uncontested(elecs):
    """ Print data on number of uncontested races in Florida
    """
    print "Fl 1996. %d/%d uncontested" % (len(filter(lambda x: x < 2, elecs['1996_FL_11'].status)),elecs['1996_FL_11'].Ndists)
    print "Fl 1998. %d/%d uncontested" % (len(filter(lambda x: x < 2, elecs['1998_FL_11'].status)),elecs['1998_FL_11'].Ndists)

def TN_NC_PA_metrics(elecs):
    """ Check two values
    """
    med = np.median(elecs['2006_TN_11'].demfrac)
    avg = np.mean(elecs['2006_TN_11'].demfrac)
    print "2006 TN mean-median is %.2f, should be -0.13" % (avg-med)

    print "2014 NC declination is %.2f, should be 0.54" % (get_declination('NC',elecs['2014_NC_11'].demfrac))
    
    elec = elecs['2012_PA_11']
    dec = get_declination('PA',elec.demfrac)*math.log(elec.Ndists)/2
    print "2012 PA tilde(delta) is %.2f, should be 0.76" % (dec)

def compare_tau_at_twofifths(elecs,mmd,chm):
    """ rewritten from make_table_one. 
    """
    farr = []
    oarr = []
    # elec.Ndists >= 8 and \
    for elec in elecs.values():
        if elec.chamber == chm and int(elec.yr) >= 1972 and \
           1 <= len(filter(lambda x: x > 0.5, elec.demfrac)) < elec.Ndists and \
           (elec.chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
            fa = get_declination(elec.state,elec.demfrac)
            ogap = get_tau_gap(elec.demfrac,0.4)
            farr.append(fa)
            oarr.append(ogap)
    a1,p1 = stats.pearsonr(farr,oarr)
    # print stats.pearsonr(farr,oarr)
    print "Correlation between tau=2/5 and declination is r=%.2f p < %.4f among %d cong elections" % (a1,p1,len(farr))
    print "Should be r=0.91 for 827 elections"

def count_totally_uncontested(cycles):
    """ count districts uncontested during entire cycle
    """
    for c in cycles:
        tot_contested = 0
        tot_uncontested = 0
        tot_demonly = 0
        tot_reponly = 0
        tot_both = 0
        tot_neither = 0
        # pick a state and collect its districts
        yrmin = int(c.min_year)
        yrmax = int(c.max_year)
        for st in c.states:
            dists = []
            for elec in c.elecs.values():
                if elec.state == st:
                    for d in elec.dists:
                        if d not in dists:
                            dists.append(d)
            # now run through districts and see which are contested
            for dist in dists:
                is_contested = False
                has_dem = False
                has_rep = False
                does_exist = False
                for yr in [(yrmin+2*k) for k in range((yrmax-yrmin)/2+1)]:
                    myid = '_'.join([str(yr),st,c.chamber])
                    if myid in c.elecs.keys():
                        elec = c.elecs[myid]
                        does_exist = True
                        if dist not in elec.dists:
                            a = 0
                            # print "Missing district",elec.chamber,elec.yr,elec.state,dist
                        else:
                            idx = elec.dists.index(dist)
                            if elec.status[idx] == 2:
                                is_contested = True
                            if elec.dcands[idx].winner:
                                has_dem = True
                            if elec.rcands[idx].winner:
                                has_rep = True
                if does_exist:
                    if is_contested:
                        tot_contested += 1
                    else:
                        tot_uncontested += 1                    
                        if has_dem and not has_rep:
                            tot_demonly += 1
                        if (not has_dem) and has_rep:
                            tot_reponly += 1
                        if has_dem and has_rep:
                            tot_both += 1
                        if (not has_dem) and (not has_rep):
                            tot_neither += 1
                            print "neither",c.chamber,st,yrmin,dist
        print "%s contested: %d uncont: %d (demonly: %d reponly: %d both %d neither %d)" % \
                            (c.chamber,tot_contested,tot_uncontested,tot_demonly,tot_reponly,tot_both,tot_neither)

def table_read_in(elecs,mmd,yrmin=1972,yrmax=2016):
    """ make table enumerating elections of various types read in
    """
    d = dict()
    tot = [0,0,0,0]
    for yr in range(yrmin,yrmax+2):
        d[yr] = [0,0,0,0]
        states = []
        for elec in elecs.values():
            if int(elec.yr) == yr:
                if elec.chamber == '11':
                    d[yr][3] += 1
                    tot[3] += 1
                else:
                    if str(yr) in mmd.keys() and elec.state in mmd[str(yr)]:
                        d[yr][0] += 1
                        tot[0] += 1
                        if elec.state not in states:
                            states.append(elec.state)
                    else:
                        d[yr][1] += 1
                        tot[1] += 1
                    d[yr][2] += 1
                    tot[2] += 1
        print yr,d[yr],sorted(states)
    print tot

###########################################################################################

# Section 3.1 - data on number of uncontested
# count_uncontested(Nelections,Nmmd,'9')
# count_uncontested(Nelections,Nmmd,'11',False)

# characterize_uncontested(Nelections,Nmmd,'9',0)
# characterize_uncontested(Nelections,Nmmd,'11',0)

# fl_uncontested(Nelections)
# TN_NC_PA_metrics(Nelections)

# compare_tau_at_twofifths(Nelections,Nmmd,'11')
