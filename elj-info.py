def count_uncontested(elecs,chamber,verbose=True):
    """ Count uncontested races for 3.1 of paper
    """
    tot_elec = 0  # number of elections
    tot_unc = 0   # number of uncontested races
    tot_race = 0  # total number of races
    for elec in elecs.values():
        if elec.chamber != chamber:
            continue
        for i in range(elec.Ndists):
            if elec.status[i] < 2:
                tot_unc += 1
            tot_race += 1
        tot_elec += 1
        # print '_'.join([elec.yr,elec.state,elec.chamber])
    if verbose:
        print "Section 3.1:"
    print "   Chamber %s: %d elections, %d races, %d uncontested" % (chamber,tot_elec, tot_race, tot_unc)

def get_frac_uncontested(elec):
    """ return the fraction of races in the election that were uncontested
    """
    return len(filter(lambda i: elec.status[i] < 2, range(elec.Ndists)))*1.0/elec.Ndists

def characterize_uncontested(elecs,chamber,minN=1,verbose=True):
    """ Characterize number of elections with various amounts of uncontestedness
    """
    frac = []     # stores fraction of races uncontested in given election
    for elec in elecs.values():
        if elec.chamber != chamber or elec.Ndists < minN:
            continue
        # print elec.yr,elec.state,elec.chamber,elec.Ndists
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
    print "   Chamber %s: avg uncontest %.2f" % (chamber,np.mean(sfrac))
    make_histogram('uncontest-' + ("%s" % (chamber)),sfrac)

def fl_uncontested(elecs,verbose=True):
    """ Print data on number of uncontested races in Florida
    """
    if verbose:
        print "Section 3.1:"
        print "   Fl 1996. %d/%d uncontested" % (len(filter(lambda x: x < 2, elecs['1996_FL_11'].status)),elecs['1996_FL_11'].Ndists)
        print "   Fl 1998. %d/%d uncontested" % (len(filter(lambda x: x < 2, elecs['1998_FL_11'].status)),elecs['1998_FL_11'].Ndists)

def TN_NC_PA_metrics(elecs,verbose=True):
    """ Check two values
    """
    med = np.median(elecs['2006_TN_11'].demfrac)
    avg = np.mean(elecs['2006_TN_11'].demfrac)
    if verbose:
        print "Section 6.3:"
        print "   2006 TN mean-median is %.2f, value in paper is -0.13" % (avg-med)

    if verbose:
        print "Figure 1:"
        print "   2014 NC declination is %.2f, value in paper is 0.54" % (get_declination('NC',elecs['2014_NC_11'].demfrac))
    
    elec = elecs['2012_PA_11']
    dec = get_declination('PA',elec.demfrac)*math.log(elec.Ndists)/2
    if verbose:
        print "Section 3.2:"
        print "   2012 PA tilde(delta) is %.2f, value in paper is 0.76" % (dec)

def compare_tau_at_twofifths(elecs,chm,verbose=True):
    """ rewritten from make_table_one. 
    """
    farr = []
    oarr = []
    # elec.Ndists >= 8 and \
    for elec in elecs.values():
        if elec.chamber == chm and int(elec.yr) >= GLOBAL_MIN_YEAR and \
           1 <= len(filter(lambda x: x > 0.5, elec.demfrac)) < elec.Ndists:
            fa = get_declination(elec.state,elec.demfrac)
            ogap = get_tau_gap(elec.demfrac,0.4)
            farr.append(fa)
            oarr.append(ogap)
    a1,p1 = stats.pearsonr(farr,oarr)
    # print stats.pearsonr(farr,oarr)
    if verbose:
        print "Section 6.2"
        print "  Correlation between tau=2/5 and declination is"
        print "     r=%.2f p < %.4f among %d cong elections" % (a1,p1,len(farr))
        print "     value in paper is r=0.91 for 827 elections"

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
                for yr in [(yrmin+k) for k in range(yrmax-yrmin+1)]:
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

def table_read_in(elecs,yrmin=GLOBAL_MIN_YEAR,yrmax=2016):
    """ make table enumerating elections of various types read in
    """
    d = dict()
    tot = [0,0]
    for yr in range(yrmin,yrmax+1):
        d[yr] = [0,0]
        for elec in elecs.values():
            if int(elec.yr) == yr:
                if elec.chamber == '11':
                    d[yr][1] += 1
                    tot[1] += 1
                else:
                    d[yr][0] += 1
                    tot[0] += 1
        print yr,d[yr]
    print tot

def scatter_info(elecs):
    """ look for state races with very low EGap and dec close to 0
    """
    for elec in elecs.values():
        if elec.chamber == '11' and elec.Ndists > 0:
            vals = elec.demfrac
            dec = get_declination('',vals)
            eg = get_tau_gap(vals,0)
            if abs(dec) < 0.2 and eg < -0.35:
                print elec.yr,elec.state,elec.chamber,dec,eg
                
def new_wi_scatter_decade(elecs,mmd):
    lang = [[] for i in range(5)]
    lzgap = [[] for i in range(5)]
    cnt = 0
    for elec in elecs.values():
        # elec.myprint()
        # print "yr: ",elec.yr
           # (elec.yr not in Mmmd.keys() or elec.state not in Mmmd[elec.yr]) and \
        if 2010 >= int(elec.yr) >= 1972 and \
           elec.state not in mmd['1972'] and \
            elec.Ndists > 0 and elec.chamber=='9':
            ang = get_declination(elec.state,elec.demfrac)
            zgap = get_tau_gap(elec.demfrac,0)
            cnt += 1

            yridx = int((int(elec.yr)-1972)/10)
            if abs(ang) != 2:
                lang[yridx].append(ang)
                lzgap[yridx].append(zgap)
            # print "blah"
            print "% .3f % .3f %3d %s %s %2d" % (ang,zgap,elec.Ndists,elec.yr,elec.state,int(elec.chamber))
    print "total: ",cnt
    plt.figure(figsize=(8,8))
    plt.gca().set_axis_bgcolor('none')
    plt.axis([-0.6,0.6,-0.6,0.6])
    # plt.axis([-40,40,-40,40])
    plt.axvline(0,color='black',ls='dotted')
    plt.axhline(0,color='black',ls='dotted')
    cols = ['green','blue','red','orange']
    # markers = ['x','o','v','*']
    markers = ['o','o','o','o']
    legs = []
    for i in range(4):
    # tmp, = plt.plot([i for i in range(22)],o1,label='$\mathrm{Gap}_1$',linestyle='dashed')
    # 
        tmp = plt.scatter(lang[i],lzgap[i],color=cols[i],marker=markers[i])
        print stats.pearsonr(lang[0] + lang[1] + lang[2] + lang[3],lzgap[0] + lzgap[1] + lzgap[2] + lzgap[3])
        print i,np.mean(lang[i]),np.mean(lzgap[i])
        legs.append(tmp)
        # print np.std(lang)
        # print np.std(lzgap)
    plt.legend(legs,('1972-1980','1982-1990','1992-2000','2002-2010'),loc='upper left')
    plt.xlabel('Declination',fontsize=18)
    plt.ylabel('$\mathrm{Gap}_0$',fontsize=18)
    datax = [0.196,0.165,0.140]
    datay = [0.204,0.157,0.136]
    #  datax = [0.055,0.343,0.244,0.328]
    # datay = [0.150,0.247,0.251,0.280]
    plt.scatter(datax,datay,color='black',marker='o',s=100)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'blah') # 'scatter-wi-nommd')
    plt.close()
    # make_scatter('wi-ex-scatter',lang,lagap)

def sensitivity(elections,chm,amt):
    """ check sensitivity to imputation values
    """
    print "Section 3.1:"
    for dwinup in [1]:
        for dloseup in [1]:
            angdiff = []
            ogapdiff = []
            sz = []
            for elec in elections.values():
                if int(elec.yr) >= GLOBAL_MIN_YEAR and elec.chamber == chm and 1 in elec.status and \
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
                    ogap = get_tau_gap(elec.demfrac,0)
                    newogap = get_tau_gap(vals,0)
                    sz.append(len(filter(lambda x: x == 1, elec.status))*1.0/elec.Ndists)
                    angdiff.append(newang-ang)
                    ogapdiff.append(newogap-ogap)
                    # if angdiff[-1] == 0:
                    #    print elec.yr,elec.state,elec.demfrac,elec.status
            print "  Chamber %s -- Dwin adj: %.2f Dlose adj: %.2f yields mean ang diff: %.2f" % \
                (chm,dwinup*amt,dloseup*amt,np.mean(angdiff))
            make_scatter("ang-systemic_%d_%d" % (dwinup,dloseup),sz,angdiff)
            # make_scatter("ogap-systemic_%d_%d" % (dwinup,dloseup),sz,ogapdiff)            

            slope, intercept, r_value, p_value, std_err = stats.linregress(sz,angdiff)
            print "  Ang Dwin adj: %.2f Dlose adj: %.2f slope: %.2f %.2f" % (dwinup*amt,dloseup*amt,slope,r_value)
            # slope, intercept, r_value, p_value, std_err = stats.linregress(sz,ogapdiff)
            # print "Zgap Dwin adj: %d Dlose adj: %d slope: %.2f %.2f" % (dwinup,dloseup,slope,r_value)

def percent_persist_sign(elecs,states,chmlist,thresh):
    """ Count percent that persist in sign throughout decade (don't worry about plan changes)
    Not used in paper - only looks at ones starting decade with value above threshold - much fewer of these.
    """
    tot = 0
    cnt = 0
    for chamber in chmlist:
        for state in states:
            for yr in ['1971','1981','1991','2001']:
                keya = '_'.join([yr,state,chamber])
                keyb = '_'.join([str(int(yr)+1),state,chamber])
                if keya in elecs.keys():
                    elec = elecs[keya]
                elif keyb in elecs.keys():
                    elec = elecs[keyb]
                else:
                    continue
                dec = get_declination(state,elec.demfrac)
                if elec.state != elec.cyc_state or abs(dec) == 2:
                    continue
                dec *= math.log(elec.Ndists)/2
                if abs(dec) <= thresh:
                    continue
                tot += 1
                keeps_sign = True
                for j in range(10):
                    newkey = '_'.join([str(int(elec.yr)+j),state,chamber])
                    if newkey in elecs.keys():
                        newelec = elecs[newkey]
                        if get_declination(state,newelec.demfrac)*dec < 0:
                            keeps_sign = False
                if keeps_sign:
                    cnt += 1
    print "Section 3.2: "
    print "  %.2f = %d / %d with initial abs(tilde_dec) > %.2f persist in sign over decade" % (cnt*1.0/tot,cnt,tot,thresh)
                        
def new_percent_persist_sign(elecs,states,chmlist,thresh):
    """ Count percent that persist in sign throughout decade (don't worry about plan changes)
    """
    tot = 0
    cnt = 0
    for elec in elecs.values():
        dec = get_declination(elec.state,elec.demfrac)
        if elec.state != elec.cyc_state or abs(dec) == 2:
            continue
        dec *= math.log(elec.Ndists)/2
        if abs(dec) <= thresh:
            continue
        # this is an election that meets the criteria
        tot += 1
        keeps_sign = True
        yrrem = int(elec.yr)%10
        if yrrem == 0:
            yrrem = 10
        for j in range(-yrrem+1,10-yrrem+1):
            newkey = '_'.join([str(int(elec.yr)+j),elec.state,elec.chamber])
            if newkey in elecs.keys():
                newelec = elecs[newkey]
                if get_declination(elec.state,newelec.demfrac)*dec < 0:
                    keeps_sign = False
        if keeps_sign:
            cnt += 1
    print "Section 3.2: "
    print "  %.2f = %d / %d with initial abs(tilde_dec) > %.2f persist in sign over decade" % (cnt*1.0/tot,cnt,tot,thresh)
                

def list_third_party(ImpElecs,elecs):
    """ list third-party candidates listed in state elections
    """
    d = dict()
    tendiff = 0
    fivediff = 0
    print "Section 3.3:"
    for elec in elecs.values():
        ekey = '_'.join([elec.yr,elec.state,elec.chamber])
        if elec.chamber != '9':
            continue
        try_mod = True # False
        for i in range(elec.Ndists):
            if elec.thdpty[i] != None:
                dv = elec.dcands[i].votes
                rv = elec.rcands[i].votes
                maxv = max(dv,rv)
                minv = min(dv,rv)
                # print elec.yr,elec.state,elec.dcands[i].votes,elec.rcands[i].votes
                osum = 0
                for j in range(len(elec.thdpty[i])):
                    osum += elec.thdpty[i][j].votes
                    # print "  ",elec.thdpty[i][j].party,elec.thdpty[i][j].votes
                if minv+osum >= maxv or 1==1:
                    try_mod = True
                    if ekey in d:
                        d[ekey] += 1
                    else:
                        d[ekey] = 1
                    # print elec.yr,elec.state,elec.dists[i]," -+- ",elec.dcands[i].votes,elec.rcands[i].votes
                    # for j in range(len(elec.thdpty[i])):
                    #    print "  ",elec.thdpty[i][j].party,elec.thdpty[i][j].votes

        if try_mod:
            ikey = '_'.join([elec.yr,elec.state,elec.chamber])
            if ikey not in ImpElecs.keys():
                print ikey
                continue
            ielec = ImpElecs[ikey]
            # original declination
            orig_dec = get_declination('',ielec.demfrac)
            # get vector of values of third-party votes
            tp = [0 for j in range(elec.Ndists)]
            for i in range(elec.Ndists):
                if elec.thdpty[i] != None:
                   for j in range(len(elec.thdpty[i])):
                       tp[i] += elec.thdpty[i][j].votes
            # create a modified dem fraction for each district
            # - if district wasn't contested by both parties, then leave the same
            demadd_frac = []
            repadd_frac = []
            for i in range(elec.Ndists):
                if elec.status[i] == 2: # was contested
                    demadd_frac.append((elec.dcands[i].votes+tp[i])*1.0/(elec.dcands[i].votes+elec.rcands[i].votes+tp[i]))
                    repadd_frac.append(max(0,(elec.dcands[i].votes)*1.0/(elec.dcands[i].votes+elec.rcands[i].votes+tp[i])))
                else:
                    demadd_frac.append(ielec.demfrac[i])
                    repadd_frac.append(ielec.demfrac[i])

            dem_dec = get_declination('',demadd_frac)
            rep_dec = get_declination('',repadd_frac)
            # for i in range(elec.Ndists):
            #     print "%s %d %.3f %.3f %.3f" % (elec.dists[i],ielec.status[i],\
            #                                  ielec.demfrac[i],demadd_frac[i],repadd_frac[i])
            maxdiff = max(abs(dem_dec-orig_dec),abs(rep_dec-orig_dec))
            if maxdiff >= 0.1:
                tendiff += 1
            if maxdiff >= 0.05:
                fivediff += 1
            if elec.state == 'NV' and elec.yr == '2008' and elec.chamber == '9':
                print "  %s %s %.3f %.3f -_- Orig dec: %.3f Dem-leaning dec: %.3f Rep-leaning dec: %.3f" % \
                    (elec.yr,elec.state,maxdiff,maxdiff*math.log(elec.Ndists)/2,orig_dec,dem_dec,rep_dec)
    print "  geq 0.10 %d ... geq 0.05 %d" % (tendiff,fivediff)

def fixed_validate(elections,years,states,tot=20,winper=0.65):
    """ just compare to a fixed win percentage
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
                # print elec.dcands[rdi],elec.dcands[rdi].winner
                # print type(elec.dcands[rdi].winner),elec.dcands[rdi].winner
                if elec.dcands[rdi].winner:
                    ans += pow(elec.demfrac[rdi]-winper,2)
                    # print "D win -- %s %s dist: %2d per: %.2f" % (elec.yr,elec.state,rdi,elec.demfrac[rdi])
                else:
                    ans += pow(elec.demfrac[rdi]-(1-winper),2)
                    # print "D los -- %s %s dist: %2d per: %.2f" % (elec.yr,elec.state,rdi,1-elec.demfrac[rdi])
                cnt += 1
    print "Section 7:"
    print "   Total imputed-race overrides %d with value %.2f. RMS %.2f" % (tot,winper,pow(ans*1.0/tot,0.5))

def compare_imputations(elecs,flecs):
    """ Look at how much imputed values change
    """
    rms = 0
    cnt = 0
    arr = []
    for key in elecs.keys():
        if key in flecs.keys():
            eleca = elecs[key]
            elecb = flecs[key]
            # look at difference in declination
            deca = get_declination('',eleca.demfrac)
            decb = get_declination('',elecb.demfrac)
            rms += (decb-deca)*(decb-deca)
            cnt += 1
            # compare imputed values
            if abs(deca) < 2 and abs(decb) < 2:
                arr.append([deca,decb])
                # print out the elections for which values *really* change
                if abs(deca-decb) > 0.1:
                   print "   Election with high change: ",key
    rms /= cnt
    make_scatter('impdec',[x[0] for x in arr],[x[1] for x in arr])
    slope, intercept, r_value, p_value, std_err = stats.linregress([x[0] for x in arr],[x[1] for x in arr])
    print "   slope: %.5f %.5f" % (slope,r_value)
    print "   RMS of declination difference (due to imputation differences) is: %.5f" \
        % (math.pow(rms/cnt,0.5))
    
def fig8_vals(elecs):
    """
    """
    print "   1974 NC congress Gap_0: %.2f" % (get_tau_gap(elecs['1974_NC_11'].demfrac,0))
    print "   1974 NC congress Gap_1: %.2f" % (get_tau_gap(elecs['1974_NC_11'].demfrac,1))
    print "   2006 TN congress med-mean: %.2f" % \
        (np.mean(elecs['2006_TN_11'].demfrac)-np.median(elecs['2006_TN_11'].demfrac))
    print "   2012 IN congress med-mean: %.2f" % \
        (np.mean(elecs['2012_IN_11'].demfrac)-np.median(elecs['2012_IN_11'].demfrac))

###########################################################################################

# Section 3.1 - data on number of uncontested
# count_uncontested(Nelections,Nmmd,'9')
# count_uncontested(Nelections,Nmmd,'11',False)

# characterize_uncontested(Nelections,Nmmd,'9',0)
# characterize_uncontested(Nelections,Nmmd,'11',0)

# fl_uncontested(Nelections)
# TN_NC_PA_metrics(Nelections)

# compare_tau_at_twofifths(Nelections,Nmmd,'11')

def make_elj_info(arr,elecs,flecs,third_party_elecs,years,states,cycles,cyclesb):
    """ Convenience function
    """
    # Sec 3.1
    count_uncontested(elecs,'9')
    count_uncontested(elecs,'11',False)

    # Sec 3.1
    characterize_uncontested(elecs,'9',1)
    characterize_uncontested(elecs,'11',1)

    # Sec 3.1
    fl_uncontested(elecs)

    # Sec 3.1
    sensitivity(elecs,'11',0.03)
    sensitivity(elecs,'9',0.03)        

    # Sec 3.1, Sec 6.3, Sec 3.2
    TN_NC_PA_metrics(elecs)

    # Sec 3.2
    new_percent_persist_sign(elecs,states,['9','11'],0.47)

    # Sec 3.3
    list_third_party(elecs,third_party_elecs)

    # Sec 6.2
    compare_tau_at_twofifths(elecs,'11')
        
    print "Section 7"
    print "---------"
    count_totally_uncontested(cycles)
    count_totally_uncontested(cyclesb)

    print "Figure 8"
    print "--------"
    fig8_vals(elecs)

    # scatter_info(elecs)

    # Sec 7
    fixed_validate(elecs,years,states,tot=2000,winper=0.65)

    # General
    print "Comparing imputation runs"
    print "-------------------------"
    compare_imputations(elecs,flecs)

