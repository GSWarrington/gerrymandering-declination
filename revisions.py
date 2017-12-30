# def plot_uncontestedness(elections,cycles):
#     """ look at how uncontested races affect variance of declination
#     """
    
def get_frac_uncontested(elec):
    """ return the fraction of races in the election that were uncontested
    """
    return len(filter(lambda i: elec.status[i] < 2, range(elec.Ndists)))*1.0/elec.Ndists

def plot_uncontestedness_delta(elections,mmd):
    """ look at uncontestedness percent changes year over year

    Issues to be concerned about with imputation
    - wild changes in percent uncontested -> wild changes in declination
    - systematic bias in imputation affecting declination
    - keep track of how many elections have given percentage

    """
    unarr = []
    dearr = []
    Narr = []
    farr = []
    totraces = 0
    totuncont = 0
    for elec in elections.values():
        # skip elections with multimember districts
        if elec.chamber == '9' and elec.yr in mmd.keys() and elec.state in mmd[elec.yr]:
            continue
        # if elec.Ndists < 5:
        #     continue
        # if elec.chamber == '9':
        #     continue
        if int(elec.yr) < 1972:
            continue
        new_yr = str(int(elec.yr)+2)
        new_key = '_'.join([new_yr,elec.state,elec.chamber])
        if new_key in elections.keys():
            totraces += elec.Ndists
            totuncont += len(filter(lambda i: elec.status[i] < 2, range(elec.Ndists)))
            nelec = elections[new_key]
            ori_un = get_frac_uncontested(elec)
            new_un = get_frac_uncontested(nelec)
            ori_dec = get_declination('',elec.demfrac)
            new_dec = get_declination('',nelec.demfrac)
            # declination isn't defined one of two years
            if abs(ori_dec) == 2 or abs(new_dec) == 2:
                continue
            print "%s %s %s %d: %.2f %.2f" % (elec.yr,elec.state,elec.chamber,elec.Ndists,new_un-ori_un,new_dec-ori_dec)
            # print "%s %s %s %d %.2f" % (elec.yr,elec.state,elec.chamber,elec.Ndists,ori_un)
            unarr.append(abs(new_un-ori_un))
            dearr.append(abs(new_dec-ori_dec))
            Narr.append(nelec.Ndists)
            farr.append(ori_un)
    make_scatter('unc-v-dec',unarr,dearr)
    make_scatter('Nvun',Narr,unarr)    
    make_scatter('fun',Narr,farr)                
    print totuncont,totraces

def list_third_party(ImpElecs,elecs,lmmd):
    """ list third-party candidates listed in state elections
    """
    d = dict()
    for elec in elecs.values():
        ekey = '_'.join([elec.yr,elec.state,elec.chamber])
        if elec.yr in lmmd.keys() and elec.state in lmmd[elec.yr]:
            continue
        if elec.chamber != '9':
            continue
        if elec.state != 'NV' or elec.yr != '2008':
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
                    print elec.yr,elec.state,elec.dists[i]," -+- ",elec.dcands[i].votes,elec.rcands[i].votes
                    for j in range(len(elec.thdpty[i])):
                       print "  ",elec.thdpty[i][j].party,elec.thdpty[i][j].votes

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
            for i in range(elec.Ndists):
                print "%s %d %.3f %.3f %.3f" % (elec.dists[i],ielec.status[i],\
                                             ielec.demfrac[i],demadd_frac[i],repadd_frac[i])
            maxdiff = max(abs(dem_dec-orig_dec),abs(rep_dec-orig_dec))
            print "%s %s %.3f %.3f -_- %.3f %.3f %.3f" % \
                (elec.yr,elec.state,maxdiff,maxdiff*math.log(elec.Ndists)/2,orig_dec,dem_dec,rep_dec)

            # print len(ielec.demfrac),len(demadd_frac),len(repadd_frac)
            # fig = plt.figure(figsize=(scalef*8,scalef*4),dpi=mydpi)
            # plt.scatter(range(elec.Ndists),sorted(demadd_frac),color='blue')
            # plt.scatter(range(elec.Ndists),sorted(repadd_frac),color='red')
            # plt.scatter(range(elec.Ndists),sorted(ielec.demfrac),color='green')
            # output_fig(fig,"impact")
            # plt.close()

# 2006 TX TX17.  -+-  19640 19225
#    400 1283
# 2006 TX TX32.  -+-  17607 16840
#    400 2038
# 2006 TX TX85.  -+-  14323 14106
#    400 793
# 2006 TX TX93.  -+-  10936 10349
#    400 759
# 2006 TX TX106.  -+-  10224 10459
#    400 591
# 2006 TX TX118.  -+-  10982 10082
#    400 1701
# 2006 TX 0.093 -_- 0.025 0.052 0.118

    # for key in d.keys():
    #     print "%.3f %s" % (d[key]*1.0/elecs[key].Ndists,key)
