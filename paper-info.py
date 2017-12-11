def make_yoy_corr_scatter(elecs,mmd,chm):
    """ compute correlation between one year and the next
    """
    farr = []
    oarr = []
    mmarr = []
    ddarr = []
    for yr in [1972 + 2*j for j in range (22)]:
        for elec in elecs.values():
            if elec.chamber == chm and int(elec.yr) == yr and elec.Ndists >= 8 and \
               (elec.chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
                mmdiff = np.median(elec.demfrac)-np.mean(elec.demfrac)
                fa = get_declination(elec.state,elec.demfrac)
                tmp = '_'.join([str(yr + 2),elec.state,'11'])
                if tmp in elecs.keys():
                    nelec = elecs[tmp]
                    mmdiff2 = np.median(nelec.demfrac)-np.mean(nelec.demfrac)
                    mmarr.append([mmdiff,mmdiff2])
                    fa2 = get_declination(nelec.state,nelec.demfrac)
                    if abs(fa) < 2 and abs(fa2) < 2:
                        farr.append([fa,fa2])
            #    mmarr.append(-np.median(elec.demfrac)+np.mean(elec.demfrac))
            #    ddarr.append(fa)
            #    print " % .2f % .2f %s %s %s %d " % (np.median(elec.demfrac)-np.mean(elec.demfrac),fa,elec.yr,elec.state,elec.chamber,elec.Ndists)
    a1,pv1 = stats.pearsonr([x[0] for x in mmarr],[x[1] for x in mmarr])
    print "mm: ",a1,pv1
    make_scatter('mmscatter',[x[0] for x in mmarr], [x[1] for x in mmarr])
    a1,pv1 = stats.pearsonr([x[0] for x in farr],[x[1] for x in farr])
    print "fa: ",a1,pv1
    make_scatter('fascatter',[x[0] for x in farr], [x[1] for x in farr])
    # return mmarr,ddarr

def median_mean(elecs,mmd,chm):
    """ compute average value of metrics as well as average y-to-y variability
    """
    farr = []
    oarr = []
    mmarr = []
    ddarr = []
    for elec in elecs.values():
        if elec.chamber == chm and int(elec.yr) >= 1972 and elec.Ndists >= 8 and \
           (elec.chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
            mmdiff = abs(np.median(elec.demfrac)-np.mean(elec.demfrac))
            fa = get_declination(elec.state,elec.demfrac)
            if abs(fa) < 2:
                mmarr.append(-np.median(elec.demfrac)+np.mean(elec.demfrac))
                ddarr.append(fa)
                print " % .2f % .2f %s %s %s %d " % (np.median(elec.demfrac)-np.mean(elec.demfrac),fa,elec.yr,elec.state,elec.chamber,elec.Ndists)
    return mmarr,ddarr

def make_table_one(elecs,mmd,chm):
    """ compute average value of metrics as well as average y-to-y variability
    """
    farr = []
    oarr = []
    for elec in elecs.values():
        if elec.chamber == chm and int(elec.yr) >= 1972 and elec.Ndists >= 8 and \
           1 <= len(filter(lambda x: x > 0.5, elec.demfrac)) < elec.Ndists and \
           (elec.chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
            fa = get_declination(elec.state,elec.demfrac)
            ogap = get_tau_gap(elec.demfrac,0)
            farr.append(fa)
            oarr.append(ogap)
            demdists = len(filter(lambda x: x > 0.5, elec.demfrac))
            if abs(fa) < 2:
                # print "% .2f % .2f % .2f %s %s %s %d" % \
                #  (np.median(elec.demfrac)-np.mean(elec.demfrac),fa,fa*elec.Ndists/2,elec.yr,elec.state,elec.chamber,elec.Ndists)
                print " % .1f % .2f % .2f % .2f %s %s %s %d %.2f %.2f" % \
                (ogap,fa,fa*elec.Ndists/2,fa*math.log(elec.Ndists)/2,elec.yr,elec.state,elec.chamber,elec.Ndists,demdists*1.0/elec.Ndists,np.mean(elec.demfrac))
    a1,p1 = stats.pearsonr(farr,oarr)
    print len(farr),len(oarr),a1,p1

def Ndists_table(elecs,states,mmd,chm):
    """ compute average value of metrics as well as average y-to-y variability
    """
    if chm == '11':
        ul = 2018
    else:
        ul = 2012

    # first get headerline down
    mystr = ''
    for j in range(1972,ul,2):
        if j > 0:
            mystr += ' & '
        mystr += str(j)
    print mystr

    # now generate line for each state
    for i,st in enumerate(states):
        mystr = st
        for j in range(1972,ul,2):
            myid = '_'.join([str(j),st,chm])
            if myid not in elecs.keys() or \
               (j in mmd.keys() and elec.state in mmd[j]):
                tmpstr = ''
            else:
                elec = elecs[myid]
                tmpstr = " % 3d" % (elec.Ndists)
            mystr += ' & ' + tmpstr
            if j == ul-2:
                mystr += '\\\\'
                
        print mystr

def print_extremes(elecs,states,mmd,chm):
    """ make latex table out of extreme values
    """
    ans = []
    for elec in elecs.values():
        if elec.chamber == chm and int(elec.yr) >= 1972 and \
           (elec.chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]) and \
           elec.Ndists >= 8:
            ang = get_declination(elec.state,elec.demfrac)
            zgap = get_tau_gap(elec.demfrac,0)
            ogap = get_tau_gap(elec.demfrac,1)
            if ang <= -2:
                ang = 0.0
            print "%s %s %3d %.2f" % (elec.yr,elec.state,elec.Ndists,ang)


def make_extreme_table(elecs,states,mmd,chm):
    """ make latex table out of extreme values
    """
    ans = []
    for elec in elecs.values():
        if elec.chamber == chm and int(elec.yr) >= 1972 and \
           (elec.chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
            ang = get_declination(elec.state,elec.demfrac)
            deltaN = ang*elec.Ndists*1.0/2
            deltae = ang*math.log(elec.Ndists)*1.0/2
            # zgap = get_tau_gap(elec.demfrac,0)
            # ogap = get_tau_gap(elec.demfrac,1)
            if abs(ang) >= 2:
                ang = 0.0
                deltae = 0.0
            ans.append([elec.yr,elec.state,elec.Ndists,deltae,ang,deltaN])
    ans.sort(key=lambda x: x[3])
    ans = ans[::-1]
    # print "Year & State & $\\delta_e$ & $\\delta$ & $\\delta_N$\\\\\\midrule"
    for i,x in enumerate(ans):
        # print ans
        tmpstr = "%s & %s & %d & % .2f & % .2f & %.1f & & " % (x[0],x[1],x[2],x[3],x[4],x[5])
        y = ans[-i-1]
        tmp2str = "%s & %s & %d & % .2f & % .2f & %.1f\\\\" % (y[0],y[1],y[2],y[3],y[4],y[5])
        print tmpstr + tmp2str

def make_latex_table(elecs,states,mmd,chm,declination):
    """ 
    """
    if chm == '11':
        ul = 2018
    else:
        ul = 2012

    # first get headerline down
    mystr = ''
    for j in range(1972,ul,2):
        if j > 0:
            mystr += ' & '
        mystr += str(j)
    print mystr

    # now generate line for each state
    for i,st in enumerate(states):
        mystr = st
        for j in range(1972,ul,2):
            myid = '_'.join([str(j),st,chm])
            if myid not in elecs.keys() or \
               (chm == '9' and str(j) in mmd.keys() and st in mmd[str(j)]):
                tmpstr = ''
            else:
                elec = elecs[myid]
                if declination:
                    ang = get_declination(elec.state,elec.demfrac)
                else:
                    ang = get_tau_gap(elec.demfrac,1)
                if ang <= -2:
                    tmpstr = ' '
                else:
                    tmpstr = " %.2f" % (ang)
            mystr += ' & ' + tmpstr
            if j == ul-2:
                mystr += '\\\\'
                
        print mystr

def average_var_mag(elecs,chm):
    """ compute average value of metrics as well as average y-to-y variability
    """
    avg_mag_ang = []
    avg_mag_0gap = []
    avg_mag_1gap = []

    avg_diff_ang = []
    avg_diff_0gap = []
    avg_diff_1gap = []

    for elec in elecs.values():
        if int(elec.yr) >= 1972 and elec.chamber==chm and elec.Ndists >= 8:
            ang = get_declination(elec.state,elec.demfrac)
            zgap = get_tau_gap(elec.demfrac,0)
            ogap = get_tau_gap(elec.demfrac,1)
            if ang > -2:
                avg_mag_ang.append(abs(ang))
            avg_mag_0gap.append(abs(zgap))
            avg_mag_1gap.append(abs(ogap))
            myid2 = '_'.join([str(int(elec.yr)-2),elec.state,elec.chamber])
            if myid2 in elecs.keys():
                elec2 = elecs[myid2]
                ang2 = get_declination(elec.state,elec2.demfrac)
                zgap2 = get_tau_gap(elec2.demfrac,0)
                ogap2 = get_tau_gap(elec2.demfrac,1)
                if ang2 > -2:
                    avg_diff_ang.append(abs(ang-ang2))
                    avg_diff_0gap.append(abs(zgap-zgap2))
                    avg_diff_1gap.append(abs(ogap-ogap2))

    print "Chamber",chm
    print "ang mean: %.3f ang avg diff %.3f" % (np.median(avg_mag_ang),np.median(avg_diff_ang))
    print "0gp mean: %.3f 0gp avg diff %.3f" % (np.median(avg_mag_0gap),np.median(avg_diff_0gap))
    print "1gp mean: %.3f 1gp avg diff %.3f" % (np.median(avg_mag_1gap),np.median(avg_diff_1gap))
                
def corr_with_half(elecs):
    """ see above description
    """
    ans = [[] for i in range(4)]
    for elec in elecs.values():
        if int(elec.yr) >= 1972 and elec.chamber=='11' and elec.Ndists >= 8:
            kprimeN = len(filter(lambda x: x > 0.5, elec.demfrac))*1.0/elec.Ndists
            pbar = np.mean(elec.demfrac)

            # gap0 = get_tau_gap(elec.demfrac,0)
            gap0 = 2*(2*(pbar-0.5)+0.5-kprimeN) # equivalently

            # gapinfty = get_tau_gap(elec.demfrac,10000)
            gapinfty = 1 - 2*kprimeN
            ans[0].append(kprimeN)
            ans[1].append(pbar)
            ans[2].append(gap0)
            ans[3].append(gapinfty)
    labs = ['kprimeN','pbar','gap0','gapinfty']
    for i in range(4):
        for j in range(i+1,4):
            a1,pv1 = stats.pearsonr(ans[i],ans[j])
            print "Correlation between %s and %s: %.3f, %.3f" % (labs[i],labs[j],a1,pv1)
            make_scatter('test-' + labs[i] + '-' + labs[j],ans[i],ans[j])


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

####################################################################33
# 
def bridget_diff(elecs,states,mmd,chm):
    """ make latex table out of extreme values
    """
    ans = []
    for elec in elecs.values():
        if elec.chamber == chm and int(elec.yr) >= 1972 and \
           (elec.chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]) and \
           elec.Ndists >= 4:
            ang = get_declination(elec.state,elec.demfrac)
            zgap = get_tau_gap(elec.demfrac,0)
            mykey = elec.yr + '_' + elec.state + '_' + chm
            if abs(ang) < 2:
                print "%s %3d %.2f %.2f %.2f" % (mykey,elec.Ndists,ang,zgap,ang-zgap)
        
