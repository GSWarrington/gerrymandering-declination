def make_extreme_table(elecs,states,chm):
    """ make latex table out of extreme values
    """
    ans = []
    for elec in elecs.values():
        if elec.chamber == chm and int(elec.yr) >= GLOBAL_MIN_YEAR:
            ang = get_declination(elec.state,elec.demfrac)
            deltaN = ang*elec.Ndists*1.0/2
            deltae = ang*math.log(elec.Ndists)*1.0/2
            unc = len(filter(lambda i: elec.status[i] < 2, range(elec.Ndists)))
            if abs(ang) >= 2:
                ang = 0.0
                deltae = 0.0
            ans.append([elec.yr,elec.state,elec.Ndists,unc,deltae,ang,deltaN])
    ans.sort(key=lambda x: x[4])
    ans = ans[::-1]
    # print "Year & State & $\\delta_e$ & $\\delta$ & $\\delta_N$\\\\\\midrule"
    for i,x in enumerate(ans[:32]):
        # print ans
        tmpstr = "%s & %s & %d & %d & % .2f & % .2f & %.1f & & " % (x[0],x[1],x[2],x[3],x[4],x[5],x[6])
        y = ans[-i-1]
        tmp2str = "%s & %s & %d & %d & % .2f & % .2f & %.1f\\\\" % (y[0],y[1],y[2],y[3],y[4],y[5],x[6])
        print tmpstr + tmp2str

def make_table_2012(elecs):
    """ compute average value of metrics as well as average y-to-y variability
    """
    farr = []
    oarr = []
    ans = []
    for elec in elecs.values():
        if elec.chamber == '11' and int(elec.yr) == 2012 and elec.Ndists >= 8 and \
           1 <= len(filter(lambda x: x > 0.5, elec.demfrac)) < elec.Ndists:
            dec = get_declination(elec.state,elec.demfrac)
            unc = len(filter(lambda x: x < 2, elec.status))
            if abs(dec) < 2:
                # print " % .2f % .2f % .2f %s %d %d" % \
                # (dec*math.log(elec.Ndists)/2, dec, dec*elec.Ndists/2, elec.state, elec.Ndists, unc)
                ans.append([dec*math.log(elec.Ndists)/2, " %.2f & % .2f & % .2f & %s & %d & %d &" % \
                            (dec*math.log(elec.Ndists)/2, dec, dec*elec.Ndists/2, elec.state, \
                             elec.Ndists, unc)])
    ans.sort(key = lambda x: x[0])
    ans = ans[::-1]
    for x in ans:
        print x[1]

def make_alldec_table(elecs,states,yrmax,chamber,verbose=False):
    """ Output data for all elections in database.

    Need to be a little careful to watch for state(s?) such as KY that switch from 
    odd-year elections to even-year elections. Could be written more nicely....
    """
    tot = 0
    oddyrs = []
    omitted = []
    for state in sorted(states):
        row = [state + ' & ' ]
        isok = False
        for yrint in range(GLOBAL_MIN_YEAR,yrmax+1,2):
            key = '_'.join([str(yrint),state,chamber])
            keyb = '_'.join([str(yrint+1),state,chamber])
            dec = 2
            # only state election not MMD and one side wins all seats is 1974_AL_9.
            if key in elecs.keys():
                dec = get_declination(state,elecs[key].demfrac)
                if elecs[key].state not in oddyrs:
                    oddyrs.append(elecs[key].state)
            if keyb in elecs.keys():
                dec = get_declination(state,elecs[keyb].demfrac)
            if key in elecs.keys() and keyb in elecs.keys():
                print "WARNING: Elections for two successive years (shouldn't happen) %s %s %s" % \
                    (elecs[key].yr,elecs[key].state,elecs[key].chamber)
            if abs(dec) == 2:
                row.append(" ")
                if abs(dec) == 2 and (key in elecs.keys() or keyb in elecs.keys()) and verbose:
                    if verbose:
                        print "Whoops: One side won all of the seats",key,keyb
            else:
                isok = True
                row.append("%.2f" % (dec))
                tot += 1  # - if count here get 633
            # take care for state(s?) KY that transitions from odd-year to even-year elections
            if yrint < yrmax-1:
                row.append(" & ")
            else:
                row.append("\\\\")
        if isok:
            print ''.join(row)
        else:
            omitted.append(state)
    if verbose:
        print "Total number of values for %s is: %d" % (chamber,tot)
    print "Omitted: ",omitted
    print "States with odd-year elections: %s " % (oddyrs)
    return tot

def make_elj_tables(arr,elecs,states):
    """ Convenience function
    """
    if 1 in arr:
        print ""
        print "Table 1"
        print "-------"
        # Table 1 - only states in 2012
        make_table_2012(elecs)
    if 2 in arr:
        print ""
        print "Table 2"
        print "-------"
        # Table 2 - worst of all time
        make_extreme_table(elecs,Nstates,'11')
    if 3 in arr:
        print ""
        print "Table 3"
        print "-------"
        # Table 3 - all state elections
        print make_alldec_table(elecs,states,2010,'9',False)
    if 4 in arr:
        print ""
        print "Table 4"
        print "-------"
        # Table 4 - all congressional elections
        print make_alldec_table(elecs,states,2016,'11',False)
        
