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
            unc = len(filter(lambda i: elec.status[i] < 2, range(elec.Ndists)))
            if abs(ang) >= 2:
                ang = 0.0
                deltae = 0.0
            ans.append([elec.yr,elec.state,elec.Ndists,unc,deltae,ang,deltaN])
    ans.sort(key=lambda x: x[4])
    ans = ans[::-1]
    # print "Year & State & $\\delta_e$ & $\\delta$ & $\\delta_N$\\\\\\midrule"
    for i,x in enumerate(ans):
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
# make_extreme_table(Nelections,Nstates,Nmmd,'11')
make_table_2012(Nelections)
