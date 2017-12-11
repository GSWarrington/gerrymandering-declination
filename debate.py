################################################################
# for reconstructing pictures found in "The measure of a metric"
################################################################

def lifetime(elecs):
    """ plot initial value versus average of lifetime of plan - Fig 10
    
    EG: state >= 8: 0.86 (slope 0.73)
        cong >= 8: 0.69 (slope 0.49)
        state >= 1: 0.86 (slope 0.73)
        cong >= 1: 0.61 (slope 0.41)
    dec: state >= 8: 0.87 (slope 0.71)
        cong >= 8: 0.75 (slope 0.55)
        state >= 1: 0.87 (slope 0.71)
        cong >= 1: 0.66 (slope 0.49)
    So declination actually looks like it holds up a little better.
    Not sure why my data is so different than Stephenopoulos-McGhee, however.
    """
    # this is a crummy way to do this....
    for chm in ['9','11']:
        ansbeg = []
        ansavg = []
        for styr in ['1972','1982','1992','2002']:
            for cycst in Ncyc:
                arr = []
                for elec in elecs.values():
                    if elec.state == cycst and (int(styr) <= int(elec.yr) < int(styr) + 10) and \
                       elec.chamber == chm and elec.Ndists >= 1:
                        arr.append(elec)
                if len(arr) >= 3:
                    minpos = 0
                    for i in range(1,len(arr)):
                        if int(arr[i].yr) < int(arr[minpos].yr):
                            minpos = i
                    # begdec = get_EG(arr[minpos].demfrac)
                    # alldec = [get_EG(arr[i].demfrac) for i in range(len(arr))]
                    begdec = get_declination('',arr[minpos].demfrac)
                    alldec = [get_declination('',arr[i].demfrac) for i in range(len(arr))]
                    avgdec = np.mean(alldec)
                    if abs(begdec) == 2 or 2 in map(lambda x: abs(x), alldec):
                        continue
                    ansbeg.append(begdec)
                    ansavg.append(avgdec)
        print stats.linregress(ansbeg,ansavg)
        make_scatter('beg_avg_' + chm,ansbeg,ansavg)

def compete(elecs):
    """ plot avg margin of victory versus EG (or declination) to see if any correlation - Fig. 3

    With declination, even for state only accounts for 4% of variation; less for congress
    LinregressResult(slope=0.381, intercept=0.013, rvalue=0.20...
    LinregressResult(slope=0.084, intercept=0.174, rvalue=0.04...

    With EG and at least 6 districts 
    LinregressResult(slope=0.242, intercept=-0.018, rvalue=0.361...
    LinregressResult(slope=-0.194, intercept=0.135, rvalue=-0.230...

    """
    for chm in ['9','11']:
        xarr = []
        yarr = []
        for elec in elecs.values():
            if elec.chamber == chm and elec.Ndists > 5:
                arr = []
                for i in range(elec.Ndists):
                    if elec.demfrac[i] > 0.5:
                        arr.append(abs(2*elec.demfrac[i]-1))
                    else:
                        arr.append(abs(2*(1-elec.demfrac[i])-1))
                # dec = abs(get_declination('',elec.demfrac))
                dec = abs(get_EG(elec.demfrac))
                # if dec == 2:
                #     continue
                xarr.append(np.mean(arr))
                yarr.append(dec)
        print stats.linregress(xarr,yarr)
        make_scatter('avg_margin_' + chm,xarr,yarr)
        
