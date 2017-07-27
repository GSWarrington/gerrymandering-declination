def beginning_end_scatter(elections,states):
    """
    """
    arr1 = []
    arr2 = []
    for yr in ['1972','1982','1992','2002']:
        for st in states:
            key1 = '_'.join([yr,st,'11'])
            key2 = '_'.join([str(int(yr)+8),st,'11'])
            if key1 in elections.keys() and key2 in elections.keys():
                elec1 = elections[key1]
                elec2 = elections[key2]
                fa1 = find_angle(elec1.state,elec1.demfrac)
                fa2 = find_angle(elec2.state,elec2.demfrac)
                # fa1 = compute_alpha_curve(elec1.demfrac,0)
                # fa2 = compute_alpha_curve(elec2.demfrac,0)
                if elec1.Ndists >= 10 and fa1 > -2 and fa2 > -2:
                    arr1.append(fa1)
                    arr2.append(fa2)
    print "total: ",len(arr1)
    make_scatter('begend',arr1,arr2)
    abs1 = 0
    tot1 = 0
    abs2 = 0
    tot2 = 0
    for i in range(len(arr1)):
        st1 = 0.1
        en1 = 0.1
        st2 = 0.2
        en2 = 0.0
        if abs(arr1[i]) >= st1:
            tot1 += 1
            if (arr1[i] >= st1 and arr2[i] >= en1) or \
               (arr1[i] <= -st1 and arr2[i] <= -en1):
                abs1 += 1
        if abs(arr1[i]) >= st2:
            tot2 += 1
            if (arr1[i] >= st2 and arr2[i] >= en2) or \
               (arr1[i] <= -st2 and arr2[i] <= -en2):
                abs2 += 1
    print "abs1: %.3f tot1: %.3f: frac: %.3f" % (abs1,tot1,abs1*1.0/tot1)
    print "abs2: %.3f tot2: %.3f: frac: %.3f" % (abs2,tot2,abs2*1.0/tot2)    
