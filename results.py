notokstates = ['AK','AL','AR','CO','CT','DE','HI','ID','IA','KS','KY','LA','ME','MS','MT','NE','NV','NH','NM','ND',\
               'OK','OR','RI','SC','SD','UT','VT','WV','WY','MA']

simlist = ['NC','VA','OH','WI','MI','FL','NJ','PA','MO','IN','NY','MN','WA','TX','IL',\
           'TN','GA','MD','CA','AZ']

azlist = ['MD','NC','VA','IL','PA','OH','TX','NJ','WA','TN','FL','MI','CA','WI',\
          'GA','MO','NY','AZ','MN','IN']

# stlist = ['00','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',
#           'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',
#           'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'] 



def overlap_one(l1,l2):
    """ give sequence of overlap cardinalities 
        if not forward then work with reversed lists
    """
    ans = []
    for i in range(1,len(l1)+1):
        # print l1[:i],l2[:i]
        ans.append(len(Set(l1[:i]).intersection(Set(l2[:i]))))

    return ans

def overlap(l1,l2,forward=True):
    """ give sequence of overlap cardinalities 
        if not forward then work with reversed lists
    """
    ans = []
    if forward:
        for i in range(1,len(l1)+1):
            print l1[:i],l2[:i]
            ans.append(len(Set(l1[:i]).intersection(Set(l2[:i]))))
    else:
        for i in range(1,len(l1)+1):
            ans.append(len(Set(l1[-i:]).intersection(Set(l2[-i:]))))

    return ans

def triple_overlap(l1,l2,forward):
    """ give sequence of overlap cardinalities 
        if not forward then work with reversed lists
    """
    ans = []
    if forward:
        for i in range(1,len(l1)+1):
            tmp = 0
            # print "arg: ",l1[:i],l2[:i]
            for j in range(i):
                # print "blah: ",l1[j][:2],l2[:i]
                if l1[j][:2] in l2[:i]:
                    tmp += 1
            ans.append(tmp)
            # ans.append(len(Set(l1[:i]).intersection(Set(l2[:i]))))
                # else:
        # for i in range(1,len(l1)+1):
            # ans.append(len(Set(l1[-i:]).intersection(Set(l2[-i:]))))

            # print "retu: ",len(l1),len(l2),len(ans)
    return ans

def overlap_both(l1,l2):
    """ return bot overlap sequences
    """
    return overlap(l1,l2,True),overlap(l1,l2,False)

def overlap_triple(l1,l2):
    """ return bot overlap sequences
    """
    return triple_overlap(l1,l2,True)

def compare_overlap_sequences(elections,yr):
    """ see how the rankings compare to other methods
        hmm - how do i deal with states whose number of seats has changed over time....
    """
    l1 = get_angle_sequence(elections,yr,True)
    l2 = get_gap_sequence(elections,yr,0,True)
    l3 = get_gap_sequence(elections,yr,1,True)
    l4 = get_gap_sequence(elections,yr,8,True)
    # lsim = get_simlist()
    laz = get_azlist()

    plt.figure(figsize=(8,8))
    plt.axis([0,21,0,21])

    legs = []
    o1,o2 = overlap_both(l1,laz)
    tmp, = plt.plot([i for i in range(22)],o1,label='Declination')
    legs.append(tmp)

    o1,o2 = overlap_both(l2,laz)
    tmp, = plt.plot([i for i in range(22)],o1,label='$\mathrm{Gap}_0$',linestyle='dotted')
    legs.append(tmp)

    o1,o2 = overlap_both(l3,laz)
    tmp, = plt.plot([i for i in range(22)],o1,label='$\mathrm{Gap}_1$',linestyle='dashed')
    legs.append(tmp)

    # o1,o2 = overlap_both(l4,laz)
    # tmp, = plt.plot([i for i in range(22)],o1,label='$\mathrm{Gap}_3$')
    # legs.append(tmp)

    plt.legend(handles=legs,loc='upper left')

    plt.savefig('/home/gswarrin/research/gerrymander/pics/overlap-az-triple')
    plt.close()

def compare_avg_overlap_sequences(elections,yr):
    """ see how the rankings compare to other methods
        hmm - how do i deal with states whose number of seats has changed over time....
    """
    l1 = get_avg_angle_sequence(elections,yr)
    l2 = get_avg_alpha_sequence(elections,yr,0)
    l3 = get_avg_alpha_sequence(elections,yr,1)
    l4 = get_avg_alpha_sequence(elections,yr,8)
    # lsim = get_simlist()
    laz = get_azlist()

    plt.figure(figsize=(8,8))
    # plt.axis([0,len(l1),0,len(l1)])

    print l1
    print l2
    print l3

    print len(l1),len(l2),len(l3),len(l4)
    legs = []
    o1 = overlap_one(l1,laz)
    tmp, = plt.plot([i+1 for i in range(len(l1))],o1,label='Declination')
    legs.append(tmp)

    o1 = overlap_one(l2,laz)
    tmp, = plt.plot([i+1 for i in range(len(l1))],o1,label='$\mathrm{Gap}_0$',linestyle='dotted')
    legs.append(tmp)

    o1 = overlap_one(l3,laz)
    tmp, = plt.plot([i+1 for i in range(len(l1))],o1,label='$\mathrm{Gap}_1$',linestyle='dashed')
    legs.append(tmp)

    # o1,o2 = overlap_both(l4,laz)
    # tmp, = plt.plot([i for i in range(22)],o1,label='$\mathrm{Gap}_3$')
    # legs.append(tmp)

    plt.legend(handles=legs,loc='upper left')

    plt.savefig('/home/gswarrin/research/gerrymander/pics/overlap-az-triple-avg')
    plt.close()

def get_avg_angle_sequence(elections,yr):
    """
    """
    ans = []
    d = dict()
    for elec in elections.values():
        if int(elec.yr) >= int(yr) and elec.chamber == '11' and elec.state not in notokstates:
            ang = find_angle(elec.state,elec.demfrac)
            if ang > -2: # don't want to include fake ones!
                if elec.state not in d.keys():
                    d[elec.state] = [abs(ang)]
                else:
                    d[elec.state].append(abs(ang))
    for k in d.keys():
        ans.append([np.mean(d[k]),k])
    ans.sort(key=lambda x: x[0])
    ans = ans[::-1] # reverse so in decreasing order
    return [x[1] for x in ans]

def get_avg_alpha_sequence(elections,yr,alpha):
    """
    """
    ans = []
    d = dict()
    for elec in elections.values():
        if int(elec.yr) >= int(yr) and elec.chamber == '11' and elec.state not in notokstates:
            gap = compute_alpha_curve(elec.demfrac,alpha)
            if elec.state not in d.keys():
                d[elec.state] = [abs(gap)]
            else:
                d[elec.state].append(abs(gap))
    for k in d.keys():
        ans.append([np.mean(d[k]),k])
    ans.sort(key=lambda x: x[0])
    ans = ans[::-1] # reverse so in decreasing order
    return [x[1] for x in ans]

def get_angle_sequence(elections,yr,doabs=False):
    """
    """
    ans = []
    for elec in elections.values():
        if elec.yr == yr and elec.chamber == '11' and elec.state not in notokstates:
            ang = find_angle(elec.state,elec.demfrac)
            if doabs:
                ans.append([abs(ang),elec.state])
            else:
                ans.append([ang,elec.state])
    ans.sort(key=lambda x: x[0])
    ans = ans[::-1] # reverse so in decreasing order
    return [x[1] for x in ans]

def get_gap_sequence(elections,yr,alpha,doabs=False):
    """
    """
    ans = []
    for elec in elections.values():
        if elec.yr == yr and elec.chamber == '11' and elec.state not in notokstates:
            egap = compute_alpha_curve(elec.demfrac,alpha)
            if doabs:
                ans.append([abs(egap),elec.state])
            else:
                ans.append([egap,elec.state])
    ans.sort(key=lambda x: x[0])
    ans = ans[::-1] # reverse so in decreasing order
    return [x[1] for x in ans]

def get_simlist():
    return simlist

def get_azlist():
    return azlist

def compare_allyrs_overlap_sequences(elections,yr):
    """ see how the rankings compare to other methods
        hmm - how do i deal with states whose number of seats has changed over time....
    """
    l1 = []
    l2 = []
    l3 = []
    for elec in elections.values():
        if int(elec.yr) >= int(yr) and elec.chamber == '11' and elec.state not in notokstates:
            l1.append([abs(find_angle(elec.state,elec.demfrac)),elec.state + elec.yr[-2:]])
            l2.append([abs(compute_alpha_curve(elec.demfrac,0)),elec.state + elec.yr[-2:]])
            l3.append([abs(compute_alpha_curve(elec.demfrac,1)),elec.state + elec.yr[-2:]])
    l1.sort(key=lambda x: x[0])
    l2.sort(key=lambda x: x[0])
    l3.sort(key=lambda x: x[0])
    l1 = l1[::-1]
    l2 = l2[::-1]
    l3 = l3[::-1]
    l1 = [x[1] for x in l1]
    l2 = [x[1] for x in l2]
    l3 = [x[1] for x in l3]

    # lsim = get_simlist()
    tmplaz = get_azlist()
    laz = []
    for x in tmplaz:
        laz.append(x) # + '12')
        laz.append(x) # + '14')
        laz.append(x) # + '16')

    plt.figure(figsize=(8,4))
    plt.axis([0,len(laz)-1,0,len(laz)-1])

    print sorted(l1)
    print sorted(l2)
    print sorted(l3)
    print sorted(laz)
    print len(l1),len(l2),len(l3),len(laz)
    legs = []
    o1 = overlap_triple(l1,laz)
    print "o1: ",len(o1)
    tmp, = plt.plot([i for i in range(len(laz))],o1,label='Declination')
    legs.append(tmp)

    o1 = overlap_triple(l2,laz)
    tmp, = plt.plot([i for i in range(len(laz))],o1,label='$\mathrm{Gap}_0$',linestyle='dashed')
    legs.append(tmp)

    o1 = overlap_triple(l3,laz)
    tmp, = plt.plot([i for i in range(len(laz))],o1,label='$\mathrm{Gap}_1$',linestyle='dotted')
    legs.append(tmp)

    # o1,o2 = overlap_both(l4,laz)
    # tmp, = plt.plot([i for i in range(22)],o1,label='$\mathrm{Gap}_3$')
    # legs.append(tmp)

    plt.legend(handles=legs,loc='upper left')
    plt.xlabel('Depth')
    plt.ylabel('Overlap')
    plt.tight_layout()

    plt.savefig('/home/gswarrin/research/gerrymander/pics/combine-overlap-az-triple')
    plt.close()


def make_cong_histo(elections):
    totcong = 0
    totstate = 0
    congang = []
    stateang = []
    for elec in Melections.values():
        if elec.Ndists >= 8 and elec.chamber == '11' and int(elec.yr) >= 1972:
            totcong += 1
            fang = find_angle(elec.state,elec.demfrac)
            if fang > -2:
                congang.append(fang)
        if elec.Ndists >= 8 and elec.chamber == '9' and \
           (elec.yr not in Mmmd.keys() or elec.state not in Mmmd[elec.yr]):
            totstate += 1
            fang = find_angle(elec.state,elec.demfrac)
            if fang > -2:
                stateang.append(fang)
    print "Tot cong: ",totcong
    print "Tot state: ",totstate
    print "Mean: %.3f StdDev: %.3f" % (np.mean(congang),np.std(congang))
    print "Mean: %.3f StdDev: %.3f" % (np.mean(stateang),np.std(stateang))

    fig, axes = plt.subplots(1,2, figsize=(8,3)) # , sharex=True, sharey=True)
    axes = axes.ravel()
    axes[0].set_axis_bgcolor('none')
    axes[1].set_axis_bgcolor('none')

    n, bins, patches = axes[0].hist(congang, 25, facecolor='g', alpha=0.75)
    n, bins, patches = axes[1].hist(stateang, 25, facecolor='g', alpha=0.75)
    axes[0].set_xlabel("Declination")
    axes[1].set_xlabel("Declination")
    axes[0].set_xticks([-0.5,0,0.5])
    axes[1].set_xticks([-0.5,0,0.5])
    axes[0].set_yticks([0,15,30,45,60])
    axes[1].set_yticks([0,25,50,75,100])
    axes[0].set_title("Congress")
    axes[1].set_title("State")
    axes[0].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/cong-state-hist')
    plt.close()

def make_cong_deltaN_histo(elections):
    totcong = 0
    totstate = 0
    congang = []
    stateang = []
    congangN = []
    stateangN = []
    minyr = 2002
    for elec in elections.values():
        if elec.Ndists >= 8 and elec.chamber == '11' and int(elec.yr) >= minyr:
            totcong += 1
            fang = find_angle(elec.state,elec.demfrac)
            if abs(fang) < 2:
                congang.append(fang)
                congangN.append(fang*elec.Ndists*1.0/2)
        if elec.Ndists >= 8 and elec.chamber == '9' and int(elec.yr) >= minyr and \
           (elec.yr not in Mmmd.keys() or elec.state not in Mmmd[elec.yr]):
            totstate += 1
            fang = find_angle(elec.state,elec.demfrac)
            if abs(fang) < 2:
                stateang.append(fang)
                stateangN.append(fang*elec.Ndists*1.0/2)
    print "Tot cong: ",totcong
    print "Tot state: ",totstate
    print "Cong Median: %.3f StdDev: %.3f" % (np.median(congang),np.std(congang))
    print "State Mean: %.3f StdDev: %.3f" % (np.median(stateang),np.std(stateang))
    print "Cong delta_N Mean: %.3f StdDev: %.3f" % (np.median(congangN),np.std(congangN))
    print "State delta_N Mean: %.3f StdDev: %.3f" % (np.median(stateangN),np.std(stateangN))

    fig, axes = plt.subplots(2,2, figsize=(8,8)) # sharex=True, sharey=True)
    axes = axes.ravel()
    axes[0].set_axis_bgcolor('none')
    axes[1].set_axis_bgcolor('none')
    axes[2].set_axis_bgcolor('none')
    axes[3].set_axis_bgcolor('none')

    n, bins, patches = axes[0].hist(congang, 25, facecolor='g', alpha=0.75)
    n, bins, patches = axes[1].hist(stateang, 25, facecolor='g', alpha=0.75)
    n, bins, patches = axes[2].hist(congangN, 25, facecolor='g', alpha=0.75)
    n, bins, patches = axes[3].hist(stateangN, 25, facecolor='g', alpha=0.75)
    axes[0].set_xlabel("Declination")
    axes[1].set_xlabel("Declination")
    axes[0].set_xticks([-0.5,0,0.5])
    axes[1].set_xticks([-0.5,0,0.5])
    axes[0].set_yticks([0,15,30,45,60])
    axes[1].set_yticks([0,25,50,75,100])
    axes[0].set_title("Congress")
    axes[1].set_title("State")
    axes[0].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/cong-state-hist-deltaN')
    plt.close()

    
def make_heatmap_combine(elections,mmd):
    # import the data directly into a pandas dataframe
    # nba = pd.read_csv("http://datasets.flowingdata.com/ppg2008.csv", index_col='Name  ')
    # remove index title
    # nba.index.name = ""
    # normalize data columns
    # nba_norm = (nba - nba.mean()) / (nba.max() - nba.min())

    totcong = 0
    totstate = 0
    congang = []
    stateang = []
    congangN = []
    stateangN = []
    allyrs = [(1972+2*j) for j in range(23)]
    numbins = 8
    congtmparr = [[] for i in range(5)]
    statetmparr = [[] for i in range(5)]
    congNtmparr = [[] for i in range(5)]
    stateNtmparr = [[] for i in range(5)]
    minyr = 1972
    for elec in elections.values():
        if elec.Ndists >= 4 and elec.chamber == '11' and int(elec.yr) >= minyr and int(elec.yr)%2 == 0:
            totcong += 1
            fang = find_angle(elec.state,elec.demfrac)
            if abs(fang) < 2:
                congtmparr[allyrs.index(int(elec.yr))/5].append(fang)
                congNtmparr[allyrs.index(int(elec.yr))/5].append(fang*elec.Ndists*1.0/2)
                congang.append(fang)
                congangN.append(fang*elec.Ndists*1.0/2)
        if elec.Ndists >= 4 and elec.chamber == '9' and int(elec.yr) >= minyr and \
           (elec.yr not in Mmmd.keys() or elec.state not in Mmmd[elec.yr]) and int(elec.yr)%2 == 0:
            totstate += 1
            fang = find_angle(elec.state,elec.demfrac)
            if abs(fang) < 2:
                statetmparr[allyrs.index(int(elec.yr))/5].append(fang)
                stateNtmparr[allyrs.index(int(elec.yr))/5].append(fang*elec.Ndists*1.0/2)
                stateang.append(fang)
                stateangN.append(fang*elec.Ndists*1.0/2)

    numbins = 12
    bins = np.linspace(-0.8,0.8,numbins+1)
    sub_make_heatmap('heat-cong',congtmparr,allyrs,bins)
    numbins = 12
    bins = np.linspace(-0.7,0.7,numbins+1)
    sub_make_heatmap('heat-state',statetmparr,allyrs,bins)

    numbins = 12
    bins = np.linspace(-8,8,numbins+1)
    sub_make_heatmap('heat-congN',congNtmparr,allyrs,bins)
    numbins = 12
    bins = np.linspace(-42,42,numbins+1)
    sub_make_heatmap('heat-stateN',stateNtmparr,allyrs,bins)
