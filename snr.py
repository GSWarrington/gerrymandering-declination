def compare_snr(elecs,gval):
    """ compare snr of angle versus gap
    """
    yrs = ['1974','1976','1978','1980','1982','1984','1986','1988','1990',\
            '1992','1994','1996','1998','2000','2002','2004','2006','2008','2010']
    xvals = [gval] # -4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,5,10] # np.linspace(-10,10,50)
    rats = [[[] for i in range(len(yrs))] for k in range(len(xvals))]
    races = [0 for kk in range(len(yrs))]
    gaps = [[0 for ii in range(len(yrs))] for kk in range(len(xvals))]
    haps = [[0 for ii in range(len(yrs))] for kk in range(len(xvals))]
    for st in Mstates:
        for i in range(len(yrs)):
            prior_yr = str(int(yrs[i])-2)
            myid1 = '_'.join([prior_yr,st,'11'])
            myid2 = '_'.join([yrs[i],st,'11'])
            if myid1 in elecs.keys() and myid2 in elecs.keys() and \
                (yrs[i] not in Mmmd.keys() or st not in Mmmd[yrs[i]]) and \
                (prior_yr not in Mmmd.keys() or st not in Mmmd[prior_yr]) and \
                elecs[myid1].Ndists >= 8:
                isokay = False
                for j in range(len(xvals)):
                    ans1 = get_tau_gap(elecs[myid1].demfrac,xvals[j])
                    ans2 = get_tau_gap(elecs[myid2].demfrac,xvals[j])
                    ans3 = get_declination(elecs[myid1].state,elecs[myid1].demfrac)
                    ans4 = get_declination(elecs[myid2].state,elecs[myid2].demfrac)
                    # print "%d %.2f %.2f %.2f %.2f" % (i,xvals[j],abs(ans1-ans2),ans1,ans2)
                    if ans3 > -2 and ans4 > -2:
                        isokay = True
                        gaps[j][i] += abs(ans1-ans2)
                        haps[j][i] += abs(ans3-ans4)
                if isokay:
                    races[i] += 1
        # for i in range(len(yrs)):
        #     rats[j][i].append(1.0*gaps[j][i]/gaps[j][4])
        #     print st,j,i,rats[j][i]
                    
    for j in range(len(xvals)): # gap type
        for i in range(len(yrs)): # year
            gaps[j][i] /= races[i]
            haps[j][i] /= races[i]
        # print gaps[j][i],
    print

    for k in [0]: # [0,1,2,3,5,6,7,8]:
        intra_cycles = [0 for i in range(len(xvals))]
        inter_cycles = [0 for i in range(len(xvals))]
        ang_intra_cycles = [0 for i in range(len(xvals))]
        ang_inter_cycles = [0 for i in range(len(xvals))]
        intra_med_cycles = [[] for i in range(len(xvals))]
        inter_med_cycles = [[] for i in range(len(xvals))]
        for i in range(len(xvals)): # gap type
            for j in [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18]:
                intra_cycles[i] += gaps[i][j]
                ang_intra_cycles[i] += haps[i][j]
                intra_med_cycles[i].append(gaps[i][j])
                ## intra_cycles[i] = np.mean(rats[i][j])
            intra_cycles[i] /= 16
            ang_intra_cycles[i] /= 16
            intra_med_cycles[i] = np.median(intra_med_cycles[i])
            inter_cycles[i] = (gaps[i][4]+gaps[i][9]+gaps[i][14])/3
            ang_inter_cycles[i] = (haps[i][4]+haps[i][9]+haps[i][14])/3
            inter_med_cycles[i] = np.median([gaps[i][4],gaps[i][9],gaps[i][14]])
        
        print "ang: %.3f %.3f %.3f" % (ang_inter_cycles[0]/ang_intra_cycles[0],ang_inter_cycles[0],ang_intra_cycles[0])
        print "gap: %.3f %.3f %.3f" % (inter_cycles[0]/intra_cycles[0],inter_cycles[0],intra_cycles[0])
        # make_scatter_labels('corr-st7202avg-snr' + str(k) + \
        # '.png',intra_cycles,inter_cycles,xvals) 
        # make_scatter_labels('corr-stmed7202avg-snr' + str(k) + \
        # '.png',intra_med_cycles,inter_med_cycles,xvals) 
        # make_scatter_labels('rats-snr' + str(k) + \
        # '.png',xvals,intra_cycles,xvals) 

def plot_snr(elecs):
    """ plot signal to noise over varying alpha
    """
    yrs = ['1974','1976','1978','1980','1982','1984','1986','1988','1990',\
            '1992','1994','1996','1998','2000','2002','2004','2006','2008','2010']
    xvals = [-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,5,10] # np.linspace(-10,10,50)
    rats = [[[] for i in range(len(yrs))] for k in range(len(xvals))]
    races = [0 for kk in range(len(yrs))]
    gaps = [[0 for ii in range(len(yrs))] for kk in range(len(xvals))]
    for st in Mstates:
        for i in range(len(yrs)):
            prior_yr = str(int(yrs[i])-2)
            myid1 = '_'.join([prior_yr,st,'9'])
            myid2 = '_'.join([yrs[i],st,'9'])
            if myid1 in elecs.keys() and myid2 in elecs.keys() and \
                st not in Mmmd[yrs[i]] and st not in Mmmd[prior_yr]:
                for j in range(len(xvals)):
                    ans1 = get_tau_gap(elecs[myid1].demfrac,xvals[j])
                    ans2 = get_tau_gap(elecs[myid2].demfrac,xvals[j])
                    # print "%d %.2f %.2f %.2f %.2f" % (i,xvals[j],abs(ans1-ans2),ans1,ans2)
                    gaps[j][i] += abs(ans1-ans2)
                races[i] += 1
        # for i in range(len(yrs)):
        #     rats[j][i].append(1.0*gaps[j][i]/gaps[j][4])
        #     print st,j,i,rats[j][i]
                    
    for j in range(len(xvals)): # gap type
        for i in range(len(yrs)): # year
            gaps[j][i] /= races[i]
        # print gaps[j][i],
    print

    for k in [0]: # [0,1,2,3,5,6,7,8]:
        intra_cycles = [0 for i in range(len(xvals))]
        inter_cycles = [0 for i in range(len(xvals))]
        intra_med_cycles = [[] for i in range(len(xvals))]
        inter_med_cycles = [[] for i in range(len(xvals))]
        for i in range(len(xvals)): # gap type
            for j in [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18]:
                intra_cycles[i] += gaps[i][j]
                intra_med_cycles[i].append(gaps[i][j])
                ## intra_cycles[i] = np.mean(rats[i][j])
            intra_cycles[i] /= 16
            intra_med_cycles[i] = np.median(intra_med_cycles[i])
            inter_cycles[i] = (gaps[i][4]+gaps[i][9]+gaps[i][14])/3
            inter_med_cycles[i] = np.median([gaps[i][4],gaps[i][9],gaps[i][14]])
        
        make_scatter_labels('corr-st7202avg-snr' + str(k) + \
        '.png',intra_cycles,inter_cycles,xvals) 
        make_scatter_labels('corr-stmed7202avg-snr' + str(k) + \
        '.png',intra_med_cycles,inter_med_cycles,xvals) 
        # make_scatter_labels('rats-snr' + str(k) + \
        # '.png',xvals,intra_cycles,xvals) 

def compare_snr_corr(elecs,myplt):
    """ plot signal to noise over varying alpha - base on correlation 
        rather than average absolute-value change
    """
    yrs = ['1974','1976','1978','1980','1982','1984','1986','1988','1990',\
            '1992','1994','1996','1998','2000','2002','2004','2006','2008','2010']
    xvals = [-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3] # np.linspace(-10,10,50)
    gaps = [[[] for ii in range(len(yrs))] for kk in range(len(xvals))]
    anscorr = [[0 for ii in range(len(yrs))] for kk in range(len(xvals))]
#     anscorr = []
    for st in Mstates:
        for i in range(len(yrs)):
            prior_yr = str(int(yrs[i])-2)
            myid1 = '_'.join([prior_yr,st,'11'])
            myid2 = '_'.join([yrs[i],st,'11'])
            if myid1 in elecs.keys() and myid2 in elecs.keys() and \
                st not in Mmmd[yrs[i]] and st not in Mmmd[prior_yr]:
                # print elecs[myid1].demfrac
                for j in range(len(xvals)):
                    ans1 = get_tau_gap(elecs[myid1].demfrac,xvals[j])
                    ans2 = get_tau_gap(elecs[myid2].demfrac,xvals[j])
                    # ans1 = get_declination('ll',elecs[myid1].demfrac)
                    # ans2 = get_declination('mm',elecs[myid2].demfrac)
                    # if ans1 != None and ans2 != None:
                    # print "%d %.2f %.2f %.2f %.2f" % \
                    # (i,xvals[j],abs(ans1-ans2),ans1,ans2)
                    # if j == 3 and i == 16:
                    #     print "three: ",st,ans1,ans2
                    gaps[j][i].append([ans1,ans2])
                    
    for j in range(len(xvals)): # gap type
        # make_scatter('hmm' + str(j),[x[0] for x in gaps[j][9]],\
        #                             [x[1] for x in gaps[j][9]])
        for i in range(len(yrs)):
            # print "here: ",j,i,gaps[j][i]
            a1,p1 = stats.pearsonr([x[0] for x in gaps[j][i]],\
            [x[1] for x in gaps[j][i]])
            # print "pear: ",j,i,a1,p1
            anscorr[j][i] = a1
    #for j in range(len(xvals)):
    #     for i in range(len(yrs)):
     #       print yrs[i],xvals[j],anscorr[j][i]

    for k in [0]: # range(len(xvals)): # gap type
        intra_med_cycles = [0 for i in range(len(xvals))]
        inter_med_cycles = [0 for i in range(len(xvals))]
        rat = [0 for i in range(len(xvals))]
        for i in range(len(xvals)): # gap type
            # for j in [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18]:
                # intra_cycles[i] += anscorr[i][j]
            #     intra_med_cycles[i].append(anscorr[i][j])
                ## intra_cycles[i] = np.mean(rats[i][j])
            # intra_cycles[i] /= 16
            # intra_med_cycles[i] = np.mean(intra_med_cycles[i])
            # inter_cycles[i] = (anscorr[i][4]+anscorr[i][9]+anscorr[i][14])/3
            inter_med_cycles[i] = np.mean([anscorr[i][4],anscorr[i][9],anscorr[i][14]])
            intra_med_cycles[i] = np.mean([anscorr[i][6],anscorr[i][11],anscorr[i][16]])
            # intra_med_cycles[i] = np.mean([anscorr[i][8],anscorr[i][12],anscorr[i][17]])
            # rat[i] = inter_med_cycles[i]/intra_med_cycles[i]
            
        # make_scatter_labels('alphas-7202avg-snr' + str(k) + \
        # '.png',intra_cycles,inter_cycles,xvals) 
        # make_scatter_labels('snrcorrblah' + str(k) + \
        # '.png',intra_med_cycles,inter_med_cycles,xvals) 
        # make_scatter_labels('snrrat' + str(k) + \
        # '.png',xvals,rat,xvals) # intra_med_cycles,inter_med_cycles,xvals) 
        # make_scatter_labels('rats-snr' + str(k) + \
        # '.png',xvals,intra_cycles,xvals) 
    
    # myplt.figure(figsize=(4,4))
    myplt.axis([0.3,0.85,0.3,0.85])
    # print "blah"
    print intra_med_cycles
    print inter_med_cycles
    myplt.scatter(intra_med_cycles,inter_med_cycles) # ,markersize=20)
    for i,txt in enumerate(xvals):
        if i in [3,5,7,9]: 
            tmpstr = "%.1f" % (txt)
        else:
            tmpstr = "%d" % (txt)
        myplt.annotate(tmpstr,(intra_med_cycles[i],inter_med_cycles[i]),\
            verticalalignment='bottom')
    # plt.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    myplt.plot([0.3,1], [0.3,1], color='k', linestyle='-', linewidth=2)
    myplt.plot([0.7122-0.3,0.7122+0.1],[0.5671-0.3,0.5671+0.1],linestyle='--')
    # myplt.xlabel('intra-cycle correlation')
    # myplt.ylabel('inter-cycle correlation')
    # plt.plot(
    myplt.grid(True)
    # plt.axhline(0.5, color='red', linestyle='dashed', linewidth=2)
    # return plt
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/snr-pic-11')

def create_anova_csv(elecs,chamber,useangle=True,gval=1):
    """ plot signal to noise over varying alpha - base on correlation 
        rather than average absolute-value change
    """
    yrs = ['1974','1976','1978','1980','1982','1984','1986','1988','1990',\
            '1992','1994','1996','1998','2000','2002','2004','2006','2008','2010','2012','2014']

#    if useangle:
#        f = open('/home/gswarrin/research/gerrymander/anova-angle-geq8-' + chamber + '14.csv','w')
#    else:
#        f = open('/home/gswarrin/research/gerrymander/anova-gap-' + chamber + '14-' + str(gval) + '.csv','w')
    f = open('/home/gswarrin/research/gerrymander/sumhalf.csv','w')

    f.write("CYCLE,YEAR,VAL\n")
    # for visualizing intra/inter-correlations
    cnt = 1
    for st in Mstates:
        for i in range(len(yrs)):
            prior_yr = str(int(yrs[i])-2)
            myid1 = '_'.join([prior_yr,st,chamber])
            myid2 = '_'.join([yrs[i],st,chamber])
            if myid1 in elecs.keys() and myid2 in elecs.keys() and \
                (yrs[i] not in Mmmd.keys() or st not in Mmmd[yrs[i]]) and \
                (prior_yr not in Mmmd.keys() or st not in Mmmd[prior_yr]) and \
                elecs[myid1].Ndists >= 8:
                # if useangle:
                ans1 = get_declination(elecs[myid1].state,elecs[myid1].demfrac)
                ans2 = get_declination(elecs[myid2].state,elecs[myid2].demfrac)
                # else:
                ans3 = get_tau_gap(elecs[myid1].demfrac,gval)
                ans4 = get_tau_gap(elecs[myid2].demfrac,gval)
                if i in [4,9,14,19]:
                    f.write("INTER,%s,%.3f\n" % (yrs[i],abs(ans1-ans2)+abs(ans3-ans4)))
                else:
                    f.write("INTRA,%s,%.3f\n" % (yrs[i],abs(ans1-ans2)+abs(ans3-ans4)))
                cnt += 1
    f.close()

def compare_corr_coeff(elecs,chamber,gval):
    """ plot signal to noise over varying alpha - base on correlation 
        rather than average absolute-value change
    """
    yrs = ['1974','1976','1978','1980','1982','1984','1986','1988','1990',\
            '1992','1994','1996','1998','2000','2002','2004','2006','2008','2010','2012','2014']
    gap_intra = []
    gap_inter = []
    ang_intra = []
    ang_inter = []

    # for visualizing intra/inter-correlations
    gap_plt = []
    ang_plt = []
    for st in Mstates:
        for i in range(len(yrs)):
            prior_yr = str(int(yrs[i])-2)
            myid1 = '_'.join([prior_yr,st,chamber])
            myid2 = '_'.join([yrs[i],st,chamber])
            if myid1 in elecs.keys() and myid2 in elecs.keys() and \
                (yrs[i] not in Mmmd.keys() or st not in Mmmd[yrs[i]]) and \
                (prior_yr not in Mmmd.keys() or st not in Mmmd[prior_yr]) and \
                elecs[myid1].Ndists >= 8:
                ans1 = get_tau_gap(elecs[myid1].demfrac,gval)
                ans2 = get_tau_gap(elecs[myid2].demfrac,gval)
                ans3 = get_declination(elecs[myid1].state,elecs[myid1].demfrac)
                ans4 = get_declination(elecs[myid2].state,elecs[myid2].demfrac)
                # print "%d %.2f %.2f %.2f %.2f" % (i,xvals[j],abs(ans1-ans2),ans1,ans2)
                if ans3 > -2 and ans4 > -2:
                    print yrs[i],elecs[myid1].state,elecs[myid1].Ndists,ans3,ans4,abs(ans3-ans4)
                    if i in [4,9,14,19]:
                        gap_inter.append(abs(ans1-ans2))
                        ang_inter.append(abs(ans3-ans4))
                        gap_plt.append([0.2+i*1.0/4,abs(ans1-ans2)])
                        ang_plt.append([0.2+i*1.0/4,abs(ans3-ans4)+abs(ans1-ans2)])
                    else:
                        gap_intra.append(abs(ans1-ans2))
                        ang_intra.append(abs(ans3-ans4))
                        gap_plt.append([6.2+i*1.0/4,abs(ans1-ans2)])
                        ang_plt.append([6.2+i*1.0/4,abs(ans3-ans4)])

    plt.figure(figsize=(8,8))
    plt.axis([0,19,0,1])
    # plt.scatter([x[0] for x in gap_plt],[x[1] for x in gap_plt],color='red')
    # plt.savefig('gapcorr')
    # plt.close()

    # plt.figure(figsize=(8,8))
    # plt.axis([0,19,0,1])
    plt.scatter([x[0] for x in ang_plt],[x[1] for x in ang_plt],color='blue')
    plt.savefig('bothcorr')
    plt.close()

    gap_all = gap_inter + gap_intra
    ang_all = ang_inter + ang_intra

    gap_y1bar = np.mean(gap_intra)
    gap_y2bar = np.mean(gap_inter)
    gap_allbar = np.mean(gap_all)

    ang_y1bar = np.mean(ang_intra)
    ang_y2bar = np.mean(ang_inter)
    ang_allbar = np.mean(ang_all)

    n1 = len(gap_intra)
    n2 = len(gap_inter)
    
    print "ang y1:%.3f y2:%.3f all:%.3f" % (ang_y1bar,ang_y2bar,ang_allbar)
    print "gap y1:%.3f y2:%.3f all:%.3f" % (gap_y1bar,gap_y2bar,gap_allbar)
    gap_denom = sum([pow(x-gap_allbar,2) for x in gap_all])
    gap_eta2 = (n1*pow(gap_y1bar-gap_allbar,2) + n2*pow(gap_y2bar-gap_allbar,2))/gap_denom

    ang_denom = sum([pow(x-ang_allbar,2) for x in ang_all])
    ang_eta2 = (n1*pow(ang_y1bar-ang_allbar,2) + n2*pow(ang_y2bar-ang_allbar,2))/ang_denom

    print n1,n2,gap_denom,ang_denom,gap_eta2
    print "ang: %.3f gap %.2f: %.3f " % (ang_eta2,gval,gap_eta2)

def anova_on_corr_coeff(elecs,chamber,gval):
    """ run an anova on correlation coefficients
    """
    yrs = ['1974','1976','1978','1980','1982','1984','1986','1988','1990',\
            '1992','1994','1996','1998','2000','2002','2004','2006','2008','2010','2012','2014']
    interyrs = ['1982','1992','2002','2012']
    intrayrs = ['1974','1976','1978','1980','1984','1986','1988','1990',\
            '1994','1996','1998','2000','2004','2006','2008','2010','2014']
    gap_intra = [[] for i in range(len(intrayrs))]
    gap_inter = [[] for i in range(len(interyrs))]
    ang_intra = [[] for i in range(len(intrayrs))]
    ang_inter = [[] for i in range(len(interyrs))]

    # for visualizing intra/inter-correlations
    for st in Mstates:
        for i in range(len(yrs)):
            prior_yr = str(int(yrs[i])-2)
            myid1 = '_'.join([prior_yr,st,chamber])
            myid2 = '_'.join([yrs[i],st,chamber])
            if myid1 in elecs.keys() and myid2 in elecs.keys() and \
                (yrs[i] not in Mmmd.keys() or st not in Mmmd[yrs[i]]) and \
                (prior_yr not in Mmmd.keys() or st not in Mmmd[prior_yr]) and \
                elecs[myid1].Ndists >= 10:
                ans1 = get_tau_gap(elecs[myid1].demfrac,gval)
                ans2 = get_tau_gap(elecs[myid2].demfrac,gval)
                ans3 = get_declination(elecs[myid1].state,elecs[myid1].demfrac)
                ans4 = get_declination(elecs[myid2].state,elecs[myid2].demfrac)
                # print "%d %.2f %.2f %.2f %.2f" % (i,xvals[j],abs(ans1-ans2),ans1,ans2)
                if ans3 > -2 and ans4 > -2:
                    # print yrs[i],elecs[myid1].state,elecs[myid1].Ndists,ans3,ans4,abs(ans3-ans4)
                    if i in [4,9,14,19]:
                        gap_inter[interyrs.index(yrs[i])].append([ans1,ans2])
                        ang_inter[interyrs.index(yrs[i])].append([ans3,ans4])
                    else:
                        gap_intra[intrayrs.index(yrs[i])].append([ans1,ans2])
                        ang_intra[intrayrs.index(yrs[i])].append([ans3,ans4])

    
    intracorr = []
    intercorr = []
    angintracorr = []
    angintercorr = []
    for i in range(len(gap_inter)):
        a1,p1 = stats.pearsonr([x[0] for x in gap_inter[i]],[x[1] for x in gap_inter[i]])
        intercorr.append(a1)
        a1,p1 = stats.pearsonr([x[0] for x in ang_inter[i]],[x[1] for x in ang_inter[i]])
        print interyrs[i],a1,p1
        angintercorr.append(a1)
    print "---"
    for i in range(len(gap_intra)):
        a1,p1 = stats.pearsonr([x[0] for x in gap_intra[i]],[x[1] for x in gap_intra[i]])
        intracorr.append(a1)
        a1,p1 = stats.pearsonr([x[0] for x in ang_intra[i]],[x[1] for x in ang_intra[i]])
        print intrayrs[i],a1,p1
        angintracorr.append(a1)
    print gval,stats.f_oneway(intracorr,intercorr)
    print "ang: ",stats.f_oneway(angintracorr,angintercorr)

    # print "ang: %.3f gap %.2f: %.3f " % (ang_eta2,gval,gap_eta2)

def plot_snr_corr(elecs,myplt):
    """ plot signal to noise over varying alpha - base on correlation 
        rather than average absolute-value change
    """
    yrs = ['1974','1976','1978','1980','1982','1984','1986','1988','1990',\
            '1992','1994','1996','1998','2000','2002','2004','2006','2008','2010']
    xvals = [-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3] # np.linspace(-10,10,50)
    gaps = [[[] for ii in range(len(yrs))] for kk in range(len(xvals))]
    anscorr = [[0 for ii in range(len(yrs))] for kk in range(len(xvals))]
#     anscorr = []
    for st in Mstates:
        for i in range(len(yrs)):
            prior_yr = str(int(yrs[i])-2)
            myid1 = '_'.join([prior_yr,st,'11'])
            myid2 = '_'.join([yrs[i],st,'11'])
            if myid1 in elecs.keys() and myid2 in elecs.keys() and \
                st not in Mmmd[yrs[i]] and st not in Mmmd[prior_yr]:
                # print elecs[myid1].demfrac
                for j in range(len(xvals)):
                    ans1 = get_tau_gap(elecs[myid1].demfrac,xvals[j])
                    ans2 = get_tau_gap(elecs[myid2].demfrac,xvals[j])
                    # ans1 = get_declination('ll',elecs[myid1].demfrac)
                    # ans2 = get_declination('mm',elecs[myid2].demfrac)
                    # if ans1 != None and ans2 != None:
                    # print "%d %.2f %.2f %.2f %.2f" % \
                    # (i,xvals[j],abs(ans1-ans2),ans1,ans2)
                    # if j == 3 and i == 16:
                    #     print "three: ",st,ans1,ans2
                    gaps[j][i].append([ans1,ans2])
                    
    for j in range(len(xvals)): # gap type
        # make_scatter('hmm' + str(j),[x[0] for x in gaps[j][9]],\
        #                             [x[1] for x in gaps[j][9]])
        for i in range(len(yrs)):
            # print "here: ",j,i,gaps[j][i]
            a1,p1 = stats.pearsonr([x[0] for x in gaps[j][i]],\
            [x[1] for x in gaps[j][i]])
            # print "pear: ",j,i,a1,p1
            anscorr[j][i] = a1
    #for j in range(len(xvals)):
    #     for i in range(len(yrs)):
     #       print yrs[i],xvals[j],anscorr[j][i]

    for k in [0]: # range(len(xvals)): # gap type
        intra_med_cycles = [0 for i in range(len(xvals))]
        inter_med_cycles = [0 for i in range(len(xvals))]
        rat = [0 for i in range(len(xvals))]
        for i in range(len(xvals)): # gap type
            # for j in [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18]:
                # intra_cycles[i] += anscorr[i][j]
            #     intra_med_cycles[i].append(anscorr[i][j])
                ## intra_cycles[i] = np.mean(rats[i][j])
            # intra_cycles[i] /= 16
            # intra_med_cycles[i] = np.mean(intra_med_cycles[i])
            # inter_cycles[i] = (anscorr[i][4]+anscorr[i][9]+anscorr[i][14])/3
            inter_med_cycles[i] = np.mean([anscorr[i][4],anscorr[i][9],anscorr[i][14]])
            intra_med_cycles[i] = np.mean([anscorr[i][6],anscorr[i][11],anscorr[i][16]])
            # intra_med_cycles[i] = np.mean([anscorr[i][8],anscorr[i][12],anscorr[i][17]])
            # rat[i] = inter_med_cycles[i]/intra_med_cycles[i]
            
        # make_scatter_labels('alphas-7202avg-snr' + str(k) + \
        # '.png',intra_cycles,inter_cycles,xvals) 
        # make_scatter_labels('snrcorrblah' + str(k) + \
        # '.png',intra_med_cycles,inter_med_cycles,xvals) 
        # make_scatter_labels('snrrat' + str(k) + \
        # '.png',xvals,rat,xvals) # intra_med_cycles,inter_med_cycles,xvals) 
        # make_scatter_labels('rats-snr' + str(k) + \
        # '.png',xvals,intra_cycles,xvals) 
    
    # myplt.figure(figsize=(4,4))
    myplt.axis([0.3,0.85,0.3,0.85])
    # print "blah"
    print intra_med_cycles
    print inter_med_cycles
    myplt.scatter(intra_med_cycles,inter_med_cycles) # ,markersize=20)
    for i,txt in enumerate(xvals):
        if i in [3,5,7,9]: 
            tmpstr = "%.1f" % (txt)
        else:
            tmpstr = "%d" % (txt)
        myplt.annotate(tmpstr,(intra_med_cycles[i],inter_med_cycles[i]),\
            verticalalignment='bottom')
    # plt.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    myplt.plot([0.3,1], [0.3,1], color='k', linestyle='-', linewidth=2)
    myplt.plot([0.7122-0.3,0.7122+0.1],[0.5671-0.3,0.5671+0.1],linestyle='--')
    # myplt.xlabel('intra-cycle correlation')
    # myplt.ylabel('inter-cycle correlation')
    # plt.plot(
    myplt.grid(True)
    # plt.axhline(0.5, color='red', linestyle='dashed', linewidth=2)
    # return plt
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/snr-pic-11')

# plot_snr(electionsa)
# plot_snr_corr(electionsa)



# # run through
# gaps = [[0 for i in range(9)] for k in range(14)]
# cnt = 0
# curyrs = []
# races = [0 for i in range(9)]
# for yr in ['1994','1996','1998','2000','2002','2004','2006','2008','2010']:
#     for st in ['WI']: # states:
#         prior_yr = str(int(yr)-2)
#         myid1 = '_'.join([yr,st,'9'])
#         myid2 = '_'.join([prior_yr,st,'9'])
#         if st not in mmd_dict[yr] and st not in mmd_dict[prior_yr] and \
#             myid1 in elections.keys() and myid2 in elections.keys():
#             gaps1 = elections[myid1].gaps
#             gaps2 = elections[myid2].gaps
#             races[cnt] += 1
#             print elections[myid1].yr,elections[myid1].state,elections[myid1].gaps[10]
#             for k in range(len(gaps1)):
#                 gaps[k][cnt] += abs(gaps2[k] - gaps1[k])
#     cnt += 1
# for i in range(14): # gap type
#     for j in range(9): # year
#         gaps[i][j] /= races[j]
#         print gaps[i][j],
#     print

# intra_cycles = [0 for i in range(14)]
# inter_cycles = [0 for i in range(14)]
# for i in range(14): # gap type
#     for j in [0,1,2,3,5,6,7,8]:
#         intra_cycles[i] += gaps[i][j]
#     intra_cycles[i] /= 8
#     inter_cycles[i] = gaps[i][4]

# for i in range(14):
#     print "%d %.4f %.4f %.4f" % (i,intra_cycles[i],inter_cycles[i],inter_cycles[i]/intra_cycles[i])
# # print [inter_cycles[i]/intra_cycles[i] for i in range(14)]
# make_scatter('alpha2.png',intra_cycles,inter_cycles)   
# # inter_cycles = [0 for i in ]
# # intra_cycles = [
# # make_scatter([    

# # for yr in ['00','02','04','06','08','10','12']:
# #     mystr = '20' + yr + '_' + st + '_9'

