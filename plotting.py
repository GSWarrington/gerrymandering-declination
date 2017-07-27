# 4 - extracting data from elections

def make_histogram(fn,arr,numbins=20):
    """
    """
    plt.figure(figsize=(12,8))

    n, bins, patches = plt.hist(arr, numbins, facecolor='g', alpha=0.75)

    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    plt.show()
    plt.close()

def make_scatter(fn,arr1,arr2,tmpstr='',bounds=[]):
    """
    """
    fig = plt.figure(figsize=(8,8))
    plt.scatter(arr1,arr2)
    if bounds != []:
        plt.axis(bounds)
    plt.grid(True)
    plt.xlabel(tmpstr)
    # plt.axhline(0.5, color='red', linestyle='dashed', linewidth=2)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    # fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    # fig.suptitle('asdfasdfasdf')
    # fig.set_title('borg')
    plt.show()
    plt.close()

def make_scatter_stlabels(fn,arr1,arr2,labels):
    """
    """
    fig = plt.figure(figsize=(8,8))
    plt.scatter(arr1,arr2)
    for i,txt in enumerate(labels):
        if txt != '': # and (abs(float(txt)+2) < 0.25 or abs(float(txt)) < 0.25 or abs(float(txt)-2) < 0.25):
            plt.annotate(txt,(arr1[i],arr2[i]))
    plt.grid(True)
    # plt.axhline(0.5, color='red', linestyle='dashed', linewidth=2)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    # fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    # fig.suptitle('asdfasdfasdf')
    # fig.set_title('borg')
    plt.show()
    plt.close()

def make_scatter_labels(fn,arr1,arr2,labels):
    """
    """
    fig = plt.figure(figsize=(8,8))
    plt.scatter(arr1,arr2)
    for i,txt in enumerate(labels):
        if txt != '': # and (abs(float(txt)+2) < 0.25 or abs(float(txt)) < 0.25 or abs(float(txt)-2) < 0.25):
            tmpstr = "%.2f" % (txt)
            plt.annotate(tmpstr,(arr1[i],arr2[i]))
    plt.grid(True)
    # plt.axhline(0.5, color='red', linestyle='dashed', linewidth=2)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    # fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    # fig.suptitle('asdfasdfasdf')
    # fig.set_title('borg')
    plt.show()
    plt.close()

# fig = plt.figure()
# plt.plot(data)
# fig.suptitle('test title', fontsize=20)
# plt.xlabel('xlabel', fontsize=18)
# plt.ylabel('ylabel', fontsize=16)
# fig.savefig('test.jpg')

def get_frac_array(elections):
    """ get array of fraction of votes won by all candidates; seems roughly bell shaped (truncated? beta?)
    """ 
    ans = []
    for k in elections.keys():
        elec = elections[k]
        for i in range(elec.Ndists):
            if elec.dcands[i] != None and not math.isnan(elec.dcands[i].frac):
                ans.append(elec.dcands[i].frac)
            if elec.rcands[i] != None and not math.isnan(elec.rcands[i].frac):
                ans.append(elec.rcands[i].frac)
    return ans

def discrepancy(elections):
    """ find races with large egap-fgap discrepancies that are not too one-sided
    """
    for k in elections.keys():
        elec = elections[k]
        print "%.4f %.4f %.4f %.4f %d %s %s %s" % \
            (elec.egap-elec.fgap,elec.egap,elec.fgap,elec.Ddists*1.0/elec.Ndists,elec.Ndists,elec.yr,elec.state,elec.chamber)

def plot_race(loc_elections,yr,state,chamber):
    """
    """
    k = '_'.join([yr,state,chamber])
    elec = loc_elections[k]
    tmpstr = "e=%.4f..f=%.4f" % (elec.egap,elec.fgap)
    # print tmpstr
    fn = 'newraces-' + k # + tmpstr
    arr1 = []
    arr2 = []
    for i in range(elec.Ndists):
        if elec.dcands[i] == None: # if no democratic candidate, then got 0
            val = 0.0
        elif elec.rcands[i] == None: # no republican candidate
            val = 1.0
        else:
            val = elec.dcands[i].frac
            # print val
        # print " blah %s %s" % (str(elec.dcands[i]),str(elec.rcands[i]))
        arr2.append(val)
        arr1.append(i)
    # print arr1
    # print arr2
    make_scatter(fn,arr1,sorted(arr2),tmpstr)

def plot_all_races(elections):
    """
    """
    for k in elections.keys():
        yr,state,chamber = k.split('_')
        plot_race(elections,yr,state,chamber)

def get_stabilizers(elections):
    """
    """
    stabe = []
    stabf = []
    stabh = []
    stabi = []
    for elec in elections.values():
        tstr = '_'.join([str(int(elec.yr)-2),elec.state,elec.chamber])
        if tstr in elections.keys() and elec.Ndists >= 8:
            nelec = elections[tstr]
            stabe.append([nelec.egap,elec.egap])
            stabf.append([nelec.fgap,elec.fgap])
            stabh.append([nelec.hgap,elec.hgap])
            stabi.append([nelec.igap,elec.igap])
    make_scatter('stabe',[x[0] for x in stabe],[x[1] for x in stabe],'')
    make_scatter('stabf',[x[0] for x in stabf],[x[1] for x in stabf],'')
    make_scatter('stabh',[x[0] for x in stabh],[x[1] for x in stabh],'')
    make_scatter('stabi',[x[0] for x in stabi],[x[1] for x in stabi],'')
    slope, intercept, r_value, p_value, std_err = stats.linregress([x[0] for x in stabe],[x[1] for x in stabe])
    print slope, intercept, r_value**2
    slope, intercept, r_value, p_value, std_err = stats.linregress([x[0] for x in stabf],[x[1] for x in stabf])
    print slope, intercept, r_value**2
    slope, intercept, r_value, p_value, std_err = stats.linregress([x[0] for x in stabh],[x[1] for x in stabh])
    print slope, intercept, r_value**2
    slope, intercept, r_value, p_value, std_err = stats.linregress([x[0] for x in stabi],[x[1] for x in stabi])
    print slope, intercept, r_value**2

def new_plot_timeseries(elections):
    ""
    ""
    clist = ['bs-','r^-','go-','cs-','m^--','yo-']
    for st in set([t.state for t in elections.values()]):
        l9 = []
        l11 = []
        for elec in filter(lambda y: y.state == st and int(y.yr) >= 1972, elections.values()):
            a = compute_alpha_curve(elec.demfrac,0)
            b = compute_alpha_curve(elec.demfrac,1)
            c = find_angle(elec.state,elec.demfrac)
            if elec.chamber == '9':
                l9.append([elec.yr,a,b,c])
            if elec.chamber == '11':
                l11.append([elec.yr,a,b,c])
        l9 = sorted(l9)
        l11 = sorted(l11)
        chamberlist = [l11]
        labs = ["state","Congress"]
        legs = []
        plt.figure(figsize=(8,8))
        plt.axis([1972,2016,-1,1])
        # print l9
        for j in range(len(chamberlist)):
            tmp, = plt.plot([y[0] for y in chamberlist[j]],[y[1] for y in chamberlist[j]],clist[j],label=labs[j])
            legs.append(tmp)
            tmp2, = plt.plot([y[0] for y in chamberlist[j]],[y[2] for y in chamberlist[j]],clist[1+j])
            legs.append(tmp2)
            tmp3, = plt.plot([y[0] for y in chamberlist[j]],[y[3] for y in chamberlist[j]],clist[2+j])
            legs.append(tmp3)
                # plt.plot(['1980','2000','1990'],[1+j,3-j,5+j],clist[j])
        plt.grid(True)
        plt.legend(handles=legs,loc='upper left')
        plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'triseries-' + st)
        plt.show()
        plt.close()    

def plot_timeseries(elections):
    ""
    ""
    clist = ['bs-','r^-','go-','cs-','m^-','yo-']
    for st in set([t.state for t in elections.values()]):
        l8 = []
        l9 = []
        l11 = []
        for elec in filter(lambda y: y.state == st, elections.values()):
            if elec.chamber == '8':
                l8.append([elec.yr,elec.egap,elec.fgap,elec.hgap,elec.igap])
            if elec.chamber == '9':
                l9.append([elec.yr,elec.egap,elec.fgap,elec.hgap,elec.igap])
            if elec.chamber == '11':
                l11.append([elec.yr,elec.egap,elec.fgap,elec.hgap,elec.igap])
        l8 = sorted(l8)
        l9 = sorted(l9)
        l11 = sorted(l11)
        chamberlist = [l9,l11]
        plt.figure(figsize=(8,8))
        plt.axis([1950,2016,-0.5,0.5])
        # print l9
        for j in range(len(chamberlist)):
            plt.plot([y[0] for y in chamberlist[j]],[y[4] for y in chamberlist[j]],clist[j])
            plt.plot([y[0] for y in chamberlist[j]],[y[2] for y in chamberlist[j]],clist[3+j])
                # plt.plot(['1980','2000','1990'],[1+j,3-j,5+j],clist[j])
        plt.grid(True)
        plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'hseries-' + st)
        plt.show()
        plt.close()    

def get_frac_unopposed(elections):
    """ get array of fraction of races in an election that are unopposed
    """ 
    arr = []
    for k in elections.keys():
        unop = 0
        for i in range(elections[k].Ndists):
            actual_votes,dv,rv = elections[k].compute_district_vote(i)
            if actual_votes == False:
                unop += 1
        if elections[k].chamber != '11':
            elections[k].unopposed = unop*1.0/elections[k].Ndists
            arr.append(unop*1.0/elections[k].Ndists)
    make_histogram('unopposed',arr)
                                   
# discrepancy(elections)
# plot_race('2008','CA','11')
# plot_race('2008','IL','11')
# plot_race('1982','AR','9')
# plot_race('2012','WI','11')
# plot_race('2012','WI','9')
# plot_all_races(elections)

# get_frac_unopposed(elections)

##############################################################################
##############################################################################
def plot_extreme_grid(fnstr,c,r,arrlist,extlist,statone,stattwo):
    """ Make grid of angles for paper
        arrlist - election keys
        extlist - string to print explaining why it's extreme (upper left)
        statone - strings to print in lower right
        stattwo - strings to print in lower right
    """
    fig, axes = plt.subplots(r,c, figsize=(4*c,4*r), sharex=True, sharey=True)
    axes = axes.ravel()

    i = -1
    for jj,k in enumerate(arrlist):
        elec = Melections[k]
        vals = sorted(elec.demfrac)

        N = len(vals)
        m = len(filter(lambda x: x < 0.5, vals))
        n = N-m

        i += 1

        # plot actual vote fractions
        x1 = [j*1.0/N-1.0/(2*N) for j in range(1,m+1)]
        x2 = [m*1.0/N + j*1.0/N-1.0/(2*N) for j in range(1,n+1)]
        y1 = vals[:m]
        y2 = vals[m:]
    
        # plot mid line
        axes[i].axhline(0.5, color = 'black', linestyle = 'dotted')

        axes[i].scatter(x1,y1,color = 'red',s=40)
        axes[i].scatter(x2,y2,color = 'blue',s=40)

        # plot values of metrics
        # if k[0] < -1:
        #     tmpstr = '$\\alpha = N/A'
        # else:
        #     tmpstr = '$\\alpha = ' + ("% .3f$" % (k[0]))
        axes[i].annotate(statone[jj], (0.49,0.12))
        # tmpstr = '$\mathrm{Gap}_1 = ' + ("% .3f$" % (compute_alpha_curve(elec.demfrac,1)))
        axes[i].annotate(stattwo[jj], (0.34,0.02))
        axes[i].annotate(extlist[jj], (0.04,0.82))

        axes[i].get_xaxis().set_ticks([])
        axes[i].set_ylim(0,1)
        axes[i].set_xlim(0,1)
        # yr,st,chm = k.split('_')
        axes[i].set_title(elec.state + ' ' + elec.yr)
        if not i%c:
            axes[i].set_ylabel('dem. vote')
        if i>=(r-1)*c:
            axes[i].set_xlabel('district')

        if m > 0 and n > 0:
            # plot angled lines
            ybar = np.mean(vals[:m])
            zbar = np.mean(vals[m:])

            axes[i].plot([m*1.0/(2*N),m*1.0/N], [ybar,0.5], 'k-', linewidth=3)
            axes[i].plot([m*1.0/N,m*1.0/N+n*1.0/(2*N)], [0.5,zbar], 'k-', linewidth=3)
            axes[i].plot([m*1.0/(2*N),m*1.0/N,m*1.0/N+n*1.0/(2*N)],[ybar,0.5,zbar],'ko',markersize=10)

    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png')
    # plt.show()
    plt.close()   
