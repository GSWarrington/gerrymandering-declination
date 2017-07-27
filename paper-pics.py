#######################################################
# plot difference function for various values of tau
# WORKING ON
#######################################################
def return_diff_functions(alphas,myplt):
    """
    """
    # plt.figure(figsize=(2,8))
    # plt.axis([0,1,-1,1])
    xvals = np.linspace(0,1,31)
    legs = []
    for alph in alphas:
        if alph >= 0:
            yvals = [2*pow(x,alph+1)-1 for x in xvals]
        else:
            yvals = [1-2*pow(1-x,-alph+1) for x in xvals]
        tmp, = myplt.plot(xvals,yvals,label = "$\\tau=%s$" % (alph))
        legs.append(tmp)
    return legs
# plt.rc(font
# matplotlib.rcParams.update({'font.size': 36})
# matplotlib.rc('legend',fontsize=24)

# plt.figsize=(12,8)


###########################################################################
# setting font sizes
# SMALL_SIZE = 18
# MEDIUM_SIZE = 24
# BIGGER_SIZE = 36
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#################################################################################################

def make_f_corr_grid(tmpstr,elections):
    """ 
    """ 
    xmin = -0.2
    xmax = 0.2
    # plt.figure(figsize=(8,8))
    # plt.axis([xmin,xmax,-0.5,0.5])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes = axes.ravel()

    # graph difference in waste functions for single-district election
    axes[0].axis([0,1,-1,1])
    legs = return_diff_functions([-1,0,1],axes[0])   
    # axes[0].grid(True)
    axes[0].legend(handles=legs,loc='lower right')
    # axes[0].xlabel('$a_i$')
    # axes[0].ylabel('$w_{P,\\alpha}-w_{Q,\\alpha}$')
    # axes[0].savefig('/home/gswarrin/research/gerrymander/pics/' + 'wastefun.png')

    # make correlation plot
    plot_snr_corr(elections,axes[1])

    # for i,k in enumerate(keys):
    #     elec = elections[k]
    #     vals = []
    #     for j in range(elec.Ndists):
    #         # if elec.dcands[j] != None:
    #         vals.append(elec.demfrac[j])
    #     vals = sorted(vals)
    #     N = len(vals)
    #     # print k,vals
    #     m = len(filter(lambda x: x < 0.5, vals))
    #     n = N-m
    #     ans,m1,i1,m2,i2 = find_lines(elec.state,vals)
    #     if ans == None:
    #         print "help",i
    #         continue
            
    #     x1 = range(1,m+1)
    #     x2 = range(m+1,N+1)
    #     y1 = vals[:m]
    #     y2 = vals[m:]
    #     # ptpts.append(ans)
    #     # d['_'.join([elec.yr,elec.state,elec.chamber])] = [ans[0]*ans[0] + \
    #     #        ans[1]*ans[1],ans[0],ans[1]]
    #     axes[i].scatter(x1,y1,color = 'red')
    #     axes[i].scatter(x2,y2,color = 'blue')
    #     # plt.scatter([x[0] for x in ptpts],[x[1] for x in ptpts])

    #     # y = srrs_mn.log_radon[srrs_mn.county==c]
    #     # x = srrs_mn.floor[srrs_mn.county==c]
    #     # axes[i].scatter(x + np.random.randn(len(x))*0.01, y, alpha=0.4)
        
    #     # No pooling model
    #     # b = unpooled_estimates[c]
        
    #     # Plot both models and data
    #     xvals1 = np.linspace(1,m,m+1)
    #     xvals2 = np.linspace(m+1,N,n+1)
    #     xvals3 = np.linspace(1,N,10)
    #     axes[i].plot(xvals1, m1*xvals1+i1, 'r')
    #     axes[i].plot(xvals2, m2*xvals2+i2, 'b')
    #     axes[i].plot(xvals3, 0.5 + 0*xvals3, color = 'black')
    #     axes[i].set_xticks([1,N])
    #     # axes[i].set_xticklabels(['basement', 'floor'])
    #     axes[i].set_ylim(0,1)
    #     yr,st,chm = k.split('_')
    #     axes[i].set_title(st + ' ' + yr)
    #     if not i%2:
    #         axes[i].set_ylabel('democratic vote fraction')
    
    # axes[1].text(1,0.05,'z = (0.00,-0.17)')
    # axes[0].text(1,0.05,'z = (-0.07,-0.07)')
    # axes[2].text(1,0.05,'z = (0.03,0.03)')
    # axes[3].text(1,0.05,'z = (0.09,-0.12)')

    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'combo-' + tmpstr + '.png')
    # plt.show()
    plt.close()   

########################################################################################

def plot_paper_timeseries(elections,ckeys):
    """ for paper
    """
    clist = ['-.','-','--',':','^m--','yo-']
#    clist = ['.','^-','o--','*:','^m--','yo-']
    plt.figure(figsize=(8,4))
    plt.gca().set_axis_bgcolor('none')
    plt.axis([1970,2016,-1,1])
    legs = []
    labs = []
    linestyles = ['_', '-', '--', ':']
    for i,k in enumerate(ckeys):
        minyr,st,chamber = k.split('_')
        l = []
        for elec in filter(lambda y: y.state == st and int(y.yr) >= 1972 and \
                           y.chamber == chamber, elections.values()):
            # b = compute_alpha_curve(elec.demfrac,1)
            b = find_angle(elec.state,elec.demfrac)
            l.append([elec.yr,b])
        l = sorted(l)
        labs.append(st)
        # legs = []
        # print l9
        tmp, = plt.plot([y[0] for y in l],[y[1] for y in l],clist[i],label=labs[i])
        legs.append(tmp)
    # plt.grid(True)
    plt.xlabel('Year')
    plt.ylabel('Declination')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(0., 0.8, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0., handles=legs)
    # plt.legend(handles=legs,loc='upper left')
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'paperseries-ang')
    plt.show()
    plt.close()    

########################################################################################

def plot_angle(k,elections):
    """
    """
    plt.figure(figsize=(10,5),frameon=False)
    axes = plt.gca()
    # axes.set_axis_bgcolor('none')
    axes.set_frame_on(True)
    axes.set_axis_on()
    # plt.set_frame_on(True)
    elec = elections[k]
    vals = sorted(elec.demfrac)
    N = len(vals)
    m = len(filter(lambda x: x < 0.5, vals))
    n = N-m
    # ans,m1,i1,m2,i2 = find_lines(elec.state,vals)
    ybar = np.mean(vals[:m])
    zbar = np.mean(vals[m+1:])

    plt.grid(False)
    # axes.set_facecolor('green')
    
    # plot actual vote fractions
    x1 = [i*1.0/N-1.0/(2*N) for i in range(1,m+1)]
    x2 = [m*1.0/N + i*1.0/N-1.0/(2*N) for i in range(1,n+1)]
    y1 = vals[:m]
    y2 = vals[m:]
    plt.scatter(x1,y1,color = 'red',s=60)
    plt.scatter(x2,y2,color = 'blue',s=60)
    
    # plot mid line
    plt.plot([0,1], [0.5,0.5], color = 'black')
    # plot angled lines
    plt.plot([m*1.0/(2*N),m*1.0/N], [ybar,0.5], 'k-')
    plt.plot([m*1.0/N,m*1.0/N+n*1.0/(2*N)], [0.5,zbar], 'k-')

    # ax = gca()
    ptF = [m*1.0/(2*N),ybar]
    ptG = [m*1.0/N,0.5]
    ptH = [m*1.0/N + n*1.0/(2*N),zbar]
    ptT = [0,0.5]
    ptU = [1,0.5]

    # line, = axes.plot([ptT[0],ptG[0],ptF[0]], [ptT[1],ptG[1],ptF[1]], 'k-', lw=2)
    #    line2, = axes.plot([ptU[0],ptG[0],ptH[0]], [ptU[1],ptG[1],ptH[1]], 'k-', lw=2)
    axes.add_patch(matplotlib.patches.Arc(ptG, .2, .2, 0, 0, 65, color='green',lw=3))
    plt.annotate('$\\theta_P$',(0.85,0.57))
    axes.add_patch(matplotlib.patches.Arc(ptG, .4, .4, 0, 180, 197, color='green',lw=3))
    plt.annotate('$\\theta_Q$',(0.52,0.46))
    plt.plot([ptG[0],1],[ptG[1],ptG[1]+(1-ptG[0])*(0.5-ybar)/(ptG[0]-ptF[0])],'k-.')
    axes.add_patch(matplotlib.patches.Arc(ptG, .4, .4, 0, 17, 65, color='green',lw=3))
    plt.annotate('$\\delta\\pi/2$',(0.94,0.63))
    plt.annotate('F',(0.37,0.34))
    plt.annotate('G',(0.76,0.46))
    plt.annotate('H',(0.9,0.71))
    plt.annotate('T',(0,0.46))
    plt.annotate('U',(1,0.46))
    plt.xlabel('District')
    plt.ylabel('Democratic vote fraction')
    plt.plot([m*1.0/(2*N),m*1.0/N+n*1.0/(2*N)],[ybar,zbar],'ko',markersize=5)
    plt.plot([ptT[0],ptG[0],ptU[0]],[ptT[1],ptG[1],ptU[1]],'ko',markersize=5)
    plt.axis([-0.1,1.1,0.25,0.8])
    plt.gca().set_axis_bgcolor('none')    
    plt.tight_layout() # make sure labels fit

    # add_corner_arc(axes, line, text='$\\theta$') # u'%d\u00b0' % 90)
    # add_corner_arc(axes, line2, text='$\\phi$') # u'%d\u00b0' % 90)
    # add_corner_arc(ax, line, radius=.7, color='black', text=None, text_radius=.5, text_rotatation=0, **kwargs):

    # plt.set_xticks([1,N])
    # plt.set_ylim(0,1)
    yr,st,chm = k.split('_')
    # plt.set_title(st + ' ' + yr)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/paper-angle-plot-' + k + '.png')
    plt.show()
    plt.close()   
    
########################################################################################

def split_declination_states(fnstr,c,r,yr,chamber,states,elections,mymmd):
    """ Make grid of angles for paper
    """
    nstates = []
    for x in states:
        tmpkey = '_'.join([yr,x,chamber])
        if tmpkey in elections.keys():
            elec = elections[tmpkey]
            ndem = len(filter(lambda y: y >= 0.5, elec.demfrac))
            if elec.Ndists > 1 and (yr not in mymmd or x not in mymmd[yr]) and \
               0 < ndem < elec.Ndists:
                nstates.append([find_angle(x,elec.demfrac),elec])
    nstates.sort(key=lambda x: x[0])
    nstates.reverse()
    
    plot_declination_grid(fnstr,r,c,nstates)
    # plot_declination_grid(fnstr + '-pg1',r,c,yr,chamber,nstates[:24])
    # plot_declination_grid(fnstr + '-pg2',r,c,yr,chamber,nstates[24:])

def grid_all_states(fnstr,c,r,chamber,states,elections,mymmd):
    """ Make grid of angles for paper
    """
    for st in states:
        grid_one_state(fnstr,6,4,chamber,st,elections,mymmd)

def grid_one_state(fnstr,c,r,chamber,st,elections,mymmd):
    """ Make grid of angles for paper
    """
    nstates = []
    maxdists = 0
    for j in range(24):
        yr = str(1972+2*j)
        tmpkey = '_'.join([yr,st,chamber])
        if tmpkey in elections.keys():
            elec = elections[tmpkey]
            maxdists = max(elec.Ndists,maxdists)
            if elec.Ndists > 1 and (chamber == '11' or yr not in mymmd or st not in mymmd[yr]):
                nstates.append([find_angle(x,elec.demfrac),elec])
    # didn't have more than one district at any point
    # if maxdists == 1:
    #     return
    plot_declination_grid('angle-plot-' + st + '-' + chamber,r,c,nstates)

def plot_nc_tx_pic(fnstr,elections):
    """ Make grid of angles for paper
    """
    fig, axes = plt.subplots(2,2, figsize=(8,8), facecolor = 'white', frameon=True)
    axes = axes.ravel()
    # fig.patch.set_visible(False)

    elec = elections['1974_TX_11']
    plot_one_declination(axes[0],elec.demfrac,'Texas 1974',plotdec=True,xaxislab=True,yaxislab=True)
    axes[0].text(0.01,.96,'A',fontsize=16, transform=fig.transFigure, fontweight='bold') # , va='top', ha='right')
    axes[1].text(0.5,.96,'B',fontsize=16, transform=fig.transFigure, fontweight='bold') # , va='top', ha='right')
    axes[2].text(0.01,.5,'C',fontsize=16, transform=fig.transFigure, fontweight='bold') # , va='top', ha='right')
    axes[3].text(0.5,.5,'D',fontsize=16, transform=fig.transFigure, fontweight='bold') # , va='top', ha='right')

    ############################
    # explore number of stolen seats
    elec = elections['2012_PA_11']
    plot_one_declination(axes[1],elec.demfrac,'Pennsylvania 2012',\
                         plotdec=True,xaxislab=True,yaxislab=True,plotfullreg=True)
    # axe.plot([m*1.0/(2*N),m*1.0/N], [ybar,0.5], 'k-', linewidth=3)

    ############################
    # hypothetical district plan
    ncols = 6
    nrows = 4
    stval = 0.47
    vx = .04
    vy = -.02
    arr = [[stval + vx*i + vy*j for i in range(ncols)] for j in range(nrows)]

    arr2 = np.array(arr)
    arr2.shape = (1,ncols*nrows)
    onedarr = sorted(list(arr2[0]))
    cdict = {'red':   [(0.0,  1.0, 1.0),(1.0,  0.0, 0.0)],\
             'green': [(0.0,  0.0, 0.0),(1.0,  0.0, 0.0)],\
             'blue':  [(0.0,  0.0, 0.0),(1.0,  1.0, 1.0)]}
    rpb_color = LinearSegmentedColormap('rpb',cdict)

    # print find_angle('',onedarr),compute_alpha_curve(onedarr,0),np.mean(onedarr)

    df = pd.DataFrame(arr) # , index=index)
    nax = sns.heatmap(df, ax=axes[2], cmap=rpb_color, linewidths=0, vmin=0.3, vmax=0.7)
    nax.set_title('Hypothetical district plan')
    nax.axis('off')
    # arr = [[.3,.4,.5],[.35,.45,.55]]

    #########################################
    # vote distribution for hypothetical plan
    plot_one_declination(axes[3],onedarr,'Hypothetical election',plotdec=True,xaxislab=True,yaxislab=True)

    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr + '-new.png')
    plt.close()   

def plot_one_declination(axe,arr,title,plotdec=True,xaxislab=True,yaxislab=True,plotfullreg=False):
    """ Make grid of angles for paper
    """
    axe.set_axis_bgcolor('none')

    vals = sorted(arr)

    N = len(vals)
    m = len(filter(lambda x: x < 0.5, vals))
    n = N-m

    # plot actual vote fractions
    x1 = [j*1.0/N-1.0/(2*N) for j in range(1,m+1)]
    x2 = [m*1.0/N + j*1.0/N-1.0/(2*N) for j in range(1,n+1)]
    y1 = vals[:m]
    y2 = vals[m:]

    # plot mid line
    axe.axhline(0.5, color = 'black', linestyle = 'dotted')

    axe.scatter(x1,y1,color = 'red',s=60)
    axe.scatter(x2,y2,color = 'blue',s=60)

    # plot values of metrics
    if plotdec:
        fa = find_angle('',vals)
        if abs(fa) >= 2:
            tmpstr = '$\\delta = N/A'
        else:
            tmpstr = '$\\delta = ' + ("% .2f$" % (fa))
        axe.annotate(tmpstr, (0.02,0.84))

    axe.get_xaxis().set_ticks([])
    axe.set_ylim(0,1)
    axe.set_xlim(0,1)
    axe.set_title(title) # elec.state + ' ' + elec.yr)
    if yaxislab:
        axe.set_ylabel('Dem. vote')
    if xaxislab:
        axe.set_xlabel('District')

    if m > 0 and n > 0:
        # plot angled lines
        ybar = np.mean(vals[:m])
        zbar = np.mean(vals[m:])

        axe.plot([m*1.0/(2*N),m*1.0/N], [ybar,0.5], 'k-', linewidth=3)
        axe.plot([m*1.0/N,m*1.0/N+n*1.0/(2*N)], [0.5,zbar], 'k-', linewidth=3)
        axe.plot([m*1.0/(2*N),m*1.0/N,m*1.0/N+n*1.0/(2*N)],[ybar,0.5,zbar],'ko',markersize=5)

        if plotfullreg:
            l = stats.linregress(x1+x2,vals)
            print "regress line: ",l
            axe.plot([0,1],[l[1],l[1]+l[0]],'k-',linewidth=1)

    axe.set_axis_bgcolor('none')

def plot_declination_grid(fnstr,c,r,arrlist,titles=[]):
    """ Make grid of angles for paper
    """
    fig, axes = plt.subplots(r,c, figsize=(4*c,4*r), sharex=True, sharey=True, facecolor = 'white', frameon=True)
    axes = axes.ravel()
    # fig.patch.set_visible(False)

    i = -1
    for idx,k in enumerate(arrlist):
        i += 1
        # axes[i].set_axis_bgcolor('black')
        # axes[i].patch.set_visible(False)
        axes[i].set_axis_bgcolor('none')
        # axes[i].set_color('black')
        # axes[i].set_axis_on()

        if titles == []:
            vals = sorted(k[1])
        else:
            elec = k[1]
            vals = sorted(elec.demfrac)

        N = len(vals)
        m = len(filter(lambda x: x < 0.5, vals))
        n = N-m

        # plot actual vote fractions
        x1 = [j*1.0/N-1.0/(2*N) for j in range(1,m+1)]
        x2 = [m*1.0/N + j*1.0/N-1.0/(2*N) for j in range(1,n+1)]
        y1 = vals[:m]
        y2 = vals[m:]
    
        # plot mid line
        axes[i].axhline(0.5, color = 'black', linestyle = 'dotted')

        axes[i].scatter(x1,y1,color = 'red',s=60)
        axes[i].scatter(x2,y2,color = 'blue',s=60)

        # plot values of metrics
        if k[0] < -1:
            tmpstr = '$\\delta = N/A'
        else:
            tmpstr = '$\\delta = ' + ("% .2f$" % (k[0]))
        axes[i].annotate(tmpstr, (0.02,0.84))
        # tmpstr = '$\mathrm{Gap}_1 = ' + ("% .3f$" % (compute_alpha_curve(elec.demfrac,1)))
        # axes[i].annotate(tmpstr, (0.34,0.02))

        axes[i].get_xaxis().set_ticks([])
        axes[i].set_ylim(0,1)
        axes[i].set_xlim(0,1)
        # yr,st,chm = k.split('_')
        axes[i].set_title(titles[idx]) # elec.state + ' ' + elec.yr)
        if not i%c:
            axes[i].set_ylabel('Dem. vote')
        if i>= min((r-1)*c,len(arrlist)-c):
            axes[i].set_xlabel('District')

        if m > 0 and n > 0:
            # plot angled lines
            ybar = np.mean(vals[:m])
            zbar = np.mean(vals[m:])

            axes[i].plot([m*1.0/(2*N),m*1.0/N], [ybar,0.5], 'k-', linewidth=3)
            axes[i].plot([m*1.0/N,m*1.0/N+n*1.0/(2*N)], [0.5,zbar], 'k-', linewidth=3)
            axes[i].plot([m*1.0/(2*N),m*1.0/N,m*1.0/N+n*1.0/(2*N)],[ybar,0.5,zbar],'ko',markersize=5)

    while i < r*c-1:
        i += 1
        axes[i].set_axis_bgcolor('none')
        
    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png')
    # plt.show()
    plt.close()   


################################################################################

def make_paper_scatter_wi():
    lang = []
    lagap = []
    lzgap = []
    lqgap = []
    for elec in Melections.values():
        # elec.myprint()
        # print "yr: ",elec.yr
        if 2022 >= int(elec.yr) >= 1972 and (elec.yr not in Mmmd.keys() or elec.state not in Mmmd[elec.yr]) and \
            elec.Ndists >= 8 and elec.chamber=='9':
            ang = find_angle(elec.state,elec.demfrac)
            agap = compute_alpha_curve(elec.demfrac,1)
            zgap = compute_alpha_curve(elec.demfrac,0)
            qgap = compute_alpha_curve(elec.demfrac,0.5)

            if ang != 0:
                lang.append(ang)
                lagap.append(agap)
                lzgap.append(zgap)
                lqgap.append(qgap)
            # print "blah"
            print "% .3f % .3f %3d %s %s %2d" % (ang,zgap,elec.Ndists,elec.yr,elec.state,int(elec.chamber))
    print stats.pearsonr(lang,lagap)
    print np.std(lang)
    print np.std(lagap)
    print np.std(lzgap)
    print np.std(lqgap)
    plt.figure(figsize=(8,8))
    plt.axis([-0.6,0.6,-0.6,0.6])
    plt.axvline(0,color='black',ls='dotted')
    plt.axhline(0,color='black',ls='dotted')
    plt.scatter(lang,lagap)
    plt.xlabel('Declination',fontsize=18)
    plt.ylabel('$\mathrm{Gap}_0$',fontsize=18)
    datax = [0.055,0.343,0.244,0.328]
    datay = [0.150,0.247,0.251,0.280]
    plt.scatter(datax[1:],datay[1:],color='red',marker='o',s=100)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'wi-ex-scatter')
    plt.close()
    # make_scatter('wi-ex-scatter',lang,lagap)

def wi_scatter_decade():
    lang = [[] for i in range(5)]
    lzgap = [[] for i in range(5)]
    cnt = 0
    for elec in Melections.values():
        # elec.myprint()
        # print "yr: ",elec.yr
           # (elec.yr not in Mmmd.keys() or elec.state not in Mmmd[elec.yr]) and \
        if 2010 >= int(elec.yr) >= 1972 and \
           elec.state not in Mmmd['1972'] and \
            elec.Ndists >= 8 and elec.chamber=='9':
            ang = find_angle(elec.state,elec.demfrac)
            zgap = compute_alpha_curve(elec.demfrac,0)
            cnt += 1

            yridx = int((int(elec.yr)-1972)/10)
            if ang != 0:
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
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'scatter-wi-nommd')
    plt.close()
    # make_scatter('wi-ex-scatter',lang,lagap)

##############################################################################
# for yy in ['1972','1982','1992','2002','2012']:
#     for ch in ['9','11']:
#         make_line_chart(Melections,yy,ch)

def make_line_chart(elections,yrmin,chamber='9',num=5):
    """
    """
    ans = []
    cnt = 0
    
    for i,state in enumerate(Mstates):
        if yrmin in Mmmd.keys() and state in Mmmd[yrmin]:
            continue
        stcnt = 0
        curmin = 10
        curmax = -10
        isokay = True
        szs = 0
        for yr in [str(int(yrmin)+2*j) for j in range(num)]:
            myid = '_'.join([yr,state,chamber])
            if myid in elections.keys() and elections[myid].Ndists >= 8:
                # isokay = False
                # break
            # if elections[myid].Ndists < 8:
                # isokay = False
                elec = elections[myid]
                if szs == 0:
                    if chamber == '9':
                        szs = 1.0
                    else:
                        szs = 1.0 # elec.Ndists*1.0/2
                aval = find_angle(elec.state,elec.demfrac) # 
                # aval = compute_alpha_curve(elec.demfrac,1)
                # aval = compute_alpha_curve(elec.demfrac,1)
                # aval = -compute_egap_directly(elec.demfrac)
                if aval == None:
                    continue
                if aval < curmin:
                    curmin = aval
                if aval > curmax:
                    curmax = aval
                cnt += 1
                stcnt += 1
            # if isokay:
        if curmin < 10 and stcnt >= 2: # and aval != None:
            ans.append([state,szs*(curmin+curmax)/2,szs*curmin,szs*curmax])
                # print ans[-1]
    ans.sort(key=lambda x: x[1])
    
    plt.figure(figsize=(8,6))
    plt.gca().set_axis_bgcolor('none')
    if chamber == '9':
        plt.axis([-1,1,0,len(ans)*1.0/5+0.15])
    else:
        plt.axis([-1,1,0,len(ans)*1.0/5+0.15])
    for j in range(1,len(ans)+1):
        if 1 == 1: # ans[i][0] != 'MA' and ans[i][0] != 'WA':
            # print ans
            # print i,ans[i]
            i = j-1
            print i,ans[i][0],ans[i]
            plt.plot([ans[i][2],ans[i][3]],[i*0.2,i*0.2],color='blue')
            plt.annotate(ans[i][0],(ans[i][3],i*0.2-0.05))
    plt.axvline(0,ls='dotted',color='black')
    plt.axvline(-0.5,ls='dotted',color='black')
    plt.axvline(0.5,ls='dotted',color='black')
    plt.gca().get_yaxis().set_visible(False)
    plt.xlabel('Declination')
    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/lineplot_angle_' + chamber + '_' + yrmin + '.png')
    plt.close()
    
############################################################################
# HERE
def make_line_chart_grid(elections,chamber,mmd,useseats=False):
    """
    """
    rarr = []
    if chamber == '9':
        r = 2
        yrlist = ['1972','1982','1992','2002']
    else:
        r = 3
        yrlist = ['1972','1982','1992','2002','2012']
    c = 2
    num = 5
    plt.grid(False)
    fig, axes = plt.subplots(r,c, figsize=(10*c,10*r), sharex=True, sharey=True)
    # plt.gca().set_axis_bgcolor('none')
    axes = axes.ravel()

    for idxyr,yrmin in enumerate(yrlist):
        cnt = 0
        ans = []
        for i,cycstate in enumerate(Ncyc):
            # print "in loop: ",idxyr,i,cycstate
            if chamber == '9' and (yrmin in mmd.keys() and cycstate[:2] in mmd[yrmin]):
                # print "        skipping yrmin,cycstate"
                continue
            stcnt = 0
            curmin = 80
            curmax = -80
            isokay = True
            szs = 0
            curcyc = []
            for yr in [str(int(yrmin)+2*j) for j in range(num)]:
                myid = '_'.join([yr,cycstate[:2],chamber])
                if myid in elections.keys() and elections[myid].Ndists >= 8:
                    elec = elections[myid]
                    if elec.cyc_state != cycstate:
                        # print "elec.cyc_state != cycstate",elec.cyc_state,cycstate
                        continue
                    if szs == 0:
                        szs = 1.0 
                    if szs == 1.0 and useseats:
                        szs = math.log(elec.Ndists)/2 # math.log(elec.Ndists)*1.0/2
                    aval = find_angle(elec.cyc_state,elec.demfrac) # 
                    curcyc.append(aval*szs)
                    if abs(aval) > 2:
                        continue
                    blah.append([elec.Ndists,szs*abs(aval)])
                    if aval < curmin:
                        curmin = aval
                    if aval > curmax:
                        curmax = aval
                    cnt += 1
                    stcnt += 1
                # if isokay:
            if curmin < 80 and stcnt >= 1: # and aval != None:
                ans.append([cycstate,szs*(curmin+curmax)/2,szs*curmin,szs*curmax])
                if int(yrmin) < 2012 and len(cycstate) == 2:
                    rarr.append(szs*(curmax-curmin))
                    # blah.append([math.exp(2*szs),szs*(curmax-curmin)])
                    # print "range: %s %s %s %.2f" % (elec.yr,elec.state,elec.chamber,szs*(curmax-curmin))
                for xx in curcyc:
                    if xx >= 0:
                        quints.append(xx-szs*curmin)
                    else:
                        quints.append(szs*curmax-xx)
                # else:
                #     print yrmin,cycstate
            # else:
            #    print "Failing: ",yrmin,stcnt,chamber,cycstate,curmin,curmax,stcnt
        ans.sort(key=lambda x: x[1])
        
        # print "ready to plot", idxyr, yrmin, "len ans: ",len(ans)
        # axes[idxyr].set_axis_bgcolor('none')
        if 1 == 0 and not useseats:
            if chamber == '9':
                axes[idxyr].axis([-0.6,0.6,0,len(ans)*1.0/5+1])
            else:
                axes[idxyr].axis([-0.75,0.75,0,len(ans)*1.0/5+1])
            for xval in [-0.6,-0.4,-0.2,0.2,0.4,0.6]:
                axes[idxyr].axvline(xval,linestyle='dotted',color='black')
        axes[idxyr].grid(False)
        axes[idxyr].axvline(0,color='black')
        if idxyr>=(r-1)*c:
            axes[idxyr].set_xlabel('Declination')
        for j in range(1,len(ans)+1):
            ii = j-1
            # print ii,ans[ii][0],ans[ii]
            axes[idxyr].plot([ans[ii][2],ans[ii][3]],[0.1+ii*0.2,0.1+ii*0.2],color='blue')
            axes[idxyr].annotate(ans[ii][0],(ans[ii][3],0.1+ii*0.2-0.05))
        if int(yrmin) < 2010:
            axes[idxyr].set_title(yrmin + '-' + str(int(yrmin)+8))
        else:
            axes[idxyr].set_title(yrmin + '-' + str(int(yrmin)+4))
                     # axes[idxyr].axvline(0,ls='dotted',color='black')
        axes[idxyr].get_yaxis().set_visible(False)
        axes[idxyr].get_xaxis().set_visible(True)

    rarr = sorted(rarr)
    print "95 percent: %s %d %.2f " % (chamber,len(rarr),rarr[int(9.5*len(rarr)/10)])
    # plt.gca().set_axis_bgcolor('none')
    plt.tight_layout()
    if useseats:
        seatsstr = 'seats'
    else:
        seatsstr = 'frac'
    plt.savefig('/home/gswarrin/research/gerrymander/pics/lineplot_grid_angle_' + chamber + '_' + seatsstr + '.png')
    plt.close()

#####################################################################################
# list elections not expected to cross 0
#####################################################################################
def beyond_zero(elecs,mmd,thresh = 0.65):
    """ list elections not expected to cross 0
    """
    for elec in elecs.values():
        if int(elec.yr) >= 1972 and \
           (elec.chamber == '11' or elec.yr not in mmd.keys() or (elec.state not in mmd[elec.yr])):
            fa = find_angle('',elec.demfrac)
            if abs(fa) < 2 and abs(fa*math.log(elec.Ndists)/2) > thresh:
                print "%s %s %s % .2f %.2f" % (elec.yr,elec.chamber,elec.state,fa,abs(fa*math.log(elec.Ndists)/2))
                
#####################################################################################
# plot correlation of t-gap with extremes 
def si_corr(fn,elecs):
    """ see above description
    """
    alphavals = [0,0.25,0.5,0.75,1,2,10000] # np.linspace(0,10,21)
    zidx = 0
    ans1 = []
    ans2 = []
    arr = [[] for i in range(len(alphavals))]
    for elec in elecs.values():
        if int(elec.yr) >= 1972 and elec.chamber=='11' and elec.Ndists >= 8:
            for i in range(len(alphavals)):
                arr[i].append(compute_alpha_curve(elec.demfrac,alphavals[i]))
    labs = []
    for i in range(len(alphavals)):
        a1,pv1 = stats.pearsonr(arr[0],arr[i]) # correlation with tau=-10 (total seats)
        a2,pv2 = stats.pearsonr(arr[i],arr[-1]) # correlation with tau=10
        ans1.append(a1)
        ans2.append(a2)
        labs.append('$\\tau = ' + str(alphavals[i]) + '$')
    labs[-1] = '$\\tau = \\infty$'
    
    fig = plt.figure(figsize=(8,8))
    plt.scatter(ans1,ans2)
    for i,txt in enumerate(labs):
        plt.annotate(txt,(ans1[i]+0.01,ans2[i]))

    # plt.grid(True)
    plt.xlabel('Correlation of $\\mathrm{Gap}_{\\tau}$ with $\\mathrm{Gap}_0$')
    plt.ylabel('Correlation of $\\mathrm{Gap}_{\\tau}$ with $\\mathrm{Gap}_\\infty$')
    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    plt.close()


###############################################################################################3
# 03.28.17 - pretty sure not used at all
def figS23_line_chart_grid(elections,chamber,mmd):
    """
    """
    rarr = []
    if chamber == '9':
        r = 2
        yrlist = ['1972','1982','1992','2002']
    else:
        r = 3
        yrlist = ['1972','1982','1992','2002','2012']
    c = 2
    num = 5
    plt.grid(False)
    fig, axes = plt.subplots(r,c, figsize=(10*c,10*r), sharex=True, sharey=True)
    # plt.gca().set_axis_bgcolor('none')
    axes = axes.ravel()

    for idxyr,yrmin in enumerate(yrlist):
        cnt = 0
        ans = []
        for i,cycstate in enumerate(Ncyc):
            # print "in loop: ",idxyr,i,cycstate
            if chamber == '9' and (yrmin in mmd.keys() and cycstate[:2] in mmd[yrmin]):
                # print "        skipping yrmin,cycstate"
                continue
            stcnt = 0
            curmin = 80
            curmax = -80
            isokay = True
            szs = 0
            curcyc = []
            for yr in [str(int(yrmin)+2*j) for j in range(num)]:
                myid = '_'.join([yr,cycstate[:2],chamber])
                if myid in elections.keys() and elections[myid].Ndists >= 8:
                    elec = elections[myid]
                    if elec.cyc_state != cycstate:
                        # print "elec.cyc_state != cycstate",elec.cyc_state,cycstate
                        continue
                    szs = math.log(elec.Ndists)/2 # math.log(elec.Ndists)*1.0/2
                    aval = find_angle(elec.cyc_state,elec.demfrac) # 
                    curcyc.append(aval*szs)
                    if abs(aval) >= 2:
                        continue
                    blah.append([elec.Ndists,szs*abs(aval)])
                    if aval < curmin:
                        curmin = aval
                    if aval > curmax:
                        curmax = aval
                    cnt += 1
                    stcnt += 1
                # if isokay:
            if curmin < 80 and stcnt >= 1: # and aval != None:
                ans.append([cycstate,szs*(curmin+curmax)/2,szs*curmin,szs*curmax])
                if int(yrmin) < 2012 and len(cycstate) == 2:
                    rarr.append(szs*(curmax-curmin))
                    # blah.append([math.exp(2*szs),szs*(curmax-curmin)])
                    # print "range: %s %s %s %.2f" % (elec.yr,elec.state,elec.chamber,szs*(curmax-curmin))
                for xx in curcyc:
                    if xx >= 0:
                        quints.append(xx-szs*curmin)
                    else:
                        quints.append(szs*curmax-xx)
                # else:
                #     print yrmin,cycstate
            # else:
            #    print "Failing: ",yrmin,stcnt,chamber,cycstate,curmin,curmax,stcnt
        ans.sort(key=lambda x: x[1])
        
        # print "ready to plot", idxyr, yrmin, "len ans: ",len(ans)
        # axes[idxyr].set_axis_bgcolor('none')
        if 1 == 0 and not useseats:
            if chamber == '9':
                axes[idxyr].axis([-0.6,0.6,0,len(ans)*1.0/5+1])
            else:
                axes[idxyr].axis([-0.75,0.75,0,len(ans)*1.0/5+1])
            for xval in [-0.6,-0.4,-0.2,0.2,0.4,0.6]:
                axes[idxyr].axvline(xval,linestyle='dotted',color='black')
        axes[idxyr].grid(False)
        axes[idxyr].axvline(0,color='black')
        if idxyr>=(r-1)*c:
            axes[idxyr].set_xlabel('Declination')
        for j in range(1,len(ans)+1):
            ii = j-1
            # print ii,ans[ii][0],ans[ii]
            axes[idxyr].plot([ans[ii][2],ans[ii][3]],[0.1+ii*0.2,0.1+ii*0.2],color='blue')
            axes[idxyr].annotate(ans[ii][0],(ans[ii][3],0.1+ii*0.2-0.05))
        if int(yrmin) < 2010:
            axes[idxyr].set_title(yrmin + '-' + str(int(yrmin)+8))
        else:
            axes[idxyr].set_title(yrmin + '-' + str(int(yrmin)+4))
                     # axes[idxyr].axvline(0,ls='dotted',color='black')
        axes[idxyr].get_yaxis().set_visible(False)
        axes[idxyr].get_xaxis().set_visible(True)

    rarr = sorted(rarr)
    print "95 percent: %s %d %.2f " % (chamber,len(rarr),rarr[int(9.5*len(rarr)/10)])
    # plt.gca().set_axis_bgcolor('none')
    plt.tight_layout()
    if useseats:
        seatsstr = 'seats'
    else:
        seatsstr = 'frac'
    plt.savefig('/home/gswarrin/research/gerrymander/pics/lineplot_grid_angle_' + chamber + '_' + seatsstr + '.png')
    plt.close()

    
