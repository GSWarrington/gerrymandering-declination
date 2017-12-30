myr = '#ff8080'
myb = '#8080ff'

# Fig 0: definition of declination
#   fig0_create
#     fig1_plot_angle    

# Fig 1: hypothetical with other proportion
#   fig1_create
#     fig1_txpa_heat
#       fig1_plot_one_declination
#       fig1_plot_one_declination
#       sns.heatmap
#       fig1_plot_one_declination

# Fig 2: How things vary 
#   fig_variations
#     fig3_scatter
#     fig_pack_crack_all

# Fig 2p: 

# Fig 3: Heatmap
#   fig2_make_heatmap_e

# Fig 4: range over decade
#   fig_deltae_only

############################################################################
# Fig 1.
def fig1_plot_angle(k,axes,fig,elections,seatsbool=False):
    """ copied from plot_angle
    """
    axes.get_xaxis().set_ticks([])
    axes.set_xlabel('District')

    elec = elections[k]
    vals = sorted(elec.demfrac)
    N = len(vals)
    m = len(filter(lambda x: x < 0.5, vals))
    n = N-m
    ybar = np.mean(vals[:m])
    zbar = np.mean(vals[m+1:])

    for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +\
                 axes.get_xticklabels() + axes.get_yticklabels()):
        item.set_fontsize(14)

    # plot actual vote fractions
    x1 = [i*1.0/N-1.0/(2*N) for i in range(1,m+1)]
    x2 = [m*1.0/N + i*1.0/N-1.0/(2*N) for i in range(1,n+1)]
    y1 = vals[:m]
    y2 = vals[m:]
    axes.scatter(x1,y1,color = myr,s=60)
    axes.scatter(x2,y2,color = myb,s=60)
    
    # plot mid line
    axes.plot([0,1], [0.5,0.5], color = 'black')
    # plot angled lines
    axes.plot([m*1.0/(2*N),m*1.0/N], [ybar,0.5], 'k-')
    axes.plot([m*1.0/N,m*1.0/N+n*1.0/(2*N)], [0.5,zbar], 'k-')

    # ax = gca()
    ptF = [m*1.0/(2*N),ybar]
    ptG = [m*1.0/N,0.5]
    ptH = [m*1.0/N + n*1.0/(2*N),zbar]
    ptT = [0,0.5]
    ptU = [1,0.5]

    axes.set_ylabel('Democratic vote')

    if not seatsbool:
        # plt
        axes.annotate('$\\theta_P$',(0.85,0.57),fontsize=16)
        axes.add_patch(matplotlib.patches.Arc(ptG, .4, .4, 0, 180, 196.5, color='green',lw=3))
        # plt
        axes.annotate('$\\theta_Q$',(0.52,0.46),fontsize=16)

        axes.add_patch(matplotlib.patches.Arc(ptG, .2, .2, 0, 0, 65, color='green',lw=3))

        axes.annotate('T',(0,0.45),fontsize=16)
        axes.annotate('U',(1,0.45),fontsize=16)
        axes.plot([ptT[0],ptU[0]],[ptT[1],ptU[1]],'ko',markersize=5)

    axes.plot([ptG[0],1],[ptG[1],ptG[1]+(1-ptG[0])*(0.5-ybar)/(ptG[0]-ptF[0])],'k-.')
    axes.add_patch(matplotlib.patches.Arc(ptG, .4, .4, 0, 16.5, 65, color='green',lw=3))

    axes.annotate('$\\delta\\pi/2$',(0.94,0.63),fontsize=16)
    axes.annotate('F',(0.38,0.33),fontsize=16)
    axes.annotate('G',(0.77,0.45),fontsize=16)
    axes.annotate('H',(0.9,0.71),fontsize=16)

    axes.plot([m*1.0/(2*N),m*1.0/N+n*1.0/(2*N)],[ybar,zbar],'ko',markersize=5)
    axes.plot([ptG[0]],[ptG[1]],'ko',markersize=5)
    axes.axis([-0.1,1.1,0.25,0.8])

# Fig. 0
def fig0_create(fnstr,elecstr,elections,seatsbool=False):
    """
    """
    fig = plt.figure(figsize=(scalef*8,scalef*4),dpi=mydpi)
    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    ax1 = fig.gca()

    fig1_plot_angle(elecstr,ax1,fig,elections,seatsbool)

    plt.tight_layout(w_pad=1,h_pad=1)

    output_fig(fig,fnstr)

    plt.close()

# Fig. 1
def fig1_txpa_heat(axes,fig,elections):
    """ Make grid of angles for paper
    """

    ######################
    # Texas 1974
    # sns.set_style("ticks")
    sns.despine()

    elec = elections['1974_TX_11']
    # Texas 1974
    fig1_plot_one_declination(axes[0],elec.demfrac,'',True,plotdec=True,xaxislab=True,yaxislab=True)

    ############################
    # explore number of stolen seats
    elec = elections['2012_PA_11']
    # Pennsylvania 2012
    fig1_plot_one_declination(axes[1],elec.demfrac,'',True,\
                         plotdec=True,xaxislab=False,yaxislab=False,plotfullreg=True)

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
    cdict = {'red':   [(0.0,  1.0, 1.0),(0.2, 1.0, 1.0),(1.0,  0.0, 0.0)],\
             'green': [(0.0,  1.0, 1.0),(0.2, 1.0, 1.0),(1.0,  0.0, 0.0)],\
             'blue':  [(0.0,  1.0, 1.0),(1.0,  1.0, 1.0)]}
    rpb_color = LinearSegmentedColormap('rpb',cdict)

    df = pd.DataFrame(arr) # , index=index)
    nax = sns.heatmap(df, ax=axes[2], cmap=rpb_color, linewidths=0, vmin=0.3, vmax=0.7)
    nax.set_xlabel('Dem. vote share by district')

    nax.set_xticks([])
    nax.set_yticks([])

    #########################################
    # vote distribution for hypothetical plan
    # Hypothetical election
    fig1_plot_one_declination(axes[3],onedarr,'',True,plotdec=True,xaxislab=True,yaxislab=True)

#################################################################################
# Fig 1.
def fig1_plot_one_declination(axe,arr,title,plotslopes=True,lrg=True,plotdec=True,xaxislab=True,yaxislab=True,
                              plotfullreg=False,ylab='Dem. vote'):
    """ Make grid of angles for paper
    """
    axe.set_axis_bgcolor('none')
    axe.xaxis.set_ticks_position('bottom')
    axe.yaxis.set_ticks_position('left')

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

    if lrg:
        lrg_sz = 60
        lrg_marksz = 5
        lrg_lw = 3
        fs = 14
        for item in ([axe.title, axe.xaxis.label, axe.yaxis.label] +\
                     axe.get_xticklabels() + axe.get_yticklabels()):
            item.set_fontsize(fs)
    else:
        lrg_sz = 420
        lrg_marksz = 20
        lrg_lw = 9
        fs = 50
        for item in ([axe.title, axe.xaxis.label, axe.yaxis.label] +\
                     axe.get_xticklabels() + axe.get_yticklabels()):
            item.set_fontsize(fs)
        
    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)

    # plot values of metrics
    if plotdec:
        fa = get_declination('',vals) # *math.log(len(vals))/2
        eg = get_tau_gap(vals,0)
        tstr = "D vote = " + ("% .2f" % (np.mean(vals)))
        if abs(fa) >= 2:
            tmpstr  = '$\\delta = N/A$'
            tmpstr2 = 'Seats = N/A'
            tmpstr3 = 'EG = ' + ("% .2f" % (eg))
        else:
            if fa >= 0:
                tmpstr = '$\ \ \ \ {\\delta} = ' + ("% .2f$" % (fa))
                tmpstr2 = 'Seats = ' + ("% .1f" % (fa*5.0*len(arr)/12))
                tmpstr3 = 'EG = ' + ("% .2f" % (eg))
            else:
                tmpstr = '$\\delta = ' + ("% .2f$" % (fa))                
                tmpstr2 = 'Seats = ' + ("% .1f" % (fa*5.0*len(arr)/12))
                tmpstr3 = 'EG = ' + ("% .2f" % (eg))
        # axe.annotate(tmpstr, (0.02,0.84))
        if lrg:
            # axe.annotate(tstr, (0.65,0.14), fontsize=fs)
            axe.annotate(tmpstr, (0.6,0.10), fontsize=fs)
            # axe.annotate(tmpstr2, (0.65,0.06), fontsize=fs)
            # axe.annotate(tmpstr3, (0.65,0.02), fontsize=fs)
        else:
            # axe.annotate(tstr, (0.5,0.14), fontsize=fs)
            axe.annotate(tmpstr, (0.5,0.10), fontsize=fs)
            # axe.annotate(tmpstr2, (0.5,0.06), fontsize=fs)
            # axe.annotate(tmpstr3, (0.5,0.02), fontsize=fs)

    axe.get_xaxis().set_ticks([])
    axe.set_ylim(0,1)
    axe.set_xlim(0,1)
    axe.set_title(title,fontsize=fs) # elec.state + ' ' + elec.yr)
    if yaxislab:
        axe.set_ylabel(ylab)
    if xaxislab:
        axe.set_xlabel('District')

    if m > 0 and n > 0 and plotslopes:
        # plot angled lines
        ybar = np.mean(vals[:m])
        zbar = np.mean(vals[m:])

        med_marksz = lrg_marksz*2.0/3
        axe.plot([m*1.0/(2*N),m*1.0/N], [ybar,0.5], 'k-', linewidth=lrg_lw)
        axe.plot([m*1.0/N,m*1.0/N+n*1.0/(2*N)], [0.5,zbar], 'k-', linewidth=lrg_lw)
        axe.plot([m*1.0/(2*N),m*1.0/N,m*1.0/N+n*1.0/(2*N)],[ybar,0.5,zbar],'ko',markersize=med_marksz)

        if plotfullreg:
            l = stats.linregress(x1+x2,vals)
            axe.plot([0,1],[l[1],l[1]+l[0]],'k-',linewidth=1)

    axe.scatter(x1,y1,color = myr,s=lrg_sz)
    axe.scatter(x2,y2,color = myb,s=lrg_sz)

    axe.set_axis_bgcolor('none')

# Fig. 1
def fig1_create(fnstr,elecstr,elections):
    """
    """
    fig = plt.figure(figsize=(scalef*8,scalef*5),dpi=mydpi)
    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    ax2 = plt.subplot2grid((2,2), (0,0), rowspan=1)
    ax3 = plt.subplot2grid((2,2), (0,1), rowspan=1)
    ax4 = plt.subplot2grid((2,2), (1,0), rowspan=1)
    ax5 = plt.subplot2grid((2,2), (1,1), rowspan=1)

    fig1_txpa_heat([ax2,ax3,ax4,ax5],fig,elections)

    plt.tight_layout(w_pad=1,h_pad=1)

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()

#####################################################################################
# Third picture
###################################################################################################
# Fig. 3
def fig3_scatter(elections,ax,mmd):
    """
    """
    ax.set_axis_bgcolor('none')
    xarr = []
    yarr = []
    for elec in elections.values():
        if (elec.chamber != 9 or elec.yr not in mmd or elec.state not in mmd[elec.yr]) and \
           int(elec.yr) >= 1972:
            fa = get_declination('',elec.demfrac)
            if abs(fa) < 2:
                xarr.append(elec.Ndists)
                yarr.append(abs(fa)*math.log(elec.Ndists)/2)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +\
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

    ax.set_axis_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlim([0,250])
    ax.set_ylim([0,1.75])
    # print "Total elections: %d" % (len(xarr))
    ax.scatter(xarr,yarr,color='darkgreen')

    ax.set_xlabel('Number of districts in election')
    ax.set_ylabel('$|\\tilde{\\delta}|$')

    # now make regression
    l = stats.linregress(xarr,yarr)
    # print "INFO: delta_e regression line: %.4f %.4f %.4f %.4f" % (l[0],l[1],l[2],l[3])
    ax.plot([0,250],[l[1],l[1]+250*l[0]],'k-',linewidth=1)

def fig_pack_crack_one(axl,axr,vals,col,sha,mylab,do_pack=True):
    """ pack a standard response distribution until you can't anymore; see what happens to angle
    """
    origfa = get_declination('',vals)
    # figure out district we're going to start with
    idx = 0
    while vals[idx] <= 0.5:
        idx += 1

    ans = [0]
    delx = 1.0/(2*len(vals))
    xvals = np.linspace(delx,1-delx,len(vals))
    tmp = [0]
    tmpx = [0]

    if do_pack:
        can_pack = True
        while can_pack:
            can_pack,narr = pack_or_crack(vals,False)
            vals = narr
            idx += 1
            
            ans.append([get_declination('',vals)-origfa,vals])
            tmp.append(ans[-1][0])
            tmpx.append(len(tmpx)*1.0/len(vals))
    else:
        can_crack = True
        while can_crack:
            can_crack,narr = pack_or_crack(vals,True)
            vals = narr
            idx += 1
            
            if idx < len(vals):
                ans.append([get_declination('',vals)-origfa,vals])
                tmp.append(ans[-1][0])
                tmpx.append(len(tmpx)*1.0/len(vals))
            else:
                can_crack = False

    blah = axl.plot(tmpx[:-1],tmp[:-1],marker=sha,linestyle='-',color=col,label=mylab)
    return blah

def fig_pack_crack_all(elections,axl,axr):
    """ illustrate what happens to declination after packing and cracking fraction of seats
    """
    N = 19
    resp = 2
    demfrac = 0.55

    delv = 1.0/(resp*N)
    adj = demfrac-0.5/resp
    vals = [min(max(0,adj+delv*i),1) for i in range(N)]
    if min(vals) < 0 or max(vals) > 1:
        print "Went out of range"
        return

    axl.set_xlabel('Fraction of districts packed/cracked')
    axl.set_ylabel('Dem. vote')
    axr.set_ylabel('Change in $\\delta$')

    shapes = ['x','o','^','+']

    lab3 = fig_pack_crack_one(axl,axr,sorted(elections['2012_CA_11'].demfrac),'orange',shapes[2],'CA',True)
    lab1 = fig_pack_crack_one(axl,axr,sorted(elections['2012_GA_11'].demfrac),'green',shapes[0],'GA',False)
    lab2 = fig_pack_crack_one(axl,axr,sorted(elections['2012_AZ_11'].demfrac),'blue',shapes[1],'AZ',False)

    axl.legend(loc='upper left')

def fig_variations(fnstr,elections,cycstates,mmd):
    """
    """
    plt.close()
    fig = plt.figure(figsize=(scalef*8,scalef*4),dpi=mydpi) 
    axl = plt.subplot2grid((1,2), (0,0))
    ax1 = plt.subplot2grid((1,2), (0,1))

    fig3_scatter(elections,ax1,mmd)

    fig_pack_crack_all(elections,axl,axl)
    axl.set_axis_bgcolor('none')

    axl.spines['top'].set_visible(False)
    axl.spines['right'].set_visible(False)
    axl.xaxis.set_ticks_position('bottom')
    axl.yaxis.set_ticks_position('left')

    axl.text(0.01,.95,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
    ax1.text(0.52,.95,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')

    plt.tight_layout()

    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()   

############################################################################################
# Fig. 3
def sub_make_heatmap(fn,curax,tmparr,allyrs,bins,xticks,yticks):
    """ actually make the image
    """

    arr = [[0 for i in range(len(bins)-1)] for j in range(len(allyrs))]

    for i in range(len(tmparr)):
        duh = sorted(tmparr[i])
        inds = np.digitize(tmparr[i],bins)
        # if len(tmparr[i]) > 0:
        #     print "%d %.2f %.2f" % (i,min(tmparr[i]),max(tmparr[i]))
        for x in inds:
            arr[i][min(max(1,x),len(bins)-1)-1] += 1

    index = []
    for i in range(len(allyrs)):
        if i%5 == 0:
            index.append(str(allyrs[i]))
        else:
            index.append(str(allyrs[i]))

    df = pd.DataFrame(np.array(arr), index=index)
    
    sns.set(font_scale=1.2)
    # plot it out
    N = df.max().max() - df.min().min() + 1
    cmap = cmap_discretize(plt.cm.Greens, N)

    nax = sns.heatmap(df, ax=curax, xticklabels=xticks, yticklabels=yticks,cmap=cmap, linewidths=.1)

def fig2_make_heatmap_e(fnstr,elections,mmd):
    """
    """
    
    minN = 8
    totcong = 0
    totstate = 0
    congang = []
    stateang = []
    congangN = []
    stateangN = []
    allyrs = [(1972+2*j) for j in range(23)]
    numbins = 8
    congtmparr = [[] for i in range(len(allyrs))]
    statetmparr = [[] for i in range(len(allyrs))]
    congNtmparr = [[] for i in range(len(allyrs))]
    stateNtmparr = [[] for i in range(len(allyrs))]
    minyr = 1972
    ccnt = 0
    scnt = 0
    for elec in elections.values():
        if elec.Ndists >= minN and elec.chamber == '11' and int(elec.yr) >= minyr and int(elec.yr)%2 == 0:
            totcong += 1
            fang = get_declination(elec.state,elec.demfrac)*math.log(elec.Ndists)/2
            if abs(fang) < 2:
                ccnt += 1
                congtmparr[allyrs.index(int(elec.yr))].append(fang)
                congNtmparr[allyrs.index(int(elec.yr))].append(fang*elec.Ndists*1.0/2)
                congang.append(fang)
                congangN.append(fang*elec.Ndists*1.0/2)
        if elec.Ndists >= minN and elec.chamber == '9' and int(elec.yr) >= minyr and \
           (elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]) and int(elec.yr)%2 == 0:
            totstate += 1
            fang = get_declination(elec.state,elec.demfrac)*math.log(elec.Ndists)/2
            if abs(fang) < 2:
                if int(elec.yr) < 2012:
                    scnt += 1
                    statetmparr[allyrs.index(int(elec.yr))].append(fang)
                    stateNtmparr[allyrs.index(int(elec.yr))].append(fang*elec.Ndists*1.0/2)
                    stateang.append(fang)
                    stateangN.append(fang*elec.Ndists*1.0/2)

    plt.close()   
    fig, axes = plt.subplots(1,2, figsize=(scalef*8,scalef*6),dpi=mydpi) 
    axes = axes.ravel()

    print "Total number of congressional races included (dec defined, at least %d seats): %d" % (minN,ccnt)
    print "Total number of state races included (dec defined, at least %d seats): %d" % (minN,scnt)

    numbins = 10
    bins = np.linspace(-1.2,0.8,numbins+1)
    # print bins
    yticks = [1972,1982,1992,2002,2012]
    sub_make_heatmap('heat-cong',axes[0],congtmparr,allyrs,bins,[-1.2,-0.8,-0.4,0,0.4,0.8],yticks)
    axes[0].set_title('Congressional elections')
    axes[0].set_xticks([0,2,4,6,8,10]) 
    axes[0].set_yticks([2.5,7.5,12.5,17.5,22.5])
    axes[0].axvline(6,color='black')

    numbins = 10
    bins = np.linspace(-1.5,1,numbins+1)
    yticks = [1972,1982,1992,2002]
    sub_make_heatmap('heat-state',axes[1],statetmparr,allyrs,bins,[-1.5,-1,-0.5,0,0.5,1],yticks)
    axes[1].set_xticks([0,2,4,6,8,10])
    axes[1].set_yticks([7.5,12.5,17.5,22.5])
    axes[1].set_title('State elections')
    axes[1].axvline(6,color='black')

    axes[0].text(0.01,.97,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
    axes[0].text(0.5,.97,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()   

############################################################################################
# 
def fig3_deltae_linechart(elections,cycstates,mmd,yrmin,ax,chamber='9',prtitle=True,num=5):
    """
    """
    ans = []
    cnt = 0
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +\
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_axis_bgcolor('none')
    ax.set_axis_on()
    for i,cycstate in enumerate(cycstates):
        if yrmin in mmd.keys() and state in mmd[yrmin]:
            continue
        stcnt = 0
        curmin = 10
        curmax = -10
        isokay = True
        szs = 0
        for yr in [str(int(yrmin)+2*j) for j in range(num)]:
            myid = '_'.join([yr,cycstate[:2],chamber])
            if myid in elections.keys(): # and elections[myid].Ndists >= 8:
                elec = elections[myid]
                if elec.Ndists <= 1 or elec.cyc_state != cycstate:
                    continue
                szs = 1.0*math.log(elec.Ndists)/2
                aval = get_declination(elec.state,elec.demfrac) 
                if aval == None or abs(aval) == 2:
                    continue
                if aval < curmin:
                    curmin = aval
                if aval > curmax:
                    curmax = aval
                cnt += 1
                stcnt += 1
        # if isokay:
        if curmin < 10 and curmax > -10 and stcnt >= 1: # and aval != None:
            if stcnt == 1:
                ans.append([cycstate,szs*(curmin+curmax)/2,szs*curmin-0.01,szs*curmax])
            else:
                ans.append([cycstate,szs*(curmin+curmax)/2,szs*curmin,szs*curmax])
    ans.sort(key=lambda x: x[1])
    
    ax.axvline(0,color='#bbbbbb',linewidth=1)
    ax.axvline(-0.5,color='#bbbbbb',linewidth=1)
    ax.axvline(0.5,color='#bbbbbb',linewidth=1)
    ax.set_xlabel('$\\tilde{\\delta}$')
    if prtitle:
        if int(yrmin) < 2010:
            ax.set_title(str(yrmin) + '-' + str(int(yrmin)+8),fontsize=16)
        else:
            ax.set_title(str(yrmin) + '-' + str(int(yrmin)+4),fontsize=16)

    sns.set(font_scale=0.8)
    vdiff = 0.1
    if chamber == '9':
        ax.axis([-1,1,0,len(ans)*vdiff+0.15])
    else:
        ax.axis([-1,1,0,len(ans)*vdiff+0.15])
    for j in range(1,len(ans)+1):
        i = j-1
        ax.plot([ans[i][2],ans[i][3]],[i*vdiff+0.05,i*vdiff+0.05],color='darkgreen')
        ax.annotate(ans[i][0],(ans[i][3]+0.01,i*vdiff))

    ax.set_yticks([])
    ax.set_axis_on()
    return cnt

def fig_deltae_only(fnstr,elections,cycstates,mmd):
    """
    """
    plt.close()
    plt.figure(figsize=(scalef*5,scalef*5),dpi=mydpi) 

    fig3_deltae_linechart(elections,cycstates,mmd,2012,plt.gca(),'11',False)

    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()   

#######################################################################################
####################################################################################################
# Fig S5
def figS5_wi_scatter(fnstr,elections,mmd):
    lang = [[] for i in range(5)]
    lzgap = [[] for i in range(5)]
    cnt = 0
    # (elec.yr not in mmd.keys() or elec.state not in mmd['1972']) and \
    for elec in elections.values():
        if 2010 >= int(elec.yr) >= 1972 and \
           ('_'.join(['1972',elec.state,'9']) in elections.keys()) and \
           ('1972' not in mmd or elec.state not in mmd['1972']) and \
            elec.chamber=='9':
            ang = get_declination(elec.state,elec.demfrac)
            zgap = get_tau_gap(elec.demfrac,0)

            yridx = int((int(elec.yr)-1972)/10)
            if abs(ang) != 2:
                cnt += 1
                lang[yridx].append(ang)
                lzgap[yridx].append(zgap)
            # print "% .3f % .3f %3d %s %s %2d" % (ang,zgap,elec.Ndists,elec.yr,elec.state,int(elec.chamber))
    print "WI scatter - total number of races: ",cnt
    plt.figure(figsize=(scalef*8,scalef*8),dpi=mydpi)
    plt.gca().set_axis_bgcolor('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

    plt.axis([-0.6,0.6,-0.6,0.6])
    plt.axvline(0,color='black',ls='dotted')
    plt.axhline(0,color='black',ls='dotted')
    cols = ['#ff0000','#009000','#00a0a0','#900090']
    markers = ['o','o','o','o']
    legs = []
    for i in range(4):
        tmp = plt.scatter(lang[i],lzgap[i],color=cols[i],marker=markers[i])
        # print i,np.mean(lang[i]),np.mean(lzgap[i])
        legs.append(tmp)
        # print np.std(lang)
        # print np.std(lzgap)

    print "WI scatter - Correlation r and p given by:"
    print stats.pearsonr(lang[0] + lang[1] + lang[2] + lang[3],lzgap[0] + lzgap[1] + lzgap[2] + lzgap[3])

    plt.legend(legs,('1972-1980','1982-1990','1992-2000','2002-2010'),loc='upper left')
    plt.xlabel('Declination',fontsize=18)
    plt.ylabel('Twice the efficiency gap',fontsize=18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    datax = [0.196,0.165,0.140]
    datay = [0.204,0.157,0.136]
    plt.scatter(datax,datay,color='black',marker='o',s=100)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()

#########################################################################################
# Fig. 1
def fig_discuss(fnstr,elections):
    """
    """
    fig, axes = plt.subplots(1,3, figsize=(scalef*10,scalef*3),dpi=mydpi) 
    axes = axes.ravel()

    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    axes[0].text(0.01,.92,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
    axes[1].text(0.33,.92,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')
    axes[1].text(0.66,.92,'C',fontsize=16, transform=fig.transFigure, fontweight='bold')

    elec = elections['1974_NC_11']
    fig1_plot_one_declination(axes[0],elec.demfrac,'',True,plotdec=True,xaxislab=True,yaxislab=True)

    elec = elections['2006_TN_11']
    fig1_plot_one_declination(axes[1],elec.demfrac,'',True,plotdec=True,xaxislab=True,yaxislab=False)

    ############################
    # explore number of stolen seats
    elec = elections['2012_IN_11']
    fig1_plot_one_declination(axes[2],elec.demfrac,'',True,\
                              plotdec=True,xaxislab=True,yaxislab=False,plotfullreg=False)

    plt.tight_layout() 

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()

##############################################################################################
def figS12_split_dec_states(fnstr,r,c,yr,chamber,states,elections,mymmd):
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
                nstates.append([get_declination(x,elec.demfrac),elec])
    nstates.sort(key=lambda x: x[0])
    nstates.reverse()
    
    fig, axes = plt.subplots(r,c, figsize=(10*c,10*r), dpi=mydpi, sharex=True, sharey=True)
    axes = axes.ravel()

    for i,st in enumerate(nstates):
        xlab = (i > (r-1)*c)
        ylab = (i%c==0)
        fig1_plot_one_declination(axes[i],st[1].demfrac,st[1].state,plotslopes=True,lrg=False,\
                                  plotdec=True,xaxislab=xlab,yaxislab=ylab,plotfullreg=False)
    for i in range(len(nstates),r*c):
        axes[i].set_axis_off()

    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    # plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()

############################################################################################
def figS34_linechart(fnstr,elections,cycstates,mmd,chamber,verbose=False):
    """
    """
    plt.close()
    if chamber == '11':
        fig, axes = plt.subplots(3,2, figsize=(scalef*10,scalef*15),dpi=mydpi)
    else:
        fig, axes = plt.subplots(2,2, figsize=(scalef*10,scalef*12),dpi=mydpi)
    axes = axes.ravel()    

    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    if chamber == '11':
        axes[0].text(0.01,.98,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
        axes[1].text(0.5,.98,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')
        axes[2].text(0.01,.65,'C',fontsize=16, transform=fig.transFigure, fontweight='bold')
        axes[3].text(0.5,.65,'D',fontsize=16, transform=fig.transFigure, fontweight='bold')
        axes[4].text(0.01,.32,'E',fontsize=16, transform=fig.transFigure, fontweight='bold')
    else:
        axes[0].text(0.01,.96,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
        axes[1].text(0.5,.96,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')
        axes[2].text(0.01,.48,'C',fontsize=16, transform=fig.transFigure, fontweight='bold')
        axes[3].text(0.5,.48,'D',fontsize=16, transform=fig.transFigure, fontweight='bold')

    tot = 0
    tot += fig3_deltae_linechart(elections,cycstates,mmd,1972,axes[0],chamber,False)
    tot += fig3_deltae_linechart(elections,cycstates,mmd,1982,axes[1],chamber,False)
    tot += fig3_deltae_linechart(elections,cycstates,mmd,1992,axes[2],chamber,False)
    tot += fig3_deltae_linechart(elections,cycstates,mmd,2002,axes[3],chamber,False)
    if chamber == '11':
        tot += fig3_deltae_linechart(elections,cycstates,mmd,2012,axes[4],chamber,False)
        axes[5].set_axis_off()

    if verbose:
        print "Total number of elections included for %s is %d" % (chamber, tot)
    axes[0].set_axis_on()
    axes[1].set_axis_on()
    axes[2].set_axis_on()
    plt.tight_layout()

    output_fig(fig,fnstr)
    plt.close()   

#################################################################################################

def make_elj_pics(arr):
    """ convenience function
    """
    if 1 in arr:
        sns.reset_orig()
        fig0_create('fig0-dec-def','2014_NC_11',Nelections)
    if 2 in arr:
        sns.reset_orig()
        fig1_create('fig1-dec-ex','2014_NC_11',Nelections)
    if 3 in arr:
        sns.reset_orig()
        fig_variations('fig-var',Nelections,Ncyc,Nmmd)
    if 4 in arr:
        sns.reset_orig()
        fig2_make_heatmap_e('fig2-heatmap-e',Nelections,Nmmd)
    if 5 in arr:
        sns.reset_orig()
        fig_deltae_only('fig-deltae-only',Nelections,Ncyc,Nmmd)
    if 6 in arr:
        sns.reset_orig()
        figS5_wi_scatter('scatter-wi-nommd',lelecs,lmmd)
    if 7 in arr:
        sns.reset_orig()
        fig_discuss('fig-discuss',Nelections)

    if 8 in arr:
        sns.reset_orig()
        figS12_split_dec_states('figS1-cong2016',7,5,'2016','11',Nstates,Nelections,Nmmd)
    if 9 in arr:
        sns.reset_orig()
        figS12_split_dec_states('figS2-st2008',7,5,'2008','9',Nstates,Nelections,Nmmd)

    if 10 in arr:
        sns.reset_orig()
        figS34_linechart('figS4-stateline',Nelections,Ncyc,Nmmd,'9',True)
    if 11 in arr:
        sns.reset_orig()
        figS34_linechart('figS3-congline',Nelections,Ncyc,Nmmd,'11',True)

# make_elj_pics(arr=range(1,12))
# make_elj_pics(arr=[4]) # range(1,12))
