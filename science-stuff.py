from PIL import Image
import cStringIO
from subprocess import call, check_output
import subprocess
import os  
import matplotlib.patches as patches

myr = '#ff8080'
myb = '#8080ff'

fileconv = ['fig0-dec-def','fig1-dec-ex','fig-var','fig2-heatmap-e','fig-deltae-only',\
            'scatter-wi-nommd','fig-discuss','figS1-cong2016','figS2-st2008','figS3-congline',\
            'figS4-stateline']

scalef = 1
mydpi = 100

# mpl.rcParams['figure.figsize'] = [8.0, 6.0]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'

def make_tiffs():
    """ make suitable TIFF files
    """
    myfiles = ['fig0-dec-def','fig1-dec-ex','fig-var','fig2-heatmap-e','fig-deltae-only',\
            'scatter-wi-nommd','fig-discuss','figS1-cong2016','figS2-st2008','figS3-congline',\
            'figS4-stateline']
    homepath = '/home/gswarrin/research/gerrymander/pics/'
    cnt = 1
    for f in myfiles:
        outf = 'WarringtonFig' + ("%02d" % (cnt)) + '.tiff'
        check_output(['convert', homepath + f + '.png', '-colorspace', 'CMYK', homepath + outf])
        cnt += 1
        
def get_tiff_name(fn):
    idx = fileconv.index(fn)+1
    return 'WarringtonFig' + ("%02d" % (idx)) + '.tiff'

def output_fig(fig,fnstr):
    """ depends on global argument to determine output format
    """
    #######################
    # save as a png
    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    #######################
    # save as a tiff
    # Save the image in memory in PNG format
    # png1 = cStringIO.StringIO()
    # fig.savefig(png1, format="png")

    # Load this image into PIL
    # png2 = Image.open(png1).convert("CMYK")

    # Save as TIFF
    # png2.save('/home/gswarrin/research/gerrymander/pics/' + get_tiff_name(fnstr))
    # png1.close()

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in xrange(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

###########################################
# functions to back up claims in paper
# Fig 1:
# - declination for North Carolina 2014
# - declination + mean votes for Texas 1974
# - declination + mean votes + regression line + egap for PA 2012
# - declination for hypothetical election
# Fig 2:
# - number of elections in each heat map
# Fig 3:
# - 

############################################################################
# Fig 1.
def fig1_plot_angle(k,axes,fig,elections,seatsbool=False):
    """ copied from plot_angle
    """
    # plt.figure(figsize=(10,5),frameon=False)
    # axes = plt.gca()
    # axes.set_axis_bgcolor('none')
    # axes.set_frame_on(True)
    # axes.set_axis_on()
    axes.get_xaxis().set_ticks([])
    axes.set_xlabel('District')

    # plt.set_frame_on(True)
    elec = elections[k]
    vals = sorted(elec.demfrac)
    N = len(vals)
    m = len(filter(lambda x: x < 0.5, vals))
    n = N-m
    # ans,m1,i1,m2,i2 = find_lines(elec.state,vals)
    ybar = np.mean(vals[:m])
    zbar = np.mean(vals[m+1:])

    # axes.patch.set_visible(True)
    # fig.patch.set_visible(True)
    # axes.grid(False)
    # axes.set_facecolor('green')
    
    for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +\
                 axes.get_xticklabels() + axes.get_yticklabels()):
        item.set_fontsize(14)

    # plot actual vote fractions
    x1 = [i*1.0/N-1.0/(2*N) for i in range(1,m+1)]
    x2 = [m*1.0/N + i*1.0/N-1.0/(2*N) for i in range(1,n+1)]
    y1 = vals[:m]
    y2 = vals[m:]
    axes.scatter(x1,y1,color = 'red',s=60)
    axes.scatter(x2,y2,color = 'blue',s=60)
    
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

    # line, = axes.plot([ptT[0],ptG[0],ptF[0]], [ptT[1],ptG[1],ptF[1]], 'k-', lw=2)
    #    line2, = axes.plot([ptU[0],ptG[0],ptH[0]], [ptU[1],ptG[1],ptH[1]], 'k-', lw=2)

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

    # axes.text(0.01,.97,'A',fontsize=16, transform=fig.transFigure, fontweight='bold') # , va='top', ha='right')

    axes.plot([ptG[0],1],[ptG[1],ptG[1]+(1-ptG[0])*(0.5-ybar)/(ptG[0]-ptF[0])],'k-.')
    axes.add_patch(matplotlib.patches.Arc(ptG, .4, .4, 0, 16.5, 65, color='green',lw=3))
    # plt
    axes.annotate('$\\delta\\pi/2$',(0.94,0.63),fontsize=16)
    axes.annotate('F',(0.38,0.33),fontsize=16)
    axes.annotate('G',(0.77,0.45),fontsize=16)
    axes.annotate('H',(0.9,0.71),fontsize=16)
    # axes.xlabel('District')
    # axes.ylabel('Dem. vote')
    axes.plot([m*1.0/(2*N),m*1.0/N+n*1.0/(2*N)],[ybar,zbar],'ko',markersize=5)
    axes.plot([ptG[0]],[ptG[1]],'ko',markersize=5)
    axes.axis([-0.1,1.1,0.25,0.8])
    # axes.set_axis_bgcolor('none')    
    # axes.grid(False)

    # turns grey part on or off
    # axes.patch.set_visible(False)

    # add_corner_arc(axes, line, text='$\\theta$') # u'%d\u00b0' % 90)
    # add_corner_arc(axes, line2, text='$\\phi$') # u'%d\u00b0' % 90)
    # add_corner_arc(ax, line, radius=.7, color='black', text=None, text_radius=.5, text_rotatation=0, **kwargs):

    # plt.set_xticks([1,N])
    # plt.set_ylim(0,1)
    yr,st,chm = k.split('_')
    # plt.set_title(st + ' ' + yr)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/paper-angle-plot-' + k + '.png')
    # plt.show()
    # plt.close()   

############################################################################
# Fig 1.
def talk_fig1_plot_angle(k,axes,fig,elections,seatsbool=False):
    """ copied from plot_angle
    """
    # plt.figure(figsize=(10,5),frameon=False)
    # axes = plt.gca()
    # axes.set_axis_bgcolor('none')
    # axes.set_frame_on(True)
    # axes.set_axis_on()
    axes.get_xaxis().set_ticks([])
    axes.set_xlabel('District')

    # plt.set_frame_on(True)
    elec = elections[k]
    vals = sorted(elec.demfrac)
    N = len(vals)
    m = len(filter(lambda x: x < 0.5, vals))
    n = N-m
    # ans,m1,i1,m2,i2 = find_lines(elec.state,vals)
    ybar = np.mean(vals[:m])
    zbar = np.mean(vals[m+1:])

    # axes.patch.set_visible(True)
    # fig.patch.set_visible(True)
    # axes.grid(False)
    # axes.set_facecolor('green')
    
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

    # line, = axes.plot([ptT[0],ptG[0],ptF[0]], [ptT[1],ptG[1],ptF[1]], 'k-', lw=2)
    #    line2, = axes.plot([ptU[0],ptG[0],ptH[0]], [ptU[1],ptG[1],ptH[1]], 'k-', lw=2)

    # if not seatsbool:
        # plt
        # axes.annotate('$\\theta_P$',(0.85,0.57),fontsize=16)
        # axes.add_patch(matplotlib.patches.Arc(ptG, .4, .4, 0, 180, 196.5, color='green',lw=3))
        # plt
        # axes.annotate('$\\theta_Q$',(0.52,0.46),fontsize=16)

        # axes.add_patch(matplotlib.patches.Arc(ptG, .2, .2, 0, 0, 65, color='green',lw=3))

        # axes.annotate('T',(0,0.45),fontsize=16)
        # axes.annotate('U',(1,0.45),fontsize=16)
        # axes.plot([ptT[0],ptU[0]],[ptT[1],ptU[1]],'ko',markersize=5)

    # axes.text(0.01,.97,'A',fontsize=16, transform=fig.transFigure, fontweight='bold') # , va='top', ha='right')

    axes.plot([ptG[0],1],[ptG[1],ptG[1]+(1-ptG[0])*(0.5-ybar)/(ptG[0]-ptF[0])],'k-.')
    axes.add_patch(matplotlib.patches.Arc(ptG, .4, .4, 0, 16.5, 65, color='green',lw=3))
    # plt
    axes.annotate('$\\delta\\pi/2$',(0.94,0.63),fontsize=16)
    # axes.annotate('F',(0.38,0.33),fontsize=16)
    # axes.annotate('G',(0.77,0.45),fontsize=16)
    # axes.annotate('H',(0.9,0.71),fontsize=16)
    # axes.xlabel('District')
    # axes.ylabel('Dem. vote')
    axes.plot([m*1.0/(2*N),m*1.0/N+n*1.0/(2*N)],[ybar,zbar],'ko',markersize=5)
    axes.plot([ptG[0]],[ptG[1]],'ko',markersize=5)
    axes.axis([-0.1,1.1,0.25,0.8])
    # axes.set_axis_bgcolor('none')    
    # axes.grid(False)

    # turns grey part on or off
    # axes.patch.set_visible(False)

    # add_corner_arc(axes, line, text='$\\theta$') # u'%d\u00b0' % 90)
    # add_corner_arc(axes, line2, text='$\\phi$') # u'%d\u00b0' % 90)
    # add_corner_arc(ax, line, radius=.7, color='black', text=None, text_radius=.5, text_rotatation=0, **kwargs):

    # plt.set_xticks([1,N])
    # plt.set_ylim(0,1)
    yr,st,chm = k.split('_')
    axes.set_title(st + ' ' + yr)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/paper-angle-plot-' + k + '.png')
    # plt.show()
    # plt.close()   

# Fig 1.
def web_plot_one_declination(axe,arr,title,plotslopes=True,lrg=True,plotdec=True,xaxislab=True,yaxislab=True,plotfullreg=False,ylab='Dem. vote'):
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

    # print "title: ",title

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
        fs = 100
        for item in ([axe.title, axe.xaxis.label, axe.yaxis.label] +\
                     axe.get_xticklabels() + axe.get_yticklabels()):
            item.set_fontsize(fs)
        
    # print "asdfasdf"

    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)

    axe.scatter(x1,y1,color = 'red',s=lrg_sz)
    axe.scatter(x2,y2,color = 'blue',s=lrg_sz)

    # plot values of metrics
    if plotdec:
        fa = get_declination('',vals) # *math.log(len(vals))/2
        eg = get_tau_gap(vals,0)
        tstr = "D vote = " + ("% .2f" % (np.mean(vals)))
        if abs(fa) >= 2:
            tmpstr  = '$\\delta = N/A$'
            tmpstr2 = 'Seats = N/A'
            # tmpstr3 = 'EG = ' + ("% .2f" % (eg))
        else:
            if fa > 0:
                tmpstr = 'Decl. = ' + ("% .2f" % (fa))
                tmpstr2 = 'Seats = ' + ("% .1f" % (fa*5.0*len(arr)/12))
                # tmpstr3 = 'EG = ' + ("% .2f" % (eg))
            else:
                tmpstr = 'Decl. = ' + ("% .2f" % (fa))                
                tmpstr2 = 'Seats = ' + ("% .1f" % (fa*5.0*len(arr)/12))
                # tmpstr3 = 'EG = ' + ("% .2f" % (eg))
        # axe.annotate(tmpstr, (0.02,0.84))
        if lrg:
            axe.annotate(tstr, (0.65,0.14), fontsize=fs)
            axe.annotate(tmpstr, (0.65,0.10), fontsize=fs)
            axe.annotate(tmpstr2, (0.65,0.06), fontsize=fs)
            # axe.annotate(tmpstr3, (0.65,0.02), fontsize=fs)
        else:
            axe.annotate(tstr, (0.5,0.14), fontsize=fs)
            axe.annotate(tmpstr, (0.5,0.10), fontsize=fs)
            axe.annotate(tmpstr2, (0.5,0.06), fontsize=fs)
            # axe.annotate(tmpstr3, (0.5,0.02), fontsize=fs)

    # print "----"

    axe.get_xaxis().set_ticks([])
    axe.set_ylim(0,1)
    axe.set_xlim(0,1)
    axe.set_title(title,fontsize=fs) # elec.state + ' ' + elec.yr)
    if yaxislab:
        axe.set_ylabel(ylab)
    if xaxislab:
        axe.set_xlabel('District')

    # print "===="

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
            # print "regress line: ",l
            axe.plot([0,1],[l[1],l[1]+l[0]],'k-',linewidth=1)

    axe.set_axis_bgcolor('none')

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

    # print "title: ",title

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
        fs = 100
        for item in ([axe.title, axe.xaxis.label, axe.yaxis.label] +\
                     axe.get_xticklabels() + axe.get_yticklabels()):
            item.set_fontsize(fs)
        
    # print "asdfasdf"

    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)

    # print y2

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

    # print "----"

    axe.get_xaxis().set_ticks([])
    axe.set_ylim(0,1)
    axe.set_xlim(0,1)
    axe.set_title(title,fontsize=fs) # elec.state + ' ' + elec.yr)
    if yaxislab:
        axe.set_ylabel(ylab)
    if xaxislab:
        axe.set_xlabel('District')

    # print "===="

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
            # print "regress line: ",l
            axe.plot([0,1],[l[1],l[1]+l[0]],'k-',linewidth=1)

    axe.scatter(x1,y1,color = myr,s=lrg_sz)
    axe.scatter(x2,y2,color = myb,s=lrg_sz)

    axe.set_axis_bgcolor('none')

# Fig 1.
def talk_plot_one_declination(axe,arr,title,plotslopes=True,lrg=True,plotdec=True,xaxislab=True,yaxislab=True,plotfullreg=False,ylab='Dem. vote'):
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

    # print "title: ",title

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
        fs = 100
        for item in ([axe.title, axe.xaxis.label, axe.yaxis.label] +\
                     axe.get_xticklabels() + axe.get_yticklabels()):
            item.set_fontsize(fs)
        
    # print "asdfasdf"

    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)

    axe.scatter(x1,y1,color = myr,s=lrg_sz)
    axe.scatter(x2,y2,color = myb,s=lrg_sz)

    # plot values of metrics
    if plotdec:
        fa = get_declination('',vals) # *math.log(len(vals))/2
        eg = get_tau_gap(vals,0)
        # tstr = "D vote = " + ("% .2f" % (np.mean(vals)))
        if abs(fa) >= 2:
            tmpstr  = '$\\delta = N/A$'
            tmpstr2 = 'Seats = N/A'
        else:
            if fa > 0:
                tmpstr = '$\ \ \ \ {\\delta} = ' + ("% .2f$" % (fa))
                tmpstr2 = 'Seats = ' + ("% .1f" % (fa*5.0*len(arr)/12))
            else:
                tmpstr = '$\\delta = ' + ("% .2f$" % (fa))                
                tmpstr2 = 'Seats = ' + ("% .1f" % (fa*5.0*len(arr)/12))
        # axe.annotate(tmpstr, (0.02,0.84))
        if lrg:
            # axe.annotate(tstr, (0.65,0.14), fontsize=fs)
            axe.annotate(tmpstr, (0.65,0.14), fontsize=fs)
            axe.annotate(tmpstr2, (0.65,0.06), fontsize=fs)
        else:
            # axe.annotate(tstr, (0.5,0.14), fontsize=fs)
            axe.annotate(tmpstr, (0.5,0.14), fontsize=fs)
            axe.annotate(tmpstr2, (0.5,0.06), fontsize=fs)

    # print "----"

    axe.get_xaxis().set_ticks([])
    axe.set_ylim(0,1)
    axe.set_xlim(0,1)
    axe.set_title(title,fontsize=fs) # elec.state + ' ' + elec.yr)
    if yaxislab:
        axe.set_ylabel(ylab)
    if xaxislab:
        axe.set_xlabel('District')

    # print "===="

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
            # print "regress line: ",l
            axe.plot([0,1],[l[1],l[1]+l[0]],'k-',linewidth=1)

    axe.set_axis_bgcolor('none')

# Fig 1.
def fig1_plot_eg(axe,arr,title,plotslopes=True,lrg=True,plotdec=True,xaxislab=True,yaxislab=True,plotfullreg=False,ylab='Dem. vote'):
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
        fs = 60
        for item in ([axe.title, axe.xaxis.label, axe.yaxis.label] +\
                     axe.get_xticklabels() + axe.get_yticklabels()):
            item.set_fontsize(fs)
        
    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)

    axe.scatter(x1,y1,color = 'red',s=lrg_sz)
    axe.scatter(x2,y2,color = 'blue',s=lrg_sz)

    # plot values of metrics
    if plotdec:
        fa = get_tau_gap(vals,0) # *math.log(len(vals))/2
        if abs(fa) >= 2:
            tmpstr = '$\\delta = N/A'
        else:
            if fa > 0:
                tmpstr = '   ' + ("% .2f" % (fa/2))
            else:
                tmpstr = '   ' + ("% .2f" % (fa/2))                
        # axe.annotate(tmpstr, (0.02,0.84))
        if lrg:
            axe.annotate(tmpstr, (0.65,0.05), fontsize=fs)
        else:
            axe.annotate(tmpstr, (0.5,0.05), fontsize=fs)

    axe.get_xaxis().set_ticks([])
    axe.set_ylim(0,1)
    axe.set_xlim(0,1)
    axe.set_title(title,fontsize=fs) # elec.state + ' ' + elec.yr)
    if yaxislab:
        axe.set_ylabel(ylab)
    if xaxislab:
        axe.set_xlabel('District')

    # if m > 0 and n > 0 and plotslopes:
    #     # plot angled lines
    #     ybar = np.mean(vals[:m])
    #     zbar = np.mean(vals[m:])

    #     med_marksz = lrg_marksz*2.0/3
    #     axe.plot([m*1.0/(2*N),m*1.0/N], [ybar,0.5], 'k-', linewidth=lrg_lw)
    #     axe.plot([m*1.0/N,m*1.0/N+n*1.0/(2*N)], [0.5,zbar], 'k-', linewidth=lrg_lw)
    #     axe.plot([m*1.0/(2*N),m*1.0/N,m*1.0/N+n*1.0/(2*N)],[ybar,0.5,zbar],'ko',markersize=med_marksz)

    #     if plotfullreg:
    #         l = stats.linregress(x1+x2,vals)
    #         print "regress line: ",l
    #         axe.plot([0,1],[l[1],l[1]+l[0]],'k-',linewidth=1)

    for i in range(m):
        axe.add_patch(patches.Rectangle((i*1.0/N + 1.0/(4*N),0.0),1.0/(2*N),vals[i],facecolor='blue',alpha=0.5))
        axe.add_patch(patches.Rectangle((i*1.0/N + 1.0/(4*N),vals[i]),1.0/(2*N),0.5-vals[i],facecolor='red',alpha=0.5))

    for i in range(m,m+n):
        axe.add_patch(patches.Rectangle((i*1.0/N + 1.0/(4*N),vals[i]),1.0/(2*N),0.5-vals[i],facecolor='blue',alpha=0.5))
        axe.add_patch(patches.Rectangle((i*1.0/N + 1.0/(4*N),vals[i]),1.0/(2*N),1-vals[i],facecolor='red',alpha=0.5))

    axe.set_axis_bgcolor('none')

# Fig. 1
def fig1_txpa_heat(axes,fig,elections):
    """ Make grid of angles for paper
    """
    # Changed labels
    # axes[0].text(0.01,.94,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
    # axes[1].text(0.5,.94,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')
    # axes[2].text(0.01,.46,'C',fontsize=16, transform=fig.transFigure, fontweight='bold')
    # axes[3].text(0.5,.46,'D',fontsize=16, transform=fig.transFigure, fontweight='bold')

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
    # axe.plot([m*1.0/(2*N),m*1.0/N], [ybar,0.5], 'k-', linewidth=3)

    ############################
    # hypothetical district plan
    # sns.set_style("white")
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
    # ,\
    #         'alpha': [(0.0,  0.0, 0.0),(1.0,  1.0, 1.0)]}
    rpb_color = LinearSegmentedColormap('rpb',cdict)

    # print get_declination('',onedarr),get_tau_gap(onedarr,0),np.mean(onedarr)

    df = pd.DataFrame(arr) # , index=index)
    nax = sns.heatmap(df, ax=axes[2], cmap=rpb_color, linewidths=0, vmin=0.3, vmax=0.7)
    nax.set_xlabel('Dem. vote share by district')
    # nax.set_title('Hypothetical district plan')
    nax.set_xticks([])
    nax.set_yticks([])
    # nax.axis('off')
    # arr = [[.3,.4,.5],[.35,.45,.55]]

    #########################################
    # vote distribution for hypothetical plan
    # Hypothetical election
    fig1_plot_one_declination(axes[3],onedarr,'',True,plotdec=True,xaxislab=True,yaxislab=True)

    # plt.tight_layout()
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr + '-new.png')
    # plt.close()   

# Fig. 0
def fig0_create(fnstr,elecstr,elections,seatsbool=False):
    """
    """
    fig = plt.figure(figsize=(scalef*8,scalef*4),dpi=mydpi)
    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    ax1 = fig.gca()
    # ax1 = plt.subplot2grid((7,2), (0,0), colspan=2, rowspan=3)
    # ax2 = plt.subplot2grid((7,2), (3,0), rowspan=2)
    # ax3 = plt.subplot2grid((7,2), (3,1), rowspan=2)
    # ax4 = plt.subplot2grid((7,2), (5,0), rowspan=2)
    # ax5 = plt.subplot2grid((7,2), (5,1), rowspan=2)

    # sns.set_style("ticks")
    # subplot illustrating definition of declination
    fig1_plot_angle(elecstr,ax1,fig,elections,seatsbool)

    # specific examples of declination
    # fig1_txpa_heat([ax2,ax3,ax4,ax5],fig,elections)

    plt.tight_layout(w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    output_fig(fig,fnstr)
    # with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
    #     fig.canvas.print_png(outfile)

    plt.close()

# Fig. 0
def talk_fig0_create(fnstr,elecstr,elections,seatsbool=False):
    """
    """
    fig = plt.figure(figsize=(scalef*8,scalef*4),dpi=mydpi)
    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    ax1 = fig.gca()
    # ax1 = plt.subplot2grid((7,2), (0,0), colspan=2, rowspan=3)
    # ax2 = plt.subplot2grid((7,2), (3,0), rowspan=2)
    # ax3 = plt.subplot2grid((7,2), (3,1), rowspan=2)
    # ax4 = plt.subplot2grid((7,2), (5,0), rowspan=2)
    # ax5 = plt.subplot2grid((7,2), (5,1), rowspan=2)

    # sns.set_style("ticks")
    # subplot illustrating definition of declination
    talk_fig1_plot_angle(elecstr,ax1,fig,elections,seatsbool)

    # specific examples of declination
    # fig1_txpa_heat([ax2,ax3,ax4,ax5],fig,elections)

    plt.tight_layout(w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    output_fig(fig,fnstr)
    # with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
    #     fig.canvas.print_png(outfile)

    plt.close()

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

    # sns.set_style("ticks")
    # subplot illustrating definition of declination
    # fig1_plot_angle(elecstr,ax1,fig,elections)

    # specific examples of declination
    fig1_txpa_heat([ax2,ax3,ax4,ax5],fig,elections)

    plt.tight_layout(w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()

# Fig. 1
def fig_discuss(fnstr,elections):
    """
    """
    fig, axes = plt.subplots(1,3, figsize=(scalef*10,scalef*3),dpi=mydpi) # , sharex=True, sharey=True, facecolor = 'white', frameon=True)
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

    # sns.set_style("ticks")
    # subplot illustrating definition of declination
    # fig1_plot_angle(elecstr,ax1,fig,elections)

    plt.tight_layout() # w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()

# Fig. 1
def fig_threebad(fnstr,elections):
    """
    """
    fig, axes = plt.subplots(1,2, figsize=(scalef*7,scalef*3),dpi=mydpi) # , sharex=True, sharey=True, facecolor = 'white', frameon=True)
    axes = axes.ravel()

    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    axes[0].text(0.01,.92,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
    axes[1].text(0.50,.92,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')
    # axes[1].text(0.66,.92,'C',fontsize=16, transform=fig.transFigure, fontweight='bold')

    elec = elections['2012_NC_11']
    fig1_plot_one_declination(axes[0],elec.demfrac,'2012 NC',True,plotdec=True,xaxislab=True,yaxislab=True)

    elec = elections['2016_MN_11']
    fig1_plot_one_declination(axes[1],elec.demfrac,'2012 PA',True,plotdec=True,xaxislab=True,yaxislab=False)

    ############################
    # explore number of stolen seats
    # elec = elections['2012_OH_11']
    # fig1_plot_one_declination(axes[2],elec.demfrac,'2012 OH',True,\
    #                           plotdec=True,xaxislab=True,yaxislab=False,plotfullreg=False)

    # sns.set_style("ticks")
    # subplot illustrating definition of declination
    # fig1_plot_angle(elecstr,ax1,fig,elections)

    plt.tight_layout() # w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()

# Fig. 1
def fig_intro(fnstr,vals,ttls,pslopes=True,pdec=True,mylab='Dem. fraction'):
    """ Make some figures for the Introductory paper
    """
    fig, axes = plt.subplots(1,len(vals), figsize=(scalef*3.3*len(vals),scalef*3),dpi=mydpi,sharey=True)
    axes = axes.ravel()

    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    labls = ['A','B','C','D']
    for i in range(len(vals)):
        axes[i].text(0.02 + .95*i/len(vals),.92,labls[i],fontsize=16, \
                     transform=fig.transFigure, fontweight='bold')
        fig1_plot_one_declination(axes[i],vals[i],ttls[i],\
                                  lrg=True,plotslopes=pslopes,plotdec=pdec,xaxislab=True,\
                                  ylab=mylab,yaxislab=(i == 0))
        
    # plt.tight_layout() # w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    # fig.subplots_adjust(hspace=3)

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()

# Fig. 1
def fig_web(fnstr,vals,ttls,pslopes=True,pdec=True,mylab='Whig fraction'):
    """ Make some figures for the Introductory paper
    """
    fig, axes = plt.subplots(2,2, figsize=(12,12),dpi=mydpi)
    axes = axes.ravel()

    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    labls = ['A','B','C','D']
    for i in range(len(vals)):
        # axes[i].text(0.02 + .95*i/len(vals),.92,labls[i],fontsize=16, \
        #              transform=fig.transFigure, fontweight='bold')
        web_plot_one_declination(axes[i],vals[i],ttls[i],\
                                  lrg=True,plotslopes=pslopes,plotdec=pdec,xaxislab=True,\
                                  ylab=mylab,yaxislab=((i%2)==0))
        
    # plt.tight_layout() # w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()

# For talk - plot efficiency gap (modeled off fig_intro)
def fig_eg(fnstr,vals,ttls,pslopes=True,pdec=True,mylab='Whig fraction'):
    """ Make some figures for the Introductory paper
    """
    fig, axes = plt.subplots(1,len(vals), figsize=(scalef*3*len(vals),scalef*3),dpi=mydpi)
    axes = axes.ravel()

    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    labls = ['A','B','C','D']
    for i in range(len(vals)):
        axes[i].text(0.01 + 1.0*i/len(vals),.92,labls[i],fontsize=16, \
                     transform=fig.transFigure, fontweight='bold')
        fig1_plot_eg(axes[i],vals[i],ttls[i],\
                     lrg=True,plotslopes=pslopes,plotdec=pdec,xaxislab=True,\
                     ylab=mylab,yaxislab=(i == 0))
        
    plt.tight_layout() # w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()

# Fig. 1
def fig_intro_mm(fnstr,vals,ttls,pslopes=True,pdec=True,mylab='Whig fraction'):
    """ Make some figures for the Introductory paper
    """
    fig, axes = plt.subplots(1,len(vals), figsize=(scalef*3*len(vals),scalef*3),dpi=mydpi)
    axes = axes.ravel()

    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    labls = ['A','B','C','D']
    for i in range(len(vals)):
        axes[i].text(0.01 + 1.0*i/len(vals),.92,labls[i],fontsize=16, \
                     transform=fig.transFigure, fontweight='bold')
        fig1_plot_one_declination(axes[i],vals[i],ttls[i],\
                                  lrg=True,plotslopes=pslopes,plotdec=pdec,xaxislab=True,\
                                  ylab=mylab,yaxislab=(i == 0))
        axes[i].axhline(np.mean(filter(lambda x: x <= 0.5, vals[i])),linestyle='dashed',color='grey')
        axes[i].axhline(np.mean(filter(lambda x: x > 0.5, vals[i])),linestyle='dashed',color='grey')

    plt.tight_layout() # w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()

# Fig. 1
def fig1old_create(fnstr,elecstr,elections):
    """
    """
    fig = plt.figure(figsize=(scalef*8,scalef*8),dpi=mydpi)
    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    ax1 = plt.subplot2grid((7,2), (0,0), colspan=2, rowspan=3)
    ax2 = plt.subplot2grid((7,2), (3,0), rowspan=2)
    ax3 = plt.subplot2grid((7,2), (3,1), rowspan=2)
    ax4 = plt.subplot2grid((7,2), (5,0), rowspan=2)
    ax5 = plt.subplot2grid((7,2), (5,1), rowspan=2)

    # sns.set_style("ticks")
    # subplot illustrating definition of declination
    fig1_plot_angle(elecstr,ax1,fig,elections)

    # specific examples of declination
    fig1_txpa_heat([ax2,ax3,ax4,ax5],fig,elections)

    plt.tight_layout(w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()

##################################################################################################
# Fig. 2
def sub_make_heatmap(fn,curax,tmparr,allyrs,bins,xticks,yticks):
    """ actually make the image
    """

    arr = [[0 for i in range(len(bins)-1)] for j in range(len(allyrs))]

    for i in range(len(tmparr)):
        duh = sorted(tmparr[i])
        # print "%d %.2f %.2f -- %.2f " % (len(tmparr[i]),min(duh),max(duh),(max(duh)-min(duh))/(2*(duh[3*len(duh)/4]-duh[len(duh)/4])/pow(len(duh),0.33)))
        inds = np.digitize(tmparr[i],bins)
        # print bins
        if len(tmparr[i]) > 0:
            print "%d %.2f %.2f" % (i,min(tmparr[i]),max(tmparr[i]))
        # for j in range(len(inds)):
            # print tmparr[i][j],inds[j]
            # if i in [2,9] and tmparr[i][j] > 0.3:
            #    print "min: ",tmparr[i][j]
        for x in inds:
            arr[i][min(max(1,x),len(bins)-1)-1] += 1

    # hmm, so inds are indices of the right endpoints of intervals.
    # -0.5355 is min

    print
    # dtype = [('Col1','int32'), ('Col2','float32'), ('Col3','float32')]
    # values = numpy.zeros(20, dtype=dtype)
    index = []
    for i in range(len(allyrs)):
        if i%5 == 0:
            index.append(str(allyrs[i]))
        else:
            index.append(str(allyrs[i]))

    df = pd.DataFrame(np.array(arr), index=index)
    
    # labels = map(lambda x: "%.2f" % (x), bins)
    # df.columns = labels

# yticks = np.linspace(0,1,6)

# x_end = 6
# xticks = np.arange(x_end+1)

# -0.7,-0.5,-0.3,-0.1
    # ax = sns.heatmap(thick_df, linewidth=0, xticklabels=xticks, yticklabels=yticks[::-1], square=True, cmap=cmap)

# ax.set_xticks(xticks*ax.get_xlim()[1]/(2*math.pi))
# ax.set_yticks(yticks*ax.get_ylim()[1])

    # set appropriate font and dpi
    # plt.yticks(rotation=90)
    sns.set(font_scale=1.2)
    # sns.set_style({"savefig.dpi": 100})
    # plot it out
    N = df.max().max() - df.min().min() + 1
    cmap = cmap_discretize(plt.cm.Greens, N)
    # ax = sns.heatmap(df, ax=ax, cmap=cmap, annot=True)

    nax = sns.heatmap(df, ax=curax, xticklabels=xticks, yticklabels=yticks,cmap=cmap, linewidths=.1)
    # nax.set_axis_off()
    # xticks=['a','b','c']
    # curax.set_xticks([2,4,8])
    # nax.set_xticks([-0.6,0.6])
    # set the x-axis labels on the top
    # ax.xaxis.tick_top()
    # rotate the x-axis labels
    # plt.xticks(rotation=90)
    # nax.xticks(rotation=90)
    # nax.yticks(rotation=90)
    # get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)
    # curfig = ax.get_figure()
    # specify dimensions and save
    # curfig.set_size_inches(10, 6)
    # return ax
    # fig.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)    
    # plt.close()
    # return curfig

# Fig. 2
def fig2_make_heatmap(fnstr,elections,mmd):
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
    congtmparr = [[] for i in range(len(allyrs))]
    statetmparr = [[] for i in range(len(allyrs))]
    congNtmparr = [[] for i in range(len(allyrs))]
    stateNtmparr = [[] for i in range(len(allyrs))]
    minyr = 1972
    ccnt = 0
    scnt = 0
    for elec in elections.values():
        if elec.Ndists >= 8 and elec.chamber == '11' and int(elec.yr) >= minyr and int(elec.yr)%2 == 0:
            totcong += 1
            fang = get_declination(elec.state,elec.demfrac)*math.log(elec.Ndists)/2
            if abs(fang) < 2:
                ccnt += 1
                congtmparr[allyrs.index(int(elec.yr))].append(fang)
                congNtmparr[allyrs.index(int(elec.yr))].append(fang*elec.Ndists*1.0/2)
                congang.append(fang)
                congangN.append(fang*elec.Ndists*1.0/2)
        if elec.Ndists >= 8 and elec.chamber == '9' and int(elec.yr) >= minyr and \
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
    fig, axes = plt.subplots(1,2, figsize=(scalef*8,scalef*6),dpi=mydpi) # , sharex=True, sharey=True, facecolor = 'white', frameon=True)
    axes = axes.ravel()

    print "cong: ",ccnt
    print "state: ",scnt

    numbins = 10
    # bins = np.linspace(-0.75,0.75,numbins+1)
    bins = np.linspace(-1.2,0.8,numbins+1)
    print bins
    # sub_make_heatmap('heat-cong',axes[0],congtmparr,allyrs,bins,[-0.6,-0.4,0,0.3,0.6])
    sub_make_heatmap('heat-cong',axes[0],congtmparr,allyrs,bins,[-0.6,-0.4,0,0.3,0.6])
    axes[0].set_title('Congressional elections')
    axes[0].set_xticks([1,3,5,7,9]) # 0,2,4,6,8,10]) # -0.6,-0.3,0.0,0.3,0,6])
    axes[0].axvline(5,color='black')

    numbins = 12
    # bins = np.linspace(-0.7,0.5,numbins+1)
    bins = np.linspace(-1.5,0.8,numbins+1)
    sub_make_heatmap('heat-state',axes[1],statetmparr,allyrs,bins,[-0.6,-0.4,-0.2,0,0.2,0.4])
    axes[1].set_xticks([1,3,5,7,9,11])
    axes[1].set_title('State elections')
    axes[1].axvline(7,color='black')

    axes[0].text(0.01,.97,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
    axes[0].text(0.5,.97,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')

    # -0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,.1,.2,.3,.4
    #     *    *    *    *    *    6    7 8  0  10 11 
    plt.tight_layout()
    output_fig(fig,fnstr)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    # plt.show()
    plt.close()   

    # numbins = 12
    # bins = np.linspace(-8,8,numbins+1)
    # sub_make_heatmap('heat-congN',congNtmparr,allyrs,bins)
    # numbins = 12
    # bins = np.linspace(-42,42,numbins+1)
    # sub_make_heatmap('heat-stateN',stateNtmparr,allyrs,bins)

# Fig. 2
def fig2_make_heatmap_e(fnstr,elections,mmd):
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
    congtmparr = [[] for i in range(len(allyrs))]
    statetmparr = [[] for i in range(len(allyrs))]
    congNtmparr = [[] for i in range(len(allyrs))]
    stateNtmparr = [[] for i in range(len(allyrs))]
    minyr = 1972
    ccnt = 0
    scnt = 0
    for elec in elections.values():
        if elec.Ndists >= 8 and elec.chamber == '11' and int(elec.yr) >= minyr and int(elec.yr)%2 == 0:
            totcong += 1
            fang = get_declination(elec.state,elec.demfrac)*math.log(elec.Ndists)/2
            if abs(fang) < 2:
                ccnt += 1
                congtmparr[allyrs.index(int(elec.yr))].append(fang)
                congNtmparr[allyrs.index(int(elec.yr))].append(fang*elec.Ndists*1.0/2)
                congang.append(fang)
                congangN.append(fang*elec.Ndists*1.0/2)
        if elec.Ndists >= 8 and elec.chamber == '9' and int(elec.yr) >= minyr and \
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
    fig, axes = plt.subplots(1,2, figsize=(scalef*8,scalef*6),dpi=mydpi) # , sharex=True, sharey=True, facecolor = 'white', frameon=True)
    axes = axes.ravel()

    print "cong: ",ccnt
    print "state: ",scnt

    numbins = 10
    # bins = np.linspace(-0.75,0.75,numbins+1)
    bins = np.linspace(-1.2,0.8,numbins+1)
    print bins
    # sub_make_heatmap('heat-cong',axes[0],congtmparr,allyrs,bins,[-0.6,-0.4,0,0.3,0.6])
    yticks = [1972,1982,1992,2002,2012]
    sub_make_heatmap('heat-cong',axes[0],congtmparr,allyrs,bins,[-1.2,-0.8,-0.4,0,0.4,0.8],yticks)
    axes[0].set_title('Congressional elections')
    axes[0].set_xticks([0,2,4,6,8,10]) # 0,2,4,6,8,10]) # -0.6,-0.3,0.0,0.3,0,6])
    axes[0].set_yticks([2.5,7.5,12.5,17.5,22.5])
    axes[0].axvline(6,color='black')

    numbins = 10
    # bins = np.linspace(-0.7,0.5,numbins+1)
    bins = np.linspace(-1.5,1,numbins+1)
    yticks = [1972,1982,1992,2002]
    sub_make_heatmap('heat-state',axes[1],statetmparr,allyrs,bins,[-1.5,-1,-0.5,0,0.5,1],yticks)
    axes[1].set_xticks([0,2,4,6,8,10])
    axes[1].set_yticks([7.5,12.5,17.5,22.5])
    axes[1].set_title('State elections')
    axes[1].axvline(6,color='black')

    axes[0].text(0.01,.97,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
    axes[0].text(0.5,.97,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')

    # -0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,.1,.2,.3,.4
    #     *    *    *    *    *    6    7 8  0  10 11 
    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    # plt.show()
    plt.close()   

    # numbins = 12
    # bins = np.linspace(-8,8,numbins+1)
    # sub_make_heatmap('heat-congN',congNtmparr,allyrs,bins)
    # numbins = 12
    # bins = np.linspace(-42,42,numbins+1)
    # sub_make_heatmap('heat-stateN',stateNtmparr,allyrs,bins)

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
    print "Total elections: %d" % (len(xarr))
    ax.scatter(xarr,yarr,color='darkgreen')

    # ax.set_title('Not sure of title...')
    ax.set_xlabel('Number of districts in election')
    ax.set_ylabel('$|\\tilde{\\delta}|$')

    # now make regression
    l = stats.linregress(xarr,yarr)
    print "INFO: delta_e regression line: %.4f %.4f %.4f %.4f" % (l[0],l[1],l[2],l[3])
    ax.plot([0,250],[l[1],l[1]+250*l[0]],'k-',linewidth=1)
    
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
    # ax.set_title('2012-2016 congressional elections')
    ax.set_xlabel('$\\tilde{\\delta}$')
    if prtitle:
        if int(yrmin) < 2010:
            ax.set_title(str(yrmin) + '-' + str(int(yrmin)+8),fontsize=16)
        else:
            ax.set_title(str(yrmin) + '-' + str(int(yrmin)+4),fontsize=16)

    # plt.gca().set_axis_bgcolor('none')
    sns.set(font_scale=0.8)
    vdiff = 0.1
    if chamber == '9':
        ax.axis([-1,1,0,len(ans)*vdiff+0.15])
    else:
        ax.axis([-1,1,0,len(ans)*vdiff+0.15])
    for j in range(1,len(ans)+1):
        i = j-1
        # print i,ans[i][0],ans[i]
        ax.plot([ans[i][2],ans[i][3]],[i*vdiff+0.05,i*vdiff+0.05],color='darkgreen')
        ax.annotate(ans[i][0],(ans[i][3]+0.01,i*vdiff))

    ax.set_yticks([])
    ax.set_axis_on()
    # ax.gca().get_yaxis().set_visible(False)
    # ax.xlabel('Declination')

def fig3_deltae(fnstr,elections,cycstates,mmd):
    """
    """
    plt.close()
    fig, axes = plt.subplots(1,2, figsize=(scalef*8,scalef*5),dpi=mydpi) 

    # show that delta_e properly adjusts
    # sns.set_style("ticks",ax=axes[0])
    # sns.despine()
    fig3_scatter(elections,axes[0],mmd)

    # now plot intervals just for 2012--2016
    # sns.set_style("ticks",ax=axes[1])
    # sns.despine()
    fig3_deltae_linechart(elections,cycstates,mmd,2012,axes[1],'11',False)

    axes[0].text(0.01,.95,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
    axes[1].text(0.52,.95,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')

    # sns.set_style("white")
    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()   

def fig_deltae_only(fnstr,elections,cycstates,mmd):
    """
    """
    plt.close()
    plt.figure(figsize=(scalef*5,scalef*5),dpi=mydpi) 

    # now plot intervals just for 2012--2016
    # sns.set_style("ticks",ax=axes[1])
    # sns.despine()
    fig3_deltae_linechart(elections,cycstates,mmd,2012,plt.gca(),'11',False)

    # sns.set_style("white")
    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()   

def figS34_linechart(fnstr,elections,cycstates,mmd,chamber):
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

    # show that delta_e properly adjusts
    # sns.set_style("ticks",ax=axes[0])
    # sns.despine()
    fig3_deltae_linechart(elections,cycstates,mmd,1972,axes[0],chamber,False)
    fig3_deltae_linechart(elections,cycstates,mmd,1982,axes[1],chamber,False)
    fig3_deltae_linechart(elections,cycstates,mmd,1992,axes[2],chamber,False)
    fig3_deltae_linechart(elections,cycstates,mmd,2002,axes[3],chamber,False)
    if chamber == '11':
        fig3_deltae_linechart(elections,cycstates,mmd,2012,axes[4],chamber,False)
        axes[5].set_axis_off()

    axes[0].set_axis_on()
    axes[1].set_axis_on()
    axes[2].set_axis_on()
    # sns.set_style("white")
    plt.tight_layout()

    output_fig(fig,fnstr)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()   

###################################################################################################
# deltae threshold
def data_deltae_threshold(elections):
    """
    """
    arr = []
    goodarr = []
    badarr = []
    zzz = []
    for elec in elections.values():
        if 1972 <= int(elec.yr) < 2012:
            chm = elec.chamber
            state = elec.state
            if int(elec.yr) < 1982:
                yrmin = 1972
            elif int(elec.yr) < 1992:
                yrmin = 1982
            elif int(elec.yr) < 2002:
                yrmin = 1992
            elif int(elec.yr) < 2012:
                yrmin = 2002
            else:
                yrmin = 2012
            curfa = math.log(elec.Ndists)*get_declination('',elec.demfrac)/2
            othfa = []
            for j in range(5):
                newyr = yrmin + 2*j
                newid = '_'.join([str(newyr),state,chm])
                if newid in elections.keys():
                    newelec = elections[newid]
                    newfa = get_declination('',newelec.demfrac)
                    if abs(newfa) < 2:
                        othfa.append(math.log(newelec.Ndists)*newfa/2)
            if len(othfa) == 5:
                notokay = False
                if max(othfa)*min(othfa) < 0:
                    # for j in range(len(othfa)):
                    # if othfa[j]*curfa < 0:
                        notokay = True
                demdist = np.mean(elec.demfrac)
                if notokay:
                    print "Not %s %s %s %.2f % .2f %.2f" % \
                        (elec.yr,elec.state,elec.chamber,max(othfa)-min(othfa),curfa,demdist)
                    arr.append([abs(curfa),False])
                    zzz.append((0 + np.random.randn(20)[0]*0.05))
                    badarr.append([abs(curfa),max(othfa)-min(othfa)])
                else:
                    print "Goo %s %s %s %.2f % .2f %.2f" % \
                        (elec.yr,elec.state,elec.chamber,max(othfa)-min(othfa),curfa,demdist)
                    arr.append([abs(curfa),True])
                    zzz.append((1 - np.random.randn(20)[0]*0.05))
                    goodarr.append([abs(curfa),max(othfa)-min(othfa)])
    for y in arr:
        print "Hello: ",y[0]
    plt.figure(figsize=(scalef*8,scalef*8),dpi=mydpi)
    plt.scatter([x[0] for x in goodarr],[x[1] for x in goodarr],color='green')
    plt.scatter([x[0] for x in badarr],[x[1] for x in badarr],color='black')

    xxx = np.linspace(0.0,1.2,121)
    yyy = []
    for jj in xxx:
        ggg = filter(lambda x: x[0] > jj, goodarr)
        hhh = filter(lambda x: x[0] > jj, badarr)
        yyy.append((len(ggg))*1.0/(len(ggg)+len(hhh)))
        print "test: ",jj,yyy[-1]
    plt.plot(xxx,yyy,'ro')
    plt.savefig('/home/gswarrin/research/gerrymander/pics/thresh')
    plt.close()

    plt.figure(figsize=(scalef*8,scalef*8),dpi=mydpi)
    plt.scatter([x[0] for x in arr],zzz)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/scatt')
    plt.close()

    tot = len(goodarr + badarr)
    print "Total elecs: ",tot
    for j in np.linspace(0.4,0.9,51):
        tmparr = filter(lambda x: x[0] >= j, arr)
        t2arr = filter(lambda x: x[1], tmparr)
        if len(tmparr) > 0:
            print "bbb: %.2f %.3f " % (j,len(t2arr)*1.0/len(tmparr))
        else:
            print "bbb: %.2f " % (j)
        # if len(tmparr) >= 0.95*tot:
        #     print "Yes: %.2f" % (j)
        # else:
        #     print "No!: %.2f" % (j)

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
    # plt.gca().set_axis_bgcolor('none')
    axes = axes.ravel()

    for i,st in enumerate(nstates):
        xlab = (i > (r-1)*c)
        ylab = (i%c==0)
        fig1_plot_one_declination(axes[i],st[1].demfrac,st[1].state,False,True,xlab,ylab,False)
    for i in range(len(nstates),r*c):
        axes[i].set_axis_off()

    # figS12_plot_declination_grid(fnstr,r,c,nstates)
    # plot_declination_grid(fnstr + '-pg1',r,c,yr,chamber,nstates[:24])
    # plot_declination_grid(fnstr + '-pg2',r,c,yr,chamber,nstates[24:])
    plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()

####################################################################################################
# Fig S5
def figS5_wi_scatter(elections,mmd):
    lang = [[] for i in range(5)]
    lzgap = [[] for i in range(5)]
    cnt = 0
    for elec in elections.values():
        # elec.myprint()
        # print "yr: ",elec.yr
           # (elec.yr not in Mmmd.keys() or elec.state not in Mmmd[elec.yr]) and \
        if 2010 >= int(elec.yr) >= 1972 and \
           (elec.yr not in mmd.keys() or elec.state not in mmd['1972']) and \
            elec.Ndists >= 8 and elec.chamber=='9':
            ang = get_declination(elec.state,elec.demfrac)
            zgap = get_tau_gap(elec.demfrac,0)
            cnt += 1

            yridx = int((int(elec.yr)-1972)/10)
            if ang != 0:
                lang[yridx].append(ang)
                lzgap[yridx].append(zgap)
            # print "blah"
            print "% .3f % .3f %3d %s %s %2d" % (ang,zgap,elec.Ndists,elec.yr,elec.state,int(elec.chamber))
    print "total: ",cnt
    plt.figure(figsize=(scalef*8,scalef*8),dpi=mydpi)
    plt.gca().set_axis_bgcolor('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

    plt.axis([-0.6,0.6,-0.6,0.6])
    # plt.axis([-40,40,-40,40])
    plt.axvline(0,color='black',ls='dotted')
    plt.axhline(0,color='black',ls='dotted')
    # [255 0 0;                 251 141 26;                 255 255 0;                   0 255 128;
    #               0 255 255;                   0 135 255;                   0 0 255;
    #             101 43 143;                 255 0 255;                 255 10 140];
    # cols = [plt.cm.Viridis(0),plt.cm.Viridis(.3),plt.cm.Viridis(.6),plt.cm.Viridis(.9)]
    cols = ['#ff0000','#009000','#00a0a0','#900090']
    # tuple(.106,.619,.467),tuple(.126,.613,.454),tuple(.147,.606,.441),tuple(.167,.599,.429)]
    # cols = ['green','blue','red','orange']
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
    plt.ylabel('Twice the efficiency gap',fontsize=18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    datax = [0.196,0.165,0.140]
    datay = [0.204,0.157,0.136]
    #  datax = [0.055,0.343,0.244,0.328]
    # datay = [0.150,0.247,0.251,0.280]
    plt.scatter(datax,datay,color='black',marker='o',s=100)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'scatter-wi-nommd')
    plt.close()
    # make_scatter('wi-ex-scatter',lang,lagap)
    
def fig_pack_crack_all(elections,axl,axr):
    """ illustrate what happens to declination after packing and cracking fraction of seats
    """
    # fig = plt.figure(figsize=(8,8))

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    N = 19
    resp = 2
    demfrac = 0.55
    # fig = plt.figure(figsize=(8,8))

    delv = 1.0/(resp*N)
    adj = demfrac-0.5/resp
    vals = [min(max(0,adj+delv*i),1) for i in range(N)]
    if min(vals) < 0 or max(vals) > 1:
        print "Went out of range"
        return

    axl.set_xlabel('Fraction of districts packed/cracked')
    axl.set_ylabel('Dem. vote')
    axr.set_ylabel('Change in $\\delta$')

    # fig_pack_crack_one(ax,vals,'blue')
    shapes = ['x','o','^','+']

    lab3 = fig_pack_crack_one(axl,axr,sorted(elections['2012_CA_11'].demfrac),'orange',shapes[2],'CA',True)
    lab1 = fig_pack_crack_one(axl,axr,sorted(elections['2012_GA_11'].demfrac),'green',shapes[0],'GA',False)

    lab2 = fig_pack_crack_one(axl,axr,sorted(elections['2012_AZ_11'].demfrac),'blue',shapes[1],'AZ',False)

    # plt.legend([lab1,lab2,lab3],('1982-1990','1992-2000','2002-2010'),loc='upper left')
    axl.legend(loc='upper left')

    # plt.savefig('/home/gswarrin/research/gerrymander/pics/fig-pack')
    # plt.close()

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
    # axl.plot(xvals,vals,'o-',color=col)
    tmp = [0]
    tmpx = [0]

    if do_pack:
        can_pack = True
        while can_pack:
            can_pack,narr = pack_or_crack(vals,False)
            vals = narr
            idx += 1
            
            ans.append([get_declination('',vals)-origfa,vals])
            # print np.mean(vals) # ,vals
            # ans.append([get_tau_gap(vals,1),vals])
            # plt.plot(xvals,vals,linestyle='dashed')
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
                # ans.append([get_tau_gap(vals,1),vals])
                # plt.plot(xvals,vals,linestyle='dashed')
                tmp.append(ans[-1][0])
                tmpx.append(len(tmpx)*1.0/len(vals))
            else:
                can_crack = False
                # axr.plot(xvals[:len(tmp)-1],tmp[:-1],'o-',color=col)
    blah = axl.plot(tmpx[:-1],tmp[:-1],marker=sha,linestyle='-',color=col,label=mylab)
    return blah

def fig_variations(fnstr,elections,cycstates,mmd):
    """
    """
    plt.close()
    # fig, axes = plt.subplots(1,2, figsize=(scalef*8,scalef*4),dpi=mydpi) 
    fig = plt.figure(figsize=(scalef*8,scalef*4),dpi=mydpi) 
    axl = plt.subplot2grid((1,2), (0,0))
    # axl = plt.subplot2grid((1,2), (0,0), sharex=axr, sharey=axr)
    ax1 = plt.subplot2grid((1,2), (0,1))
    # axr = fig.add_axes(axl.get_position(), frameon=False, sharex=axl, sharey=axl)

    # print axl
    # print axr
    # print ax1

    # show that delta_e properly adjusts
    # sns.set_style("ticks",ax=axes[0])
    # sns.despine()
    fig3_scatter(elections,ax1,mmd)

    # now plot intervals just for 2012--2016
    # sns.set_style("ticks",ax=axes[1])
    # sns.despine()
    # fig3_deltae_linechart(elections,cycstates,mmd,2012,axes[1],'11',False)

    # axr = fig.add_subplot(111,sharex=axes[0])
    # axr.yaxis.tick_right()
    # axr.yaxis.set_label_position("right")
    fig_pack_crack_all(elections,axl,axl)
    # axr.set_axis_bgcolor('none')
    axl.set_axis_bgcolor('none')

    axl.spines['top'].set_visible(False)
    axl.spines['right'].set_visible(False)
    axl.xaxis.set_ticks_position('bottom')
    axl.yaxis.set_ticks_position('left')

    axl.text(0.01,.95,'A',fontsize=16, transform=fig.transFigure, fontweight='bold')
    ax1.text(0.52,.95,'B',fontsize=16, transform=fig.transFigure, fontweight='bold')

    plt.tight_layout()

    # sns.set_style("white")
    # plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()   

#######################################################################################
#######################################################################################
def grid_one_state(fnstr,r,c,chamber,state,elections,mymmd):
    """ Make grid of angles for paper
    """
    nstates = []
    for yr in [str(1972+2*j) for j in range(23)]:
        tmpkey = '_'.join([yr,state,chamber])
        if tmpkey in elections.keys():
            # print tmpkey
            elec = elections[tmpkey]
            # ndem = len(filter(lambda y: y >= 0.5, elec.demfrac))
            # if elec.Ndists > 1 and (yr not in mymmd or x not in mymmd[yr]) and \
            #    0 < ndem < elec.Ndists:
            nstates.append([get_declination(state,elec.demfrac),elec])
            # print nstates[-1]
    # nstates.sort(key=lambda x: x[0])
    # nstates.reverse()
    # print 'blah',len(nstates)
    # for x in nstates:
    #     print x
    
    # if 1 == 1:
    #     return
    fig, axes = plt.subplots(r,c, figsize=(10*c,10*r), sharex = True, sharey = True) # dpi=mydpi, sharex=True, sharey=True)
    # plt.gca().set_axis_bgcolor('none')
    axes = axes.ravel()

    for i,st in enumerate(nstates):
        xlab = (i > (r-1)*c)
        ylab = (i%c==0)
        fig1_plot_one_declination(axes[i],st[1].demfrac,st[1].state + st[1].yr,plotslopes=True,\
                                  lrg=True,xaxislab=xlab,yaxislab=ylab,plotfullreg=False)
    for i in range(len(nstates),r*c):
        axes[i].set_axis_off()

    # figS12_plot_declination_grid(fnstr,r,c,nstates)
    # plot_declination_grid(fnstr + '-pg1',r,c,yr,chamber,nstates[:24])
    # plot_declination_grid(fnstr + '-pg2',r,c,yr,chamber,nstates[24:])
    # plt.tight_layout()
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()

#######################################################################################
#######################################################################################
def grid_from_list(fnstr,r,c,elist,elections,mymmd):
    """ Make grid of angles for paper
    """
    nstates = []
    for x in elist:
        elec = elections[x]
        nstates.append([get_declination(elec.state,elec.demfrac),elec])

    fig, axes = plt.subplots(r,c, figsize=(10*c,10*r), sharex = True, sharey = True)
    axes = axes.ravel()

    for i,st in enumerate(nstates):
        xlab = (i > (r-1)*c)
        ylab = (i%c==0)
        fig1_plot_one_declination(axes[i],st[1].demfrac,st[1].state + st[1].yr,plotslopes=True,\
                                  lrg=True,xaxislab=xlab,yaxislab=ylab,plotfullreg=False)
    for i in range(len(nstates),r*c):
        axes[i].set_axis_off()

    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)
    plt.close()
    
######################################################################################
# pictures for talk
######################################################################################

# Fig. 1
def talk_intro(fnstr,vals,ttls,pslopes=True,pdec=True,mylab='Whig fraction'):
    """ Make some figures for the Introductory paper
    """
    fig, axes = plt.subplots(1,len(vals), figsize=(scalef*5*len(vals),scalef*3),dpi=mydpi,sharey=True)
    axes = axes.ravel()

    fig.patch.set_visible(True)
    fig.patch.set_facecolor('white')

    labls = ['A','B','C','D']
    for i in range(len(vals)):
        # axes[i].text(0.02 + .95*i/len(vals),.92,labls[i],fontsize=16, \
        #              transform=fig.transFigure, fontweight='bold')
        talk_plot_one_declination(axes[i],vals[i],ttls[i],\
                                  lrg=True,plotslopes=pslopes,plotdec=pdec,xaxislab=True,\
                                  ylab=mylab,yaxislab=(i == 0))
        
    # plt.tight_layout() # w_pad=1,h_pad=1)
    # plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fnstr)

    with open('/home/gswarrin/research/gerrymander/pics/' + fnstr + '.png', 'w') as outfile:
        fig.canvas.print_png(outfile)

    plt.close()
