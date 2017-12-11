##########################################################################3
# stuff having to do with simultaneous alpha-curves

def plot_alpha_curves(vals,xmax=3,fn='alphaseries'):
    """
    """
    xmin = -xmax
    x = np.linspace(xmin,xmax,300)
    plt.figure(figsize=(8,8))
    plt.axis([xmin,xmax,-0.5,0.5])

    y = [get_tau_gap(vals,x[i]) for i in range(len(x))]
    # print y
    plt.plot(x,y)
    plt.grid(True)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn + '.png')
    plt.show()
    plt.close()
    
def plot_many_alpha_curves(tmpstr,elecs,xmax=3):
    """ run through various elections
    """ 
    xmin = 0
    xmax = 2
    x = np.linspace(xmin,xmax,200)
    plt.figure(figsize=(6,24))
    plt.axis([0,2,-1,1])

    for elec in elecs:
        if elec.Ndists >= 6:
            y = [get_tau_gap(elec.demfrac,x[i]) for i in range(len(x))]
            # y = [get_declination(elec.state,elec.demfrac) for i in range(len(x))]
            plt.plot(x,y)
            # get get_declination value and plot a dot there
            fa = get_declination(elec.state,elec.demfrac)
            z = [abs(y[i]-fa) for i in range(len(x))]
            j = z.index(min(z))
            if fa >= -1:
                print "%s %.3f" % (elec.state,fa-get_tau_gap(elec.demfrac,0))
                plt.plot(x[j],fa,'ro')

    plt.grid(True)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'curves-' + tmpstr + '.png')
    plt.show()
    plt.close()
                
# plot_alpha_curves([elections['2010_MI_11'],elections['2012_NC_11']])   
# plot_alpha_curves([0.4,0.45,0.48,.75,0.83]) 
# plot_alpha_curves([0.45,0.45,0.45,.61,.65,.7],10) 
# plot_alpha_curves([0.49,0.49,0.49,0.95,0.7],30)

# samples = filter(lambda x: 1 >= x >= 0, np.random.normal(.55,.15,200))
# s2 = []
# for x in samples:
#     if x >= 0.5:
#         s2.append(x)
#     else:
#         s2.append(x-0.1)
# make_scatter('samp.png',[i for i in range(len(samples))],sorted([x for x in samples]))
# make_scatter('s2.png',[i for i in range(len(s2))],sorted([x for x in s2]))
# plot_alpha_curves([0.2,0.3,.52,.53,.55,.65,.7,.75,.85,.95],30)
# plot_alpha_curves(samples,30,'samplespic')
# plot_alpha_curves(s2,30,'s2pic')
# yrmin = 1992 
# chm = '11'
# telecs = filter(lambda x: (yrmin+8) >= int(x.yr) >= yrmin and \
#     x.chamber == chm, electionsa.values())
# for st in states:
#     nelecs = filter(lambda x: x.state == st,telecs)
#     # print len(nelecs)
#     plot_many_alpha_curves(chm+'-'+str(yrmin)+'-'+st,nelecs,3)    
# print sorted(samples)
