def find_lines(st,vals):
    """ find slope and intercepts of < 50% and >= 50% points
    """
    bel = sorted(filter(lambda x: x <  0.5, vals))
    abo = sorted(filter(lambda x: x >= 0.5, vals))
    if len(bel) < 2 or len(abo) < 2:
        return None,len(bel),len(abo),None,None
    xbs = range(1,len(bel)+1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(xbs,bel)
    # print slope, intercept
    xas = [len(bel)+i+1 for i in range(len(abo))]
    s2, i2, r2, p2, std_err = stats.linregress(xas,abo)
    # total slope
    s3, i3, r3, p3, std3 = stats.linregress(range(1,len(vals)+1),sorted(vals))
### uses last point instead of regression line
# horrible idea - if one race jumps over line very misleading
#    if st == 'NC':
#        print bel
#        print abo
#        print bel[-1],abo[0]
#    return [(slope-s2)/s3,1-bel[-1]-abo[0]],slope,intercept,s2,i2
###    scaled slope difference; nonneg diff of distance to 0.5
    return [(slope-s2)/s3,(max(0,0.5-(intercept+len(bel)*slope))-\
    max(0,(i2+(len(bel)+1)*s2)-0.5))/s3],slope,intercept,s2,i2

#    return [(slope-s2)/s3,1 - (i3+(len(bel)+1)*s3 - 0.5) -(intercept+len(bel)*slope)-\
#    (i2+(len(bel)+1)*s2)],slope,intercept,s2,i2
    # return [slope-s2,1-(intercept+len(bel)*slope)-\
    # (i2+(len(bel)+1)*s2)],slope,intercept,s2,i2

def plot_many_lines(tmpstr,elecs,ptext=False):
    """ run through various elections
    """ 
    xmin = -3
    xmax = 1
    plt.figure(figsize=(8,8))
    # plt.axis([xmin,xmax,-0.35,0.2])
    plt.axis([xmin,xmax,-10,10])
    
    ptpts = []
    sz =[]
    d = dict()
    skipped = 0
    styr = []
    for elec in elecs:
        print elec.yr,elec.state,elec.chamber
        vals = []
        for i in range(elec.Ndists):
            if elec.dcands[i] != None:
                vals.append(elec.demfrac[i])
        ans,cc,cd,ce,cf = find_lines(elec.state,vals)
        if 1==0:
            print elec.yr,elec.state,elec.Ndists
            continue
        if ans == None:
            # print "Skipping %s %s %s %d %d" % (elec.yr,elec.state,elec.chamber,cc,cd)
            skipped += 1
            continue
        ptpts.append(ans)
        # if abs(ans[0]) > 0.2 or abs(ans[1]) > 0.2:
        #     print "blah: ",elec.yr,elec.state,ans
        if elec.chamber == '11':
            sz.append(elec.Ndists*5)
        else:
            sz.append(elec.Ndists)
        if elec.state in ['WI','MI','OH','NC','PA','VA','FL','CA','MO']:
            styr.append(elec.state + elec.yr[-2:])
        else:
            styr.append('')
        d['_'.join([elec.yr,elec.state,elec.chamber])] = [ans[0],ans[1]]
    plt.scatter([x[0] for x in ptpts],[x[1] for x in ptpts]) # ,s=sz)
    for i,txt in enumerate(styr):
        if txt != '':
            plt.annotate(txt,(ptpts[i][0],ptpts[i][1]))
    plt.axhline(color='black')
    plt.axvline(color='black')
    if ptext:
        plt.text(-0.092,-0.07,'PA 12')
        plt.text(-0.105,-0.02,'PA 14')
        plt.arrow(-0.08,-0.02,0.005,-0.001,color='black',head_width=0.005,head_length=0.005)
    
        plt.text(-0.105,-0.045,'NC 12')
        plt.text(0.005,-0.17,'NC 14')
    
        plt.text(-0.105,-0.085,'VA 12')
        plt.text(-0.10,0.005,'VA 14')
    
        plt.text(0.015,0.005,'LA 12')
        plt.text(0.02,0.135,'AZ 14')
    
        plt.text(0.005,0.05,'CA 12')
        plt.text(0.01,0.075,'CA 14')
    
        plt.text(-0.06,0.065,'FL 12')
        plt.text(-0.025,0.115,'FL 14')
    
        plt.text(-0.20,-0.02,'MO 12')
        plt.text(-0.22,0.125,'MO 14')
    
        plt.text(-0.046,-0.185,'OH 12')
        plt.text(-0.035,-0.05,'OH 14')
        plt.arrow(-0.033,-0.04,\
        -0.002,0.005,color='black',head_width=0.005,head_length=0.005)
        
        plt.text(-0.06,-0.14,'MI 12')
        plt.text(-0.031,-0.07,'MI 14')
        plt.arrow(-0.031,-0.07,-0.003,\
        -0.006,color='black',head_width=0.005,head_length=0.005)
        
    plt.xlabel('slope difference $m-m$')
    plt.ylabel('intercept difference')
    arr = []
    print "Total number of elections: ",len(d.keys())," skipped: ",skipped
    for k in d.keys():
        arr.append([d[k],k])
    arr = sorted(arr)
    for x in arr:
        print x # ,elecs[x[1]].Ndists
    plt.grid(True)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'slope-' + tmpstr + '.png')
    plt.show()
    plt.close()   

def make_regression_grid(tmpstr,elections,keys):
    """ run through various elections
    """ 
    xmin = -0.2
    xmax = 0.2
    # plt.figure(figsize=(8,8))
    # plt.axis([xmin,xmax,-0.5,0.5])

    fig, axes = plt.subplots(2, 3, figsize=(8, 8), sharey=True)
    axes = axes.ravel()

    for i,k in enumerate(keys):
        elec = elections[k]
        vals = []
        for j in range(elec.Ndists):
            # if elec.dcands[j] != None:
            vals.append(elec.demfrac[j])
        vals = sorted(vals)
        N = len(vals)
        # print k,vals
        m = len(filter(lambda x: x < 0.5, vals))
        n = N-m
        ans,m1,i1,m2,i2 = find_lines(elec.state,vals)
        if ans == None:
            print "help",i
            continue
            
        x1 = range(1,m+1)
        x2 = range(m+1,N+1)
        y1 = vals[:m]
        y2 = vals[m:]
        # ptpts.append(ans)
        # d['_'.join([elec.yr,elec.state,elec.chamber])] = [ans[0]*ans[0] + \
        #        ans[1]*ans[1],ans[0],ans[1]]
        axes[i].scatter(x1,y1,color = 'red')
        axes[i].scatter(x2,y2,color = 'blue')
        # plt.scatter([x[0] for x in ptpts],[x[1] for x in ptpts])

        # y = srrs_mn.log_radon[srrs_mn.county==c]
        # x = srrs_mn.floor[srrs_mn.county==c]
        # axes[i].scatter(x + np.random.randn(len(x))*0.01, y, alpha=0.4)
        
        # No pooling model
        # b = unpooled_estimates[c]
        
        # Plot both models and data
        xvals1 = np.linspace(1,m,m+1)
        xvals2 = np.linspace(m+1,N,n+1)
        xvals3 = np.linspace(1,N,10)
        axes[i].plot(xvals1, m1*xvals1+i1, 'r')
        axes[i].plot(xvals2, m2*xvals2+i2, 'b')
        axes[i].plot(xvals3, 0.5 + 0*xvals3, color = 'black')
        axes[i].set_xticks([1,N])
        # axes[i].set_xticklabels(['basement', 'floor'])
        axes[i].set_ylim(0,1)
        yr,st,chm = k.split('_')
        axes[i].set_title(st + ' ' + yr)
        if not i%2:
            axes[i].set_ylabel('democratic vote fraction')
    
    axes[1].text(1,0.05,'z = (0.00,-0.17)')
    axes[0].text(1,0.05,'z = (-0.07,-0.07)')
    axes[2].text(1,0.05,'z = (0.03,0.03)')
    axes[3].text(1,0.05,'z = (0.09,-0.12)')

    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'slope-' + tmpstr + '.png')
    # plt.show()
    plt.close()   

    #ptpts = []
    # d = dict()
    # for elec in elecs:
    #     vals = []
    
    # arr = []
    # for k in d.keys():
    #     arr.append([d[k],k])
    # arr = sorted(arr)
    # for x in arr:
    #     print x
    # plt.grid(True)

def plot_zwalk(tmpstr,yrmin,yrmax,allelecs,ptext=False):
    """ run through various elections
    """ 
    xmin = -3
    xmax = 1
    plt.figure(figsize=(8,8))
    plt.axis([xmin,xmax,-0.35,0.21])

    parr = []
    for yrint in range(yrmin,yrmax+1,2):
        ptpts = []
        tot = 0
        # Hmm, don't weight - measure prevalence of gm rather than effect
        for elec in allelecs.values():
            if int(elec.yr) == yrint and elec.chamber == '11': # and elec.state == 'FL':
                vals = []
                curtot = 0
                for i in range(elec.Ndists):
                    if elec.dcands[i] != None:
                        # curtot += 1
                        curtot = 1
                        vals.append(elec.dcands[i].frac)
                ans,cc,cd,ce,cf = find_lines(vals)
                if ans != None:
                    # print "Skipping %s %s %s %d %d" % \
                    # (elec.yr,elec.state,elec.chamber,cc,cd)
                    # skipped += 1
                    # continue
                    # print "appending: ",ans
                    ptpts.append([ans[0]*curtot,ans[1]*curtot])
                    # tot += curtot
                    tot += 1
        print yrint,ptpts
        parr.append([sum([x[0] for x in ptpts])*1.0/tot,\
            sum([x[1] for x in ptpts])*1.0/tot])
    # for x in parr:
    #     print x
    plt.plot([x[0] for x in parr],[x[1] for x in parr],'.r-',markersize=20)
    
    plt.grid(True)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'zwalk-' + tmpstr + '.png')
    plt.show()
    plt.close()   

# plot_zwalk('blah',1972,2014,elections)
# myelecs = ['2012_WI_9','2014_WI_9','2016_WI_9','2012_NC_9','2014_NC_9','2016_NC_9']
# myelecs = ['2012_FL_9','2014_FL_9','2016_FL_9','2012_OR_9','2014_OR_9','2016_OR_9']
# '2012_VA_11','2014_IL_11','2014_GA_11','2012_MN_11']
# make_regression_grid('linreg-FLOR',electionsa,myelecs)
yrmin = 2012
chm = '11'
# elecs = filter(lambda x: (yrmin+4) >= int(x.yr) >= yrmin and x.chamber == chm, \
#     electionsa.values())
# plot_many_lines('cong-new-scatter',elecs,False)

# plot regressions

def plot_regress_lines(k,elections):
    """
    """
    plt.figure(figsize=(12,8))
    elec = elections[k]
    vals = sorted(elec.demfrac)
    N = len(vals)
    m = len(filter(lambda x: x < 0.5, vals))
    n = N-m
    ans,m1,i1,m2,i2 = find_lines(elec.state,vals)
        
    x1 = range(1,m+1)
    x2 = range(m+1,N+1)
    y1 = vals[:m]
    y2 = vals[m:]
    plt.scatter(x1,y1,color = 'red')
    plt.scatter(x2,y2,color = 'blue')
    xvals1 = np.linspace(1,m,m+1)
    xvals2 = np.linspace(m+1,N,n+1)
    xvals3 = np.linspace(1,N,10)
    if ans != None:
        plt.plot(xvals1, m1*xvals1+i1, 'r')
        plt.plot(xvals2, m2*xvals2+i2, 'b')
    plt.plot(xvals3, 0.5 + 0*xvals3, color = 'black')
    
    # plt.set_xticks([1,N])
    # plt.set_ylim(0,1)
    yr,st,chm = k.split('_')
    # plt.set_title(st + ' ' + yr)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/regress-' + k + '.png')
    plt.show()
    plt.close()   

# Huh - though 1988_AK_9 should have been imputed....
# - oh, mmd
# Add in r^2 and p-values? Maybe 1-gap too
# Oh, maybe that's why correlations for '9' races were all screwed up
# for x in mmd_dict.keys():
#     print x,mmd_dict[x]
for k in electionsa.keys():
    if int(k[:4]) >= 1972 and \
        (k[:4] not in mmd_dict.keys() or k[5:7] not in mmd_dict[k[:4]]):
        plot_regress_lines(k,electionsa)

        
