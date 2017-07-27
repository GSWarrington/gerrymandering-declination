# for st in states:
#     plot_race('2014',st,'11')
duh = []
aaa = []
for elec in elections.values():
    if 1 == 1: # 2012 <= int(elec.yr) <= 2014 and elec.chamber == '11':
        if not math.isnan(elec.adjgaps[-1]) and elec.Ndists >= 8:
            duh.append([elec.adjgaps[-1],elec.state,elec.Ndists])
            aaa.append([elec.Dfrac,elec.adjgaps[-1]])
            # aaa.append(elec.
make_histogram('newall.png',[x[0] for x in duh])
make_scatter('newallbias.png',[x[0] for x in aaa],[x[1] for x in aaa])
for x in sorted(duh):
    print x
def generate_one_run(p,std,alpha,N):
    """
    """
    m = 0
    ans = 0
    samples = filter(lambda x: 1 >= x >= 0, np.random.normal(p,std,N))
    for sample in samples:
        ai = 2*sample-1
        if ai > 0:
            ans += pow(ai,alpha+1)
            m += 1
        else:
            ans -= pow((-ai),alpha+1)
    return (2.0/(alpha+2))*(0.5 - m*1.0/len(samples) + ans*1.0/len(samples))

def find_bias(alpha,std0,tot):
    """ generate tot samples
    """
    Nsizes = 10 + np.random.binomial(60,0.5,tot)
    stds = filter(lambda x: 0.5 >= x > 0, np.random.normal(std0,std0/2,tot))
    pvals = filter(lambda x: 1 >= x >= 0, np.random.normal(0.5,0.2,tot))
    ans = []
    avgimb = 0
    for i in range(min(len(pvals),len(stds))):
        imb = generate_one_run(pvals[i],stds[i],alpha,Nsizes[i])
        avgimb += imb
        ans.append([pvals[i],imb])
    # return avgimb*1.0/len(ans)
    # print "avg: ",avgimb*1.0/len(ans)
    # slope, intercept, r_value, p_value, std_err = stats.linregress([x[0] for x in ans],[x[1] for x in ans])
    # print r_value**2
    # print slope, intercept

    mystr = "bias-%.2f-alpha-%.1f.png" % (std0,alpha)
    make_scatter(mystr,[x[0] for x in ans], [x[1] for x in ans])

def find_std(alpha,p):
    """ try to find standard deviation assumed by egap for given alpha and p
    """
    isdone = False
    std = 0.1
    while not isdone:
        # pick a random district size
        N = 10 + np.random.binomial(100,0.5)
        pvals = filter(lambda x: 1 >= x >= 0, np.random.normal(p,std,N))
        N = len(pvals)
        ans = []
        avgimb = 0
        for i in range(min(len(pvals),len(stds))):
            imb = generate_one_run(pvals[i],stds[i],alpha,Nsizes[i])
            avgimb += imb
            ans.append([pvals[i],imb])
    
            stds = filter(lambda x: 0.5 >= x > 0, np.random.normal(std0,std0/2,tot))
        ans = []
        for i in range(min(len(pvals),len(stds))):
            imb = generate_one_run(pvals[i],stds[i],alpha,Nsizes[i])
            ans.append([pvals[i],imb])
        print "avg: ",avgimb*1.0/len(ans)
        slope, intercept, r_value, p_value, std_err = stats.linregress([x[0] for x in ans],[x[1] for x in ans])
        print r_value**2
        print slope, intercept
    
    mystr = "bias-%.2f-alpha-%.1f.png" % (std0,alpha)
    make_scatter(mystr,[x[0] for x in ans], [x[1] for x in ans])

def make_bias_pics():
    """
    """
    for std0 in [0.8]: # [0.01,0.1,0.3,0.5]:
        for tot in [5000]:
            for alpha in [0,1,2]:
                find_bias(alpha,std0,tot)
                    
# find_bias(2,5000)
find_bias(0,0.1,

          
