def pr_elecs(elecs):
    """ run through yr,st,districts and print info
    """
    for elec in elecs.values()[:10]:
        if elec.chamber == '11':
            print elec.yr,elec.state
            for i in range(elec.Ndists):
                print "   ",elec.dists[i],elec.demfrac[i]

def chart_geo_bias(fn,arr):
    """ takes precinct-level vote distribution 
    computes percent dem legislators for varying # districts
    """
    xarr = []
    yarr = []
    # run through various district sizes
    for cpow,dsize in enumerate([int(math.pow(2,m)) for m in range(int(math.log(len(arr),2)))]):
        Ndists = len(arr)/dsize
        # run through various offsets
        ttt = []
        for j in range(dsize):
            dvotes = []
            # run through various districts
            for k in range(Ndists-1):
                dvotes.append(np.mean(arr[k*dsize+j:(k+1)*dsize+j]))
            dvotes.append(np.mean(arr[:j] + arr[(Ndists-1)*dsize+j:]))
            # print cpow,dsize,dvotes
            # dvotes = [np.mean(arr[k*dsize+j:k*(dsize+1)+j]) for k in range(Ndists)]
            tmp = 0
            for y in dvotes:
                if abs(y-0.5) < 0.0001:
                    tmp += 0.5
                elif y-0.5 >= 0.0001:
                    tmp += 1
            # wontot = len(filter(lambda x: x > 0.5, dvotes))
            # tietot = len(filter(lambda x: x == 0.5, dvotes))*0.5
            # print cpow,dsize,wontot,tietot
            ttt.append(tmp*1.0/Ndists)
        yarr.append(np.mean(ttt)) # tmp*1.0/Ndists)
        xarr.append(cpow)

    plt.figure(figsize=(8,8))
    plt.scatter(xarr,yarr)
    plt.gca().set_xlabel('Log of district size')
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    plt.close()

def try_bias(fn):
    """ hand code a distribution
    """
    arr = [0.47 for x in range(1024)]
    for i in range(64):
        arr[16*i] = 0.95
    print "avg dem: ",np.mean(arr)
    chart_geo_bias(fn,arr)

# try_bias('simptower')

def try_bias_2(fn):
    """ simple gaussian
    """
    N = int(math.pow(2,12))
    xarr = np.linspace(-6,6,N)
    arr = [1+math.exp(-x*x*x*x)/2 for x in xarr]
    avg = np.mean(arr)
    narr = [x-(avg-0.5) for x in arr]
    narr = [min(1,x) for x in narr]
    navg = np.mean(narr)
    parr = [x-(navg-0.5) for x in narr]
    print "avg dem: ",np.mean(parr)
    chart_geo_bias(fn,parr)
    make_scatter('blah',xarr,parr)

#######################################################################
#######################################################################
def get_dist_inds(N,s0,n0,s1,n1):
    """ N total districts; n0 of size s0; n1 of size s1. Find start and end indices
    """
    searr = []
    for j in range(n0):
        searr.append([j*s0:(j+1)*s0])
    for j in range(n1):
        searr.append([s0*n0 + j*s1: s0*n0 + (j+1)*s1])
    return searr

def make_trapezoid(N,x0,y0=0.45,y1=1):
    """ get vote distribution from a trapezoid
    """
    x1 = (0.5-y1)/(y0-y1) - x0
    x2 = 1 - x1
    x3 = 1 - x2

    arr = []
    delx = 1.0/N
    xinit = 1.0/(2*N)

    for i in range(N):
        xi = xinit + i*delx
        if xi < x0:
            arr.append(y0)
        elif x0 <= xi < x1:
            arr.append(y0 + (y1-y0)*(xi-x0)/(x1-x0))
        elif x1 <= xi < x2:
            arr.append(y1)
        elif x2 <= xi < x3:
            arr.append(y1 - (y1-y0)*(xi-x2)/(x3-x2))
        elif xi >= x3:
            arr.append(y0)
    return arr

def get_vote_vals(trapz,dists,offset):
    """ get vote val in each district for a given offset
    """
    vals = []
    for j in range(len(dists)):
        if offset + dists[j][1] < len(trapz):
            vals.append(np.mean(trapz[offset + dists[j][0]:offset + dists[j][1]]))
        else:
            vals.append(np.mean(trapz[offset + dists[j][0]:] + \
                                trapz[:((offset + dists[j][1])%len(trapz))]))
    return vals

#########################################################################
#########################################################################
def get_size_seq(N,k):
    """ divide seq of size N into k pieces each of comparable size
    if overshoot then undershoot for a while
    """

def chart_geo_bias_linear(fn,arr):
    """ takes precinct-level vote distribution 
        computes percent dem legislators for varying # districts
    """
    xarr = []
    yarr = []
    # run through various district sizes
    for Ndists in range(1,len(arr)+1):
        dsize = 
        Ndists = len(arr)/dsize
        # run through various offsets
        ttt = []
        for j in range(dsize):
            dvotes = []
            # run through various districts
            for k in range(Ndists-1):
                dvotes.append(np.mean(arr[k*dsize+j:(k+1)*dsize+j]))
            dvotes.append(np.mean(arr[:j] + arr[(Ndists-1)*dsize+j:]))
            # print cpow,dsize,dvotes
            # dvotes = [np.mean(arr[k*dsize+j:k*(dsize+1)+j]) for k in range(Ndists)]
            tmp = 0
            for y in dvotes:
                if abs(y-0.5) < 0.0001:
                    tmp += 0.5
                elif y-0.5 >= 0.0001:
                    tmp += 1
            # wontot = len(filter(lambda x: x > 0.5, dvotes))
            # tietot = len(filter(lambda x: x == 0.5, dvotes))*0.5
            # print cpow,dsize,wontot,tietot
            ttt.append(tmp*1.0/Ndists)
        yarr.append(np.mean(ttt)) # tmp*1.0/Ndists)
        xarr.append(cpow)

    plt.figure(figsize=(8,8))
    plt.scatter(xarr,yarr)
    plt.gca().set_xlabel('Log of district size')
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + fn)
    plt.close()


# try_bias_2('gauss')
