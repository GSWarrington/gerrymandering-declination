#############################################################################
# Code for investigating the interaction between clustering and district size
#############################################################################
def make_trapezoid(N,x0,y0=0.45,y1=1,verbose=True):
    """ get vote distribution from a trapezoid
    x0 is where angle starts sloping up
    x1 is where it levels off
    ___x0////x1----x2\\\\x3___
    D's and R's should each win half of the vote
    """
    # x1 = (0.5-y1)/(y0-y1) - x0
    # x2 = 1 - x1
    # x3 = 1 - x0
    x2 = (0.5-y0)/(y1-y0) + x0
    x1 = 1 - x2
    x3 = 1 - x0

    if x1 > x2:
        return []

    arr = []
    delx = 1.0/N
    xinit = 1.0/(2*N)

    if verbose:
        print "%.2f %.2f %.2f %.2f" % (x0,x1,x2,x3)

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

#############################################################################
def get_district_vote(vals,st,en):
    """
    get average democratic vote in a given district. okay if indices wrap around
    """
    # print vals,st,en
    if en > st:
        return np.mean(vals[st:en])
    else:
        return np.mean(vals[st:] + vals[:en])

#############################################################################
def get_district_boundaries(Nprec,Ndist):
    """
    get district boundaries for a given number of precincts and districts
    doesn't try to interleave slightly different size districts
    """
    Ssz = int(Nprec/Ndist)        # size of smaller districts
    NumLarge = Nprec - Ndist*Ssz  # number we'll have to make bigger
    if NumLarge > 0:
        Freq = int(math.ceil(Ndist/NumLarge))         # how frequently we'll add them in
    else:
        Freq = 0

    # print Ssz,NumLarge,Freq
    # make an array keeping track of the sizes for the districts
    szarr = [Ssz for i in range(Ndist)]
    for i in range(Ndist):
        if Freq > 0 and ((i%Freq) == 0):
            szarr[i] += 1

    # add in extra large ones if we need to
    i = 0
    numlarge = len(filter(lambda x: x == Ssz + 1,szarr))
    while numlarge < NumLarge:
        if szarr[i] == Ssz:
            szarr[i] += 1
            numlarge += 1
        i += 1
        
    # make the actual districts
    dists = []
    st = 0
    cur = 0
    for i in range(Ndist):
        cur = st + szarr[i]
        dists.append([st,cur])
        st = cur

    return dists

#############################################################################
def shift_district_boundaries(Nprec,arr,delx):
    """ 
    shift the district boundaries by the specified amount
    """
    narr = []
    for x in arr:
        if x[1]+delx < Nprec:
            narr.append([x[0] + delx,x[1] + delx])
        else:
            narr.append([x[0] + delx,x[1] + delx - Nprec])
    return narr

#############################################################################
def average_won(vals,Nprec,Ndist):
    """
    compute the average number of districts won as we cyclically shift the district boundaries
    """
    frac_arr = []
    dists = get_district_boundaries(Nprec,Ndist)
    # print "vals: ",vals
    for i in range(dists[0][1]-dists[0][0]):
        sdists = shift_district_boundaries(Nprec,dists,i)
        # print "asdf: ",Ndist,len(dists),len(sdists)
        # print " -- ",sdists
        avgs = [get_district_vote(vals,x[0],x[1]) for x in sdists]
        # print "avgs: ",avgs
        frac_arr.append(len(filter(lambda x: x > 0.5, avgs))*1.0/Ndist)
        # print i,map(lambda x: "%.2f" % (x), avgs)
    # print "frac: ",frac_arr
    return np.mean(frac_arr)

#############################################################################
def cluster_ex(Nprecs,y0=0.3,maxDists=100):
    """
    try different numbers of districts
    """
    plt.figure(figsize=(8,8))

    for x0 in np.linspace(0,0.49,50):
        arr = []
        vals = make_trapezoid(Nprecs,x0,y0)
        if len(vals) == 0:
            continue

        for Nd in range(1,min(maxDists,Nprecs)+1):
            arr.append([Nd,average_won(vals,Nprecs,Nd)])
            print arr[-1]

        plt.scatter([x[0] for x in arr],[x[1] for x in arr])

    plt.gca().set_xlabel('Number of districts')
    plt.gca().set_ylabel('Fraction democratic')
    plt.savefig('/home/gswarrin/research/gerrymander/pics/' + 'clusterex') #  + (("%.2f" % (x0))[1:]))
    plt.close()

    list_plot(arr)
    
