# how much does dec change for a single pack/crack based on k and k'
# as iteratively continue packing/cracking what does curve of declination look like
# 

def undistribute_votes(arr,votes,stidx,enidx,minval):
    """ undoes distribute_votes
    """
    narr = sorted([x for x in arr])
    amtper = votes*1.0/(enidx-stidx)
    allfit = True
    notfit = 0.0
    # print "--: ",stidx,enidx,amtper
    for j in range(stidx,enidx):
        if narr[j]-amtper > minval:
            narr[j] -= amtper
        else:
            allfit = False
            notfit = amtper - (narr[j]-minval)
            narr[j] = minval
    return allfit,notfit,sorted(narr)

###############################################################################
###############################################################################
def uncrack_or_unpack(arr,uncrack=True,verbose=False):
    """ uncrack or unpack if possible
    """
    N = len(arr)
    delx = 1.0/(2*N)
    xvals = np.linspace(delx,1-delx,N)

    # figure out which district is being modified
    narr = sorted([x for x in arr])
    if min(narr) > 0.5 or max(narr) <= 0.5:
        if verbose: print "One side won everything. Failing"
        return False,narr

    idx = N-1
    while idx >= 0 and narr[idx] > 0.5:
        idx -= 1

    # set up parameters for filling things in
    minval = 0
    if uncrack:
        minval = 0.0
        stidx = 0
        enidx = idx
    else: # unpack
        minval = 0.501
        stidx = idx+1
        enidx = N
        # lr = stats.linregress(xvals[0:idx],narr[0:idx])

        # print "iiiiiiiiiiiiiiiiiiiii",arr,idx,N
    lr = stats.linregress(xvals[idx+1:N],narr[idx+1:N])
    # how much room we have in other districts for the votes we're trying to move
    if uncrack:
        room = sum(narr[stidx:enidx]) - idx*minval
    else:
        room = sum(narr[stidx:enidx]) - (N-idx-1)*minval
    # new value for district we're cracking
    nval = max(0.501,lr[1] + lr[0]*delx*(2*idx + 1))
    # amount we're changing that one district
    diff = nval-narr[idx]
    # print diff,nval,room
    # see if we have enough room to crack the votes
    if room < diff:
        nval = 0.501
        diff = nval-narr[idx]
        if room < diff:
            if verbose: print "Not enough room to uncrack/unpack votes; returning original"
            return False,arr
    # iteratively move the votes
    narr[idx] = nval
    allfit = False

    while not allfit:
        allfit,notfit,parr = undistribute_votes(narr,diff,stidx,enidx,minval,False)
        # print allfit,notfit,parr
        narr = parr
        diff = notfit
        if not allfit:
            while narr[stidx] == minval:
                stidx += 1
        # print "blah: ",narr
    # print "Voila: ",narr
    return True,narr

def pack_std(N,demfrac,resp=2,do_pack=True):
    """ pack a standard response distribution until you can't anymore; see what happens to angle
    """
    delx = 1.0/(2*N)
    xvals = np.linspace(delx,1-delx,N)
    fig = plt.figure(figsize=(8,8))

    ans = []
    delv = 1.0/(resp*N)
    adj = demfrac-0.5/resp
    vals = [min(max(0,adj+delv*i),1) for i in range(N)]
    if min(vals) < 0 or max(vals) > 1:
        print "Went out of range"
        return

    print "Vals: ",vals
    # figure out district we're going to start with
    idx = 0
    while vals[idx] <= 0.5:
        idx += 1

    ans.append([get_declination('',vals),vals])
    plt.plot(xvals,vals,'ro')
    tmp = [ans[-1][0]]

    if do_pack:
        can_pack = True
        while can_pack:
            can_pack,narr = pack_or_crack(vals,False)
            vals = narr
            idx += 1
            
            ans.append([get_declination('',vals),vals])
            print np.mean(vals),vals
            # ans.append([get_tau_gap(vals,1),vals])
            plt.plot(xvals,vals,linestyle='dashed')
            tmp.append(ans[-1][0])
    else:
        can_crack = True
        while can_crack:
            can_crack,narr = pack_or_crack(vals,True)
            vals = narr
            idx += 1
            
            if idx < N:
                ans.append([get_declination('',vals),vals])
                # ans.append([get_tau_gap(vals,1),vals])
                plt.plot(xvals,vals,linestyle='dashed')
                tmp.append(ans[-1][0])
            else:
                can_crack = False

    plt.plot(xvals[:len(tmp)],tmp)
    # for x in ans:
    #    print x
        
    plt.savefig('/home/gswarrin/research/gerrymander/pics/packstd')
    plt.close()

def count_extra_seats(vals,verbose=False):
    """ straighten out via unpacking/uncracking as efficiently as possible
    assume angle is positive (gerrymandered in favor of republicans)
    """
    fa = get_declination('',vals)
    if fa < 0:
        nvals = sorted([1-x for x in vals])
        return -count_extra_seats_dem(nvals,verbose)
    else:
        nvals = sorted([x for x in vals])
        return count_extra_seats_dem(nvals,verbose)

def count_extra_seats_dem(vals,verbose=False):
    """ straighten out via unpacking/uncracking as efficiently as possible
    assume angle is positive (gerrymandered in favor of republicans)
    """
    notdone = True
    cnt = 0
    arr = sorted([x for x in vals])
    fa_cur = get_declination('',arr)
    while notdone and cnt < 200:
        bc,narrc = uncrack_or_unpack(arr,True)
        bp,narrp = uncrack_or_unpack(arr,False)
        fa_crack = 'None'
        fa_pack = 'None'
        if bc:
            fa_crack = get_declination('',narrc)
        if bp:
            fa_pack = get_declination('',narrp)
        if verbose:
            print "CUR: ",fa_cur,arr
            print "crack: ",bc,fa_crack,["%.3f" % (x) for x in narrc]
            print "pack:  ",bp,fa_pack,["%.3f" % (x) for x in narrp]
        next_move = ''
        if bc and bp:
            # as close to 0 as we can get
            if abs(fa_crack) >= abs(fa_cur) and abs(fa_pack) >= abs(fa_cur):
                return cnt
            elif abs(fa_crack) < abs(fa_cur) and abs(fa_pack < fa_cur):
                if abs(fa_crack) < abs(fa_pack):
                    arr = narrc
                    next_move = 'Crack'
                    fa_cur = fa_crack
                else:
                    arr = narrp
                    next_move = 'Pack'
                    fa_cur = fa_pack
            elif abs(fa_crack) < abs(fa_cur):
                arr = narrc
                next_move = 'Crack'
                fa_cur = fa_crack
            else:
                arr = narrp
                next_move = 'Pack'
                fa_cur = fa_pack
        elif bc:
            # fa_crack = get_declination('',narrc)
            if abs(fa_crack) < abs(fa_cur):
                arr = narrc
                next_move = 'Crack'
                fa_cur = fa_crack
            else:
                return cnt
        elif bp:
            # fa_pack = get_declination('',narrp)
            if abs(fa_pack) < abs(fa_cur):
                arr = narrp
                next_move = 'Pack'
                fa_cur = fa_pack
            else:
                return cnt
        else:
            return cnt

        if verbose: print "next move: ",next_move
        cnt += 1
    return cnt

def chart_dec_total(elections,mmd,chamber):
    """
    """
    yrarr = []
    arr = []
    xarr = []
    yarr = []
    for yr in range(1972,2018,2):
        cnt = 0
        hasone = False
        for elec in elections.values():
            if elec.chamber == chamber and int(elec.yr) == yr and elec.Ndists >= 4 and \
               (chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
                hasone = True
                tmp = count_extra_seats(elec.demfrac)
                cnt += tmp
                nseats = get_declination('',elec.demfrac)*elec.Ndists*1.0/2
                xarr.append(nseats)
                yarr.append(tmp)
                print "Elec %s %s %s %d % .3f % .3f" % (elec.yr,elec.state,elec.chamber,elec.Ndists,nseats,tmp)
        if hasone:
            print yr,cnt
            arr.append(cnt)
            yrarr.append(yr)
    plt.figure(figsize=(12,8))
    plt.plot(yrarr,arr,'ro-')
    plt.savefig('/home/gswarrin/research/gerrymander/pics/seatsovertime' + chamber)
    plt.close()

    plt.figure(figsize=(12,8))
    plt.scatter(xarr,yarr)
    plt.savefig('/home/gswarrin/research/gerrymander/pics/seatsovertime-scatter' + chamber)
    plt.close()

