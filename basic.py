def print_pi(elecs,myid):
    """
    """
    elec = elecs[myid]
    for i in range(elec.Ndists):
        print elec.demfrac[i]

def compute_alpha_curve(vals,alpha):
    """ compute alpha-based curve for seq of percent won
    """  
    ans = 0.0
    m = 0.0
    N = len(vals)
    for i in range(N):
        ai = 2.0*vals[i]-1
        if ai > 0:
            m += 1
        tmp = 0.0
        if alpha >= 0: # votes close to 50% are weighed more ("traditional")
            if ai >= 0:  
                tmp = pow(ai,alpha+1)
            else:
                tmp = -pow(-ai,alpha+1)
        else:             # votes close to 50% are weighed less 
            if ai >= 0:
                tmp = pow((1-ai),-alpha+1)
            else:
                tmp = -pow((1+ai),-alpha+1)
        ans += tmp
        # print "%.2f %d %.2f %.2f" % (alpha,i,ai,tmp)
                    
    if alpha >= 0:
        return 2.0*(ans/N + 0.5 - m/N)
    else:
        return -2.0*(ans/N + 0.5 - m/N)

def find_angle(st,vals):
    """ Now expressed as a fraction of 90 degrees
    """
    bel = sorted(filter(lambda x: x <=  0.5, vals))
    abo = sorted(filter(lambda x: x > 0.5, vals))
    if len(bel) < 1 or len(abo) < 1:
        return -2.0
    # print np.mean(bel),np.mean(abo)
    theta = np.arctan((1-2*np.mean(bel))*len(vals)/len(bel))
    gamma = np.arctan((2*np.mean(abo)-1)*len(vals)/len(abo))
    # print "theta: %.3f gamma: %.3f" % (theta,gamma)
    print gamma,theta
    print len(bel)*1.0/len(vals)
    
    return 2.0*(gamma-theta)/3.1415926535 # len(vals). Hah!
    
def compute_egap_directly(vals):
    """
    """
    ans = 0
    for x in vals:
        if x >= 0.5:
            ans += (x-0.5) # votes wasted by D winner
            ans -= (1-x)   # votes wasted by R loser
        else:
            ans += x       # votes wasted by D loser
            ans -= (0.5-x) # votes wasted by R winner
    return ans/(len(vals))
