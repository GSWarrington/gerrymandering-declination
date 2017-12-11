#########################################################################
# code to compute tau-gap, EG and declination from vote distribution data
#########################################################################

def get_tau_gap(vals,tau):
    """ compute tau-gap. 

    Note that tau-gap when tau=0 is twice the EG
    TODO: Review code for when tau < 0.
    """  
    ans = 0.0
    m = 0.0
    N = len(vals)
    for i in range(N):
        ai = 2.0*vals[i]-1
        if ai > 0:
            m += 1
        tmp = 0.0
        if tau >= 0: # votes close to 50% are weighed more ("traditional")
            if ai >= 0:  
                tmp = pow(ai,tau+1)
            else:
                tmp = -pow(-ai,tau+1)
        else:             # votes close to 50% are weighed less 
            if ai >= 0:
                tmp = pow((1-ai),-tau+1)
            else:
                tmp = -pow((1+ai),-tau+1)
        ans += tmp
                    
    if tau >= 0:
        return 2.0*(ans/N + 0.5 - m/N)
    else:
        return -2.0*(ans/N + 0.5 - m/N)

def get_EG(vals):
    """ return the efficiency gap
    """
    return get_tau_gap(vals,0)/2

def get_declination(st,vals):
    """ Get declination.

    Expressed as a fraction of 90 degrees
    """
    bel = sorted(filter(lambda x: x <=  0.5, vals))
    abo = sorted(filter(lambda x: x > 0.5, vals))
    if len(bel) < 1 or len(abo) < 1:
        return -2.0

    theta = np.arctan((1-2*np.mean(bel))*len(vals)/len(bel))
    gamma = np.arctan((2*np.mean(abo)-1)*len(vals)/len(abo))

    return 2.0*(gamma-theta)/3.1415926535 # Enough precision for you?
    
