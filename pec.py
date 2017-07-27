#############################################################
# Gregory S. Warrington
# May 25, 2017
# gregory.warrington@uvm.edu
#
# Code to compute the declination and its variants as
# described in
# http://www.cems.uvm.edu/~gswarrin/research/research.html#gerrymander
# 
# Input to each function is the fraction of the two-party vote
# a given party wins. Functions assume that any uncontested races
# have been imputed.
#############################################################

import math
import numpy as np

def get_declination(vals):
    """ Compute the declination of an election
    """
    bel = sorted(filter(lambda x: x <=  0.5, vals))
    abo = sorted(filter(lambda x: x > 0.5, vals))

    # Undefined if each party does not win at least one seat
    if len(bel) < 1 or len(abo) < 1:
        return False

    theta = np.arctan((1-2*np.mean(bel))*len(vals)/len(bel))
    gamma = np.arctan((2*np.mean(abo)-1)*len(vals)/len(abo))

    # A little extra precision just in case :)
    declination = 2.0*(gamma-theta)/3.1415926535 

def get_declination_tilde(vals):
    """ Compute a variation of the declination that is somewhat 
    independent of the number of districts
    """
    dec = get_declination(vals):
    if not dec:
        return False
    return declination*math.log(len(vals))/2

def get_declination_seats(vals):
    """ Compute a variation of the declination that estimates the
    number of seats switched due to gerrymandering
    """
    dec = get_declination(vals):
    if not dec:
        return False
    return declination*len(vals)*1.0/2


