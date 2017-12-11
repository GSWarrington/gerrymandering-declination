# def plot_uncontestedness(elections,cycles):
#     """ look at how uncontested races affect variance of declination
#     """
    
def get_frac_uncontested(elec):
    """ return the fraction of races in the election that were uncontested
    """
    return len(filter(lambda i: elec.status[i] < 2, range(elec.Ndists)))*1.0/elec.Ndists

def plot_uncontestedness_delta(elections,mmd):
    """ look at uncontestedness percent changes year over year

    Issues to be concerned about with imputation
    - wild changes in percent uncontested -> wild changes in declination
    - systematic bias in imputation affecting declination
    - keep track of how many elections have given percentage

    """
    unarr = []
    dearr = []
    Narr = []
    farr = []
    totraces = 0
    totuncont = 0
    for elec in elections.values():
        # skip elections with multimember districts
        if elec.chamber == '9' and elec.yr in mmd.keys() and elec.state in mmd[elec.yr]:
            continue
        # if elec.Ndists < 5:
        #     continue
        if elec.chamber == '9':
            continue
        if int(elec.yr) < 1972:
            continue
        new_yr = str(int(elec.yr)+2)
        new_key = '_'.join([new_yr,elec.state,elec.chamber])
        if new_key in elections.keys():
            totraces += elec.Ndists
            totuncont += len(filter(lambda i: elec.status[i] < 2, range(elec.Ndists)))
            nelec = elections[new_key]
            ori_un = get_frac_uncontested(elec)
            new_un = get_frac_uncontested(nelec)
            ori_dec = get_declination('',elec.demfrac)
            new_dec = get_declination('',nelec.demfrac)
            # declination isn't defined one of two years
            if abs(ori_dec) == 2 or abs(new_dec) == 2:
                continue
            print "%s %s %s %d: %.2f %.2f" % (elec.yr,elec.state,elec.chamber,elec.Ndists,new_un-ori_un,new_dec-ori_dec)
            # print "%s %s %s %d %.2f" % (elec.yr,elec.state,elec.chamber,elec.Ndists,ori_un)
            unarr.append(abs(new_un-ori_un))
            dearr.append(abs(new_dec-ori_dec))
            Narr.append(nelec.Ndists)
            farr.append(ori_un)
    make_scatter('un',unarr,dearr)
    make_scatter('Nvun',Narr,unarr)    
    make_scatter('fun',Narr,farr)                
    print totuncont,totraces

