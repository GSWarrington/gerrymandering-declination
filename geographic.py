def pr_elecs(elecs):
    """ run through yr,st,districts and print info
    """
    for elec in elecs.values()[:10]:
        if elec.chamber == '11':
            print elec.yr,elec.state
            for i in range(elec.Ndists):
                print "   ",elec.dists[i],elec.demfrac[i]
