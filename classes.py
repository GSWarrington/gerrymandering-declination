# 3 - database for keeping track of elections
#############################################
# functions
# make_records - makes records from raw data read in from read_all_data()
# - get_cyc_state() - modify state name to account for different plans in the same decade
# - regularize_races() - create dummy candidates for uncontested races so no null (ie, None) fields
#                      - flag races that need to be imputed
# TODO: Currently in read_data - but called from classes. Move?
# - correct_errors() - manually override a few races

############################################
# 11.30.17
# Cycle
#   * get_model_input_data
#   * fit_model
#   * populate_elections
#   * get_new_dist_data
#   * impute_votes
#
# Candy
#   mycopy
#
# Gerry
#   mycopy
#   myprint
#   * regularize_races
#
# * get_cyc_state
# * make_records
# * print_race
# * print_all_races
# * create_cycles

class Cycle:
    """ store all elections in a given cycle
    # Reviewed 12.01.17
    """
    def __init__(self,min_year,max_year,chamber):
        """ typically xxx2, xxx4, ..., xxx0
        stores all the elections in a given cycle along with model for imputing values
        """
        self.min_year = min_year     # earliest allowed year of cycle (inclusive)
        self.max_year = max_year     # latest allowed year of cycle (inclusive)
        self.chamber = chamber       # chamber considered
        self.years = []              # TODO: ?
        self.states = []             # TODO: ?
        self.elecs = dict()          # TODO: ?
        self.leveliii_fit = None     # What is actually returned by pystan
        self.leveliii_data = None    # the data needed to fit the model
        self.state_lookup = None     # keeps track of which state a given race is in
        self.year_lookup = None      # keeps track of which year a given race is in
        self.dist_lookup = None      # keeps track of which district a given race is in
        self.alldist_lookup = None   # districts that haven't been contested at all (hence not in dist_lookup)

        self.dwin = [0,0] # mean and std of consistent dem wins in the cycle (for predicting for new districts)
        self.rwin = [0,0] # mean and std of consistent rep wins in the cycle (for predicting for new districts)
        
    def get_model_input_data(self):
        """ Return election data as a collection of independent arrays

        # Reviewed 12.01.17
        """
        # Run through the elections and then through the districts,
        # appending relevant info to corresponding array
        ild = []   # 1 if Dem incumbent, 0 if not
        ilr = []   # 1 if Rep incumbent, 0 if not
        pv = []    # Dem fraction of vote in the district
        sts = []   # cycle state
        dists = [] # what district race is in
        yrs = []   # year of race
        wind = []  # 1 if Dem winner, 0 if Rep winner
        winr = []  # 1 if Rep winner, 0 if Dem winner
        for elec in self.elecs.values():
            for i in range(elec.Ndists):
                # since this is data we'll be using to impute votes, don't include race if data already imputed.
                if elec.status[i] == 2:
                    if elec.dcands[i].winner:
                        wind.append(1)
                    else:
                        wind.append(0)
                    if elec.rcands[i].winner:
                        winr.append(1)
                    else:
                        winr.append(0)

                    if elec.dcands[i].incum:
                        ild.append(1)
                    else:
                        ild.append(0)
                    if elec.rcands[i].incum:
                        ilr.append(1)
                    else:
                        ilr.append(0)
                    dists.append(elec.dists[i])
                    pv.append(elec.demfrac[i])
                    sts.append(elec.cyc_state)
                    yrs.append(elec.yr)
        return wind,winr,ild,ilr,pv,sts,dists,yrs

    def fit_model(self):
        """ run the hierarchical model to impute votes

        # Reviewed 12.01.17
        """
        winD_Ni,winR_Ni,incumbentsD_Ni,incumbentsR_Ni,demvote_Ni,states_Ni,districts_Ni,years_Ni = self.get_model_input_data()
        
        # get list of all states seen
        setstates = sorted(list(set(states_Ni)))
        # put them into a dictionary so can assign a number to each
        state_lookup = dict(zip(setstates, range(len(setstates))))
        # replace list of states with indices of them
        statenums = map(lambda x: state_lookup[x]+1, states_Ni) 
        
        # get list of all years seen
        setyears = sorted(list(set(years_Ni)))
        # put them into a dictionary so can assign a number to each
        year_lookup = dict(zip(setyears, range(len(setyears))))
        # replace list of yearss with indices of them
        yearnums = map(lambda x: year_lookup[x]+1, years_Ni) 
        
        # get list of all districts seen
        setdists = sorted(list(set(districts_Ni)))
        # put them into a dictionary so can assign a number to each
        dist_lookup = dict(zip(setdists, range(len(setdists))))
        # replace list of districts with indices of them
        distnums = map(lambda x: dist_lookup[x]+1, districts_Ni) 
        # print "distnums: ",distnums
        
        # need a lookup that tells us which state number corresponds to a given district number
        d = dict()
        for i in range(len(districts_Ni)):
            if districts_Ni[i] not in d:
                d[districts_Ni[i]] = states_Ni[i]
        statenum_lookup = []
        for i in range(len(dist_lookup)):
            statenum_lookup.append(state_lookup[d[setdists[i]]]+1)
    
        leveliii_data = {'Ni': len(demvote_Ni),
                       'Nj': len(setdists),
                       'Nk': len(setstates),
                       'Nl': len(setyears),
                       'district_id': distnums,
                       'state_id': statenums,
                       'year': yearnums,
                       'state_Lookup': statenum_lookup,
                       'y': demvote_Ni,
                       'winD': winD_Ni,
                       'winR': winR_Ni,
                       'IncD': incumbentsD_Ni,
                       'IncR': incumbentsR_Ni}
        self.leveliii_data = leveliii_data
        self.state_lookup = state_lookup
        self.year_lookup = year_lookup
        self.dist_lookup = dist_lookup
           
        # if only a few years, will take more to converge
        if int(self.max_year)-int(self.min_year) <= 5:
            niter = 4000
            nchains = 4
        else:
            niter = 2000
            nchains = 2

        # do the actual model fit
        self.leveliii_fit = pystan.stan(model_code=leveliii_model, data=leveliii_data, 
                                                 iter=niter, chains=nchains)
   
    def populate_elections(self,elections):
        """ copy elections into the cycle

        # Reviewed 11.15.17
        """
        for x in elections.keys():
            # make sure year of election is in correct range
            # skip if chamber is wrong
            if int(self.min_year) <= int(elections[x].yr) <= int(self.max_year) and \
                elections[x].chamber == self.chamber:
                # REAPP
                if elections[x].cyc_state not in self.states:
                    self.states.append(elections[x].cyc_state)
                if elections[x].yr not in self.years:
                    self.years.append(elections[x].yr)
                self.elecs[x] = elections[x] # .mycopy() TODO: Do I need to copy?
                                                                   
    def get_new_dist_data(self,verbose=False):
        """ get mean and std dev among districts won by single party every year in cycle
            use these for guessing districts consistently unopposed in cycle
            this is run after model has been fitted, but before imputed votes have been inserted

        Reviewed 12.01.17
        """
        # get a lookup table for *all* districts
        alldists = []
        for elec in self.elecs.values():
            for i in range(elec.Ndists):
                if elec.dists[i] not in alldists:
                    alldists.append(elec.dists[i])
        setalldists = sorted(list(set(alldists)))
        self.alldist_lookup = dict(zip(setalldists, range(len(setalldists))))

        # run through districts and keep track of:
        # - districts consistently won by Dems each year (may need to impute all of them)
        # - districts consistently won by Reps each year (may need to impute all of them)
        # - districts that have had a competitive race during the cycle
        Ndists = len(self.alldist_lookup)
        allDem = [True for x in range(Ndists)]
        allRep = [True for x in range(Ndists)]
        wasComp = [False for x in range(Ndists)]
        # for yr in yrlist:
        # first make orphan_dist_lookup
        for elec in self.elecs.values():
            for i in range(elec.Ndists):
                if elec.status[i] == 2: # don't need to impute this
                    wasComp[self.alldist_lookup[elec.dists[i]]] = True
                if elec.dcands[i].winner: # dems won, so not consistently won by repubs
                    allRep[self.alldist_lookup[elec.dists[i]]] = False
                if elec.rcands[i].winner: # repubs won, so not consistently won by dems
                    allDem[self.alldist_lookup[elec.dists[i]]] = False

        # return lists of indices in alldist_lookup that have what we want
        allDem_fil = []
        allRep_fil = []
        wasComp_fil = []
        for i in range(len(setalldists)):
            if allDem[i] == True:
                allDem_fil.append(setalldists[i])
            if allRep[i] == True:
                allRep_fil.append(setalldists[i])
            if wasComp[i] == True:
                wasComp_fil.append(setalldists[i])

        if len(wasComp_fil) != len(self.dist_lookup):
            print "WARNING: competed districts don't match in length"
            return
        return allDem_fil,allRep_fil,wasComp_fil

    # TODO: Not exactly sure what this is saying
    # need to make sure that all districts in cycle are addressed in the same order
    # that's what's dist_lookup is for

    def impute_votes(self,verbose=False):
        """ run through all races in cycle and impute those that need it

        # Reviewed 12.01.17
        """
        predd = self.leveliii_fit.extract()
        allDem_fil,allRep_fil,wasComp_fil = self.get_new_dist_data() # so we know what to do about districts not in model

        # set up parameters for when we don't have a race in the district during the cycle
        curDdata = []
        curRdata = []
        for curdist in allDem_fil: # want it to be a district Democrats won every year in cycle
            if curdist in wasComp_fil: # need it to have been competed for
                # pull out that column from model fit
                curDdata.append(mean(predd['u_0jk'][:,self.dist_lookup[curdist]]))
        for curdist in allRep_fil: # want it to be a district Democrats won every year in cycle
            if curdist in wasComp_fil: # need it to have been competed for
                # pull out that column from model fit
                curRdata.append(mean(predd['u_0jk'][:,self.dist_lookup[curdist]]))
        make_histogram('Demhisto.png',curDdata)
        make_histogram('Rephisto.png',curRdata)

        self.dwin = [np.mean(curDdata),np.std(curDdata)]
        self.rwin = [np.mean(curRdata),np.std(curRdata)]

        if verbose:
            print "dwin: %s rwin: %s\n" % (self.dwin,self.rwin)
            print self.state_lookup

        for elec in self.elecs.values():
            for i in range(elec.Ndists):
                if elec.status[i] == -1: # shouldn't be at this stage
                    print "WARNING: status not set",elec.yr,elec.cyc_state,elec.chamber,i
                    return
                if elec.status[i] == 2: # don't need to impute
                    print "NOTE: Known. No need to impute %s %s %s %s" % (self.min_year,elec.cyc_state,elec.dists[i],elec.chamber)
                    continue
                if elec.status[i] in [0,1]: # needs to be (re)imputed
                    if elec.dists[i] in self.dist_lookup.keys(): # district is in model
                        print "NOTE: In model %s %s %s %s" % (self.min_year,elec.cyc_state,elec.dists[i],elec.chamber)
                        u0jk = mean(predd['u_0jk'][:,self.dist_lookup[elec.dists[i]]])
                    elif elec.dists[i] in allDem_fil: # impute from consistently won democratic districts
                        print "NOTE: Consistently uncontested %s %s %s %s" % (self.min_year,elec.cyc_state,elec.dists[i],elec.chamber)
                        u0jk = self.dwin[1]*np.random.randn(1)[0] + self.dwin[0]
                    elif elec.dists[i] in allRep_fil: # impute from consistently won republican districts
                        print "NOTE: Consistently uncontested %s %s %s %s" % (self.min_year,elec.cyc_state,elec.dists[i],elec.chamber)
                        u0jk = self.rwin[1]*np.random.randn(1)[0] + self.rwin[0]
                    else:
                        print "NOTE: Rare! never contested, but flipped between parties %s %s %s %s" % \
                            (self.min_year,elec.cyc_state,elec.dists[i],elec.chamber)
                        u0jk = mean(predd['u_0jk'][:,1+randrange(len(predd['u_0jk'][0])-1)])
                        
                    winD = mean(predd['win_D'])
                    winR = mean(predd['win_R'])
                    deltaD = mean(predd['delta_D'])
                    deltaR = mean(predd['delta_R'])
                    # beta0 = mean(predd['beta_0'])
                    beta0 = 0.5 # mean(predd['beta_0'])
                    # REAPP
                    if elec.cyc_state in self.state_lookup:
                        u0k = mean(predd['u_0k'][:,self.state_lookup[elec.cyc_state]])
                    else:
                        u0k = 0.0 # TODO: If no contested elections in state in cycle, then go here.
                                  # Happens in 1982 LA1, should probably use mean of state adjustments...
                    v0l = mean(predd['v_0l'][:,self.year_lookup[elec.yr]])

                    # modify based on winner
                    winnermod = 0
                    if elec.dcands[i] != None and elec.dcands[i].winner:
                        winnermod += winD
                    if elec.rcands[i] != None and elec.rcands[i].winner:
                        winnermod += winR

                    # modify based on incumbency of candidates
                    incummod = 0
                    if elec.dcands[i] != None and elec.dcands[i].incum:
                        incummod += deltaD
                    if elec.rcands[i] != None and elec.rcands[i].incum:
                        incummod += deltaR
                    # final vote fraction we'll assign to this candidate

                    newdv = beta0 + u0k + u0jk + v0l + incummod + winnermod
                    elec.impute_data[i] = [beta0,u0k,u0jk,v0l,incummod,winnermod]

                    # How to deal with imputed value contradicting model
                    if elec.dcands[i].winner and newdv < 0.5:
                        print "WARNING: Imputed winner but value too small: ",
                        print elec.yr,elec.cyc_state,elec.chamber,elec.dists[i],str(elec.dcands[i]),newdv
                        newdv = 0.51
                    elif not elec.dcands[i].winner and newdv > 0.5:
                        print "WARNING: Imputed loser but value too big: ",
                        print elec.yr,elec.cyc_state,elec.chamber,elec.dists[i],str(elec.dcands[i]),newdv
                        newdv = 0.49
                    if verbose:
                        print "Imputing %s %s %s %s beta: %.2f st: % .2f dist: % .2f \
                              yr: % .2f Incum: % .2f winner: % .2f demv: %.4f" % \
                            (elec.yr,elec.cyc_state,elec.chamber,elec.dists[i],beta0,u0k,u0jk,\
                             v0l,incummod,winnermod,newdv)
                    elec.status[i] = 1 # imputed successfully
                    elec.demfrac[i] = newdv
                    elec.dcands[i].votes = int(1000*newdv)
                    elec.dcands[i].is_imputed = True
                    elec.rcands[i].votes = 1000-int(1000*newdv)
                    elec.rcands[i].is_imputed = True
                    
#########################################################################################################
#########################################################################################################
class Candy:
    """ store information relevant to a single candidate
    """
    def __init__(self,race,cname,party,votes,incum,winner):
        self.race = race       # yr_state_chamber
        self.cname = cname
        self.party = party
        self.votes = votes # i guess we'll clobber with imputed value if necessary
        self.incum = incum
        self.winner = winner

        self.orig_votes = None      # don't think this is used anywhere
        self.is_imputed = False     # should be able to set off of status when info read in from a file
        self.frac = float('NaN')    # not sure where this gets computed.... (or used)
        
    def __str__(self):
        """
        """
        return "%s %s %s %.4f %s %s" % (self.cname,self.party,self.votes,self.frac,self.incum,self.winner)
    
    def mycopy(self):
        """
        """
        c = Candy(self.race,self.cname,self.party,self.votes,self.incum,self.winner)
        c.frac = self.frac
        c.orig_votes = self.orig_votes
        c.is_imputed = self.is_imputed
        return c
                    
class Gerry:
    """ store information relevant to a single election (year, state, chamber)
    """
    def __init__(self, yr, state, chamber):
        self.yr = yr
        self.state = state
        self.cyc_state = state                 # for reapportionments mid-decade - for multilevel model - REAPP
        self.chamber = chamber
        self.Ndists = 0                        # number of districts
        self.Ddists = 0                        # number of districts won by democrats
        self.dists = []                        # list of districts (as strings)
        self.dcands = []                       # Candy: democratic candidate in each district
        self.rcands = []                       # Candy: republican candidate in each district
        self.thdpty = []                        # Candy: list of other candidates
        self.status = []                       # uninit=-1, need to impute=0, has been imputed=1, okay from start=2
        self.demfrac = []                      # dem fraction of vote (only valid if status >= 1)
        self.pvote = []                        # dem frac of presidential vote
        self.impute_data = []                  # for storing info from multilinear model
        self.unopposed = 0                     # fraction of races that are unopposed
        self.egap = float('NaN')               # efficiency gap
        self.numpow = [-2,-1,-0.5,0,0.50,1,2]       # powers to use for "wasted" votes; negative get interpreted as on 01.23.17
        self.gaps = [float('NaN') for j in range(len(self.numpow))]
        self.adjgaps = [float('NaN') for j in range(len(self.numpow))]
        self.adj = [0 for j in range(len(self.numpow))] # how to adjust for bias from overall vote percentage
        self.l_regress = [float('NaN'),float('NaN')] # intercept for regression line for losing races
        self.w_regress = [float('NaN'),float('NaN')] # intercept for regression line for losing races

    def mycopy(self):
        """
        """
        g = Gerry(self.yr,self.state,self.chamber)
        # REAPP
        g.cyc_state = self.cyc_state
        g.Ndists = self.Ndists
        g.Ddists = self.Ddists
        g.dists = [x for x in self.dists]
        for i in range(self.Ndists):
            if self.dcands[i] != None:
                g.dcands.append(self.dcands[i].mycopy())
            else:
                g.dcands.append(None)
            if self.rcands[i] != None:
                g.rcands.append(self.rcands[i].mycopy())
            else:
                g.rcands.append(None)
            if self.thdpty[i] != None:
                g.thdpty.append([])
                for j in range(len(self.thdpty[i])):
                    g.thdpty[i].append(self.thdpty[i][j].mycopy())
            else:
                g.thdpty.append(None)    
        g.unopposed = self.unopposed
        g.egap = self.egap
        g.status = [self.status[j] for j in range(len(self.status))]
        g.impute_data = [self.impute_data[j] for j in range(len(self.status))]
        g.demfrac = [self.demfrac[j] for j in range(len(self.demfrac))]
        g.pvote = [self.pvote[j] for j in range(len(self.pvote))]
        g.numpow = [self.numpow[j] for j in range(len(self.numpow))]
        g.gaps = [self.gaps[j] for j in range(len(self.gaps))]
        g.adjgaps = [self.adjgaps[j] for j in range(len(self.adjgaps))]
        g.adj = [self.adj[j] for j in range(len(self.adj))]
        g.l_regress = [self.l_regress[j] for j in range(len(self.l_regress))]
        g.w_regress = [self.w_regress[j] for j in range(len(self.w_regress))]
        return g
            
    def myprint(self):
        """
        """
        print self.yr,self.state,self.cyc_state,self.chamber,"demfrac: ",self.demfrac
        for i in range(self.Ndists):
            # first print out dem winners
            if self.dcands[i] != None:
                dstr = str(self.dcands[i])
            else:
                dstr = 'None'
            if self.rcands[i] != None:
                rstr = str(self.rcands[i])
            else:
                rstr = 'None'
            print "   %s: %s vs. %s %s " % (self.dists[i],dstr,rstr,self.status[i])
        print "egap: %.4f" % (self.egap)

    def regularize_races(self):
        """ add stub candidates if necessary; flag if need to impute votes
            make sure winners votes are greater than loser's (even if needs to be imputed eventually)
            store demfrac in appropriate place

        # Reviewed 11.15.17
        # Edited 12.11.17 to deal with third-party only situations
        """          
        todel = []
        for i in range(self.Ndists):
            dc = self.dcands[i]
            rc = self.rcands[i]
            if dc is None and rc is None:
                # only third-party candidates - delete district and everything associated to it
                # REAPP
                todel.append(i)
                # print "WARNING: weird, this district doesn't exist",self.yr,self.cyc_state,self.chamber,i
                print "Deleting %s in %s %s %s since no Rep or Dem" % (self.dists[i],self.yr,self.state,self.chamber)
                # return # GSW: Remove! Above shouldn't happen....
                continue
            if rc is None: # unopposed democrat
                self.status[i] = 0 # needs to be imputed
                self.dcands[i].winner = True
                if self.dcands[i] in [0,'',' ']:
                    print "Dem. vote not valid in uncontested race"
                # 12.11.17 - don't think this is needed anywhere
                # self.dcands[i].votes = 1000
                self.rcands[i] = Candy(self.dcands[i].race,'','Rep',0,False,False) # votes, incum, winner
            elif dc is None: # unopposed republican
                self.status[i] = 0 # needs to be imputed
                self.rcands[i].winner = True
                if self.rcands[i] in [0,'',' ']:
                    print "Rep. vote not valid in uncontested race"
                # 12.11.17 - don't think this is needed anywhere
                # self.rcands[i].votes = 1000
                self.dcands[i] = Candy(self.rcands[i].race,'','Dem',0,False,False) # votes, incum, winner
            elif dc.votes == 0 and rc.votes == 0: # two candidates, but neither has votes listed; shouldn't happen
                if (dc.winner and rc.winner) or (not dc.winner and not rc.winner):
                    print "WARNING: both/no winners", self.yr,self.cyc_state,self.chamber,self.dists[i], str(dc), str(rc)
                    return # GSW: Remove? Shouldn't happen
                elif dc.winner:
                    self.status[i] = 0 # needs to be imputed
                else:
                    self.status[i] = 0 # needs to be imputed
            elif dc.votes == 0 or rc.votes == 0: # not going to happen in a real race
                self.status[i] = 0 # needs to be imputed
            else:
                self.demfrac[i] = dc.votes*1.0/(dc.votes+rc.votes)
                self.status[i] = 2 # okay from start
        # delete ones we don't want
        tokeep = filter(lambda x: x not in todel, range(self.Ndists))
        self.dists = [self.dists[i] for i in tokeep]
        self.dcands = [self.dcands[i] for i in tokeep]
        self.rcands = [self.rcands[i] for i in tokeep]
        self.thdpty = [self.thdpty[i] for i in tokeep]
        self.status = [self.status[i] for i in tokeep]
        self.demfrac = [self.demfrac[i] for i in tokeep]
        self.pvote = [self.pvote[i] for i in tokeep]
        self.impute_data = [self.impute_data[i] for i in tokeep]
        self.Ndists -= len(todel)

###############################################################################################################
def get_cyc_state(yrint,state,chamber):
    """ Get alternative name for state to keep track of mid-decade redistricting

    # Raeviewed 11.15.17
    TODO: Write auxiliary function and compress the code so it's easier to read
    """
    if chamber == '11':
        ############## Congress 00s                                                         
        # A few districts were changed in ~2004 as well. Split?                             
        if state == 'FL' and 2020 >= yrint >= 2012:                                         
            if yrint <= 2014:                                                               
                return 'FL1'                                                                
            else:                                                                           
                return 'FL2'                                                                
                                                                                            
        ############## Congress 00s                                                         
        # A few districts were changed in ~2004 as well. Split?                             
        if state == 'TX' and 2010 >= yrint >= 2002:                                         
            if yrint == 2002:                                                               
                return 'TX1'                                                                
            else:                                                                           
                return 'TX2'                                                                
                                                                                            
        ############## Congress 90s                                                         
        # https://www.senate.mn/departments/scr/REDIST/Redsum/txsum.htm
        # http://www.tlc.state.tx.us/redist/history/maps_congress.html
        if state == 'TX' and 2000 >= yrint >= 1992: # Bush vs. Verta                        
            if 1994 >= yrint >= 1992:                                                       
                return 'TX1'                                                                
            else:                                                                           
                return 'TX2'                                                                
        if state == 'LA' and 2000 >= yrint >= 1992:                                         
            if 1992 == yrint:                                                               
                return 'LA1'                                                                
            elif 1994 == yrint:                                                             
                return 'LA2'                                                                
            else:                                                                           
                return 'LA3'                                                                
        if state == 'GA' and 2000 >= yrint >= 1992:                                         
            if 1994 >= yrint >= 1992:                                                       
                return 'GA1'                                                                
            else:                                                                           
                return 'GA2'                                                                
                                                                                            
        ############## Congress 80s                                                         
        if state == 'TX' and 1990 >= yrint >= 1982:                                         
            if 1982 == yrint:                                                               
                return 'TX1'                                                                
            else:                                                                           
                return 'TX2'                                                                
        if state == 'NJ' and 1990 >= yrint >= 1982:                                         
            if 1982 == yrint:                                                               
                return 'NJ1'                                                                
            else:                                                                           
                return 'NJ2'                                                                
        if state == 'CA' and 1990 >= yrint >= 1982:                                         
            if 1982 == yrint:                                                               
                return 'CA1'                                                                
            else:                                                                           
                return 'CA2'                                                                
        if state == 'LA' and 1990 >= yrint >= 1982:                                         
            if 1982 == yrint:                                                               
                return 'LA1'                                                                
            else:                                                                           
                return 'LA2'                                                                
                                                                                            
        ############## Congress 70s                                                         
        if state == 'CA' and 1980 >= yrint >= 1972:                                         
            if 1972 == yrint:                                                               
                return 'CA1'                                                                
            else:                                                                           
                return 'CA2'
    else:
        # some states hold elections in odd years
        ############## State 00s
        if state == 'GA' and 2010 >= yrint >= 2001:                                         
            if 2002 == yrint:                                                               
                return 'GA1'                                                                
            else:                                                                           
                return 'GA2'
        if state == 'SC' and 2010 >= yrint >= 2001:                                         
            if 2002 == yrint:                                                               
                return 'SC1'                                                                
            else:                                                                           
                return 'SC2'

        ############## State 90s - TN, KY, SC, OH
        if state == 'SC' and 2000 >= yrint >= 1991:  # M-S has three different, couldn't find evidence
            if 1996 >= yrint:                                                               
                return 'SC1'                                                                
            else:                                                                           
                return 'SC2'
        # http://www.ncsl.org/research/redistricting/1990s-redistricting-case-summaries.aspx#TN        
        # https://www.senate.mn/departments/scr/REDIST/Redsum/kysum.htm#847%20S.W.2d%20718
        if state == 'KY' and 2000 >= yrint >= 1991:  
            if 1994 >= yrint:                                                               
                return 'KY1'                                                                
            else:                                                                           
                return 'KY2'

            # In April of 1992, the Tennessee General Assembly enacted
            # legislation reapportioning the State's single-member
            # House of Representatives and Senate
            # districts. Tenn. Code Ann. SS 3-1-102 and 103 (1992)
            # (repealed 1994). 
            # http://caselaw.findlaw.com/us-6th-circuit/1074038.html
            # this describes 13.9% problem in 1992
            # https://www.comptroller.tn.gov/lg/pdf/redist.pdf
        if state == 'TN' and 2000 >= yrint >= 1991:  # still should be checked
           if 1992 == yrint:                                                               
               return 'TN1'                                                                
           else:                                                                           
               return 'TN2'
        # CHECK! Having trouble finding definitive evidence
        if state == 'OH' and 2000 >= yrint >= 1991:  
           if 1994 >= yrint:                                                               
               return 'OH1'                                                                
           else:                                                                           
               return 'OH2'

        ############## State 80s - AL, AK, VA, NM, TN, WI, CA, ID
        # only one I have listed is 1986 - presumably because of mmd - need to look into
        # if state == 'AL' and 1990 >= yrint >= 1982:  
        #   if 1982 == yrint:                                                               
        #       return 'AL1'                                                                
        #   else:                                                                           
        #       return 'AL2'
        #
        # AK - has multi-member districts each year
        # VA - has odd years starting in 83 listed
        # 
        # https://www.nmlegis.gov/Redistricting/Documents/187014.pdf        
        if state == 'NM' and 1990 >= yrint >= 1981:  
          if 1982 == yrint:                                                               
              return 'NM1'                                                                
          else:                                                                           
              return 'NM2'
        # "Elections, partisanship and the institutionalization...."
        # also notes reapportionment in 1973
        if state == 'TN' and 1990 >= yrint >= 1981:  
          if 1982 == yrint:                                                               
              return 'TN1'                                                                
          else:                                                                           
              return 'TN2'
        # https://comm.ncsl.org/productfiles/83453486/Redistrciting_in_Wisconsin_2016.pdf
        if state == 'WI' and 1990 >= yrint >= 1981:  
          if 1982 == yrint:                                                               
              return 'WI1'                                                                
          else:                                                                           
              return 'WI2'
        # "redistricting in california: competitive elections - reason.org
        # "redistricting california 1986"
        if state == 'CA' and 1990 >= yrint >= 1981:  
          if 1982 == yrint:                                                               
              return 'CA1'                                                                
          else:                                                                           
              return 'CA2'
        # Idaho - no elections listed

        ############## State 70s - CA
        # www.stat.columbi.edu/~gelman/stuff_for_blog/piero.pdf
        if state == 'CA' and 1980 >= yrint >= 1971:  
          if 1972 == yrint:                                                               
              return 'CA1'                                                                
          else:                                                                           
              return 'CA2'

    # one plan for entire decade, so do not make any changes.
    return state
           
###############################################################################################################
def make_records(arr,mmd,elections=None,verbose=False):
    """ make a record for each year-state-chamber

    # Reviewed 11.15.17
    This takes raw data from files that is listed in arrays and populates various classes.

    Format of arr is something like the following:
      myid = '1974_AL_9' - year,state and legislative body
      newdistid = '61.' - district identifier - TODO: describe precisely
      votes = candidate's vote totals. 0 if entry in file is empty
      party = 'Dem' or 'Rep'
      incumbent = True or False
      winner = True or False
      pvote = presidential vote as a percentage or '' if unknown
    """
    tot_state = 0
    tot_cong = 0
    cong_yr_lims = [2100,1900]
    state_yr_lims = [2100,1900]

    clobber = False
    if elections == None:
        elections = dict()
    else:
        # multiple elements of arr may refer to the same key, so we need to clobber first
        # before we start adding anything back in
        for x in arr:
            # so we can completely rewrite data if we're worried about it being in there
            if x[0] in elections.keys():
                # print "Clobbering",x[0]
                del elections[x[0]]

    for x in arr:
        if x[0] not in elections.keys(): 
            yr,state,chamber = x[0].split('_')
            # don't add any elections with MMDs.
            if chamber == '9' and yr in mmd.keys() and state in mmd[yr]:
                continue
            elections[x[0]] = Gerry(yr,state,chamber)

            if chamber == '11':
                tot_cong += 1
                if int(yr) < cong_yr_lims[0]:
                    cong_yr_lims[0] = int(yr)
                if int(yr) > cong_yr_lims[1]:
                    cong_yr_lims[1] = int(yr)
            if chamber == '9':
                tot_state += 1
                if int(yr) < state_yr_lims[0]:
                    state_yr_lims[0] = int(yr)
                if int(yr) > state_yr_lims[1]:
                    state_yr_lims[1] = int(yr)

            # TODO: Make REAPP tags understandable (or remove)
            # REAPP - references code that was added to deal with mid-decade reapportionment
            # cyc_state is finer than state since it takes into account which plan during the decade
            elections[x[0]].cyc_state = get_cyc_state(int(yr),state,chamber)
            if verbose:
                print "Adding new race: %s" % (x[0])
        elec = elections[x[0]]
        # REAPP
        # See if we've already added a candidate for this race. Get index if so, otherwise add at end
        if (elec.cyc_state + x[1]) in elec.dists:
            # REAPP
            i = elec.dists.index(elec.cyc_state + x[1])
        else:
            i = elec.Ndists
            # REAPP
            elec.dists.append(elec.cyc_state + x[1])
            elec.dcands.append(None)
            elec.rcands.append(None)
            elec.thdpty.append(None)
            elec.status.append(-1)
            elec.impute_data.append(['','','','','','']) # stored with election so easier to print and save
            elec.demfrac.append(0.0)
            # added 07.27.17 - WARNING
            # this won't work for data in which we get candidates separately, but
            # that's not the case for the congressional data, which is the only stuff
            # we have presidential vote for at the moment
            # TODO: Not sure exactly what this comment means, all races have been split into
            # two lines in arr....
            elec.pvote.append(x[-1])
            elec.Ndists += 1

        if x[3] == 'Dem':
            elec.dcands[i] = Candy(x[0],'NoName','Dem',x[2],x[4],x[5]) # race,name,party,votes,incum,winner
        elif x[3] == 'Rep':
            elec.rcands[i] = Candy(x[0],'NoName','Rep',x[2],x[4],x[5]) # race,name,party,votes,incum,winner
        else:
            if elec.thdpty[i] == None:
                elec.thdpty[i] = []
            elec.thdpty[i].append(Candy(x[0],'NoName',x[3],x[2],x[4],x[5])) # race,name,party,votes,incum,winner
    for elec in elections.values():
        elec.regularize_races()
            
    correct_errors(elections)

    return elections,tot_cong,tot_state,cong_yr_lims,state_yr_lims

def print_race(elections,yr,state,chamber):
    """ Print a single race if it ex
    # reviewed 11.30.17
    # TODO: should probably be called "print_election"
    """
    race = '_'.join([yr,state,chamber])
    if race not in elections.keys():
        return
        
    elections[race].myprint()
    
def print_all_races(elections,verbose=True):
    """ Print all of the races
    # reviewed 11.30.17
    # TODO: should probably be called "print_all_elections"
    """
    yrs = []
    states = []
    chambers = []
    for k in elections.keys():
        if elections[k].yr not in yrs:
            yrs.append(elections[k].yr)
        if elections[k].state not in states:
            states.append(elections[k].state)
        if elections[k].chamber not in chambers:
            chambers.append(elections[k].chamber)
    yrs = sorted(yrs)
    states = sorted(states)
    chambers = sorted(chambers)
    for yr in yrs:
        for state in states:
            for chamber in chambers:
                print yr,state,chamber
                if verbose:
                    print_race(elections,yr,state,chamber)

def create_cycles(elections,prior_all,recent_cong,recent_state):
    """ group together all elections within a given decade.

    Uses cycstate to effectively treat different plans within the same state as completely distinct.
    # Reviewed 12.01.17
    """
    cycles = []
    # state and congress
    if prior_all:
        for yrmin in ['1971','1981','1991','2001']:
            for chamber in ['9','11']:
                newcycle = Cycle(yrmin,str(int(yrmin)+9),chamber)
                newcycle.populate_elections(elections)
                # print "bbb: ",newcycle.min_year,newcycle.max_year,len(newcycle.elecs)
                newcycle.fit_model()
                newcycle.impute_votes(True)
                cycles.append(newcycle)

    # recent congress
    if recent_cong:
        for yrmin in ['2011']: 
            for chamber in ['11']:
                newcycle = Cycle(yrmin,str(int(yrmin)+5),chamber)
                newcycle.populate_elections(elections)
                # print "bbb: ",newcycle.min_year,newcycle.max_year,len(newcycle.elecs)
                newcycle.fit_model()
                newcycle.impute_votes(True)
                cycles.append(newcycle)

    # recent state - hard to interpolate, so not clear how useful this is....
    if recent_state:
        for yrmin in ['2011']: 
            for chamber in ['9']:
                newcycle = Cycle(yrmin,str(int(yrmin)+5),chamber)
                newcycle.populate_elections(elections)
                # print "bbb: ",newcycle.min_year,newcycle.max_year,len(newcycle.elecs)
                newcycle.fit_model()
                newcycle.impute_votes(True)
                cycles.append(newcycle)

    return cycles
                       

