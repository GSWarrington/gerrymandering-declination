def print_pi(elecs,myid):
    """
    """
    elec = elecs[myid]
    for i in range(elec.Ndists):
        print elec.demfrac[i]


        
def get_EG_direct(vals):
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

    # arr2,yrs2,states2,mmd2 = read_tsv_2010('SLERs2011to2012_only_2013_05_14.csv') # 2012 state races
    # arr6,yrs6,states6 = read_fec_csv() # 2000 and later congressional - obsoleted by jacobson

def read_fec_csv():
    """
    """
    statecodemap = [[1,'CT'],[2,'ME'],[3,'MA'],[4,'NH'],[5,'RI'],[6,'VT'],[11,'DE'],[12,'NJ'],[13,'NY'],[14,'PA'],\
    [21,'IL'],[22,'IN'],[23,'MI'],[24,'OH'],[25,'WI'],[31,'IA'],[32,'KS'],[33,'MN'],[34,'MO'],[35,'NE'],[36,'ND'],\
    [37,'SD'],[41,'AL'],[42,'AR'],[43,'FL'],[44,'GA'],[45,'LA'],[46,'MS'],[47,'NC'],[48,'SC'],[49,'TX'],[40,'VA'],\
    [51,'KY'],[52,'MD'],[53,'OK'],[54,'TN'],[55,'DC'],[56,'WV'],[61,'AZ'],[62,'CO'],[63,'ID'],[64,'MT'],[65,'NV'],\
    [66,'NM'],[67,'UT'],[68,'WY'],[71,'CA'],[72,'OR'],[73,'WA'],[81,'AK'],[82,'HI']]
    
    stateabbrev = [x[1] for x in statecodemap]
      
    fecyrs = [2000,2002,2004,2006,2008,2010,2012,2014]
    
    HdrSt = ['STATE','STATE','STATE ABBREVIATION','STATE ABBREVIATION','STATE ABBREVIATION','STATE ABBREVIATION','STATE ABBREVIATION','STATE ABBREVIATION']
    HdrDist = ['DISTRICT','DISTRICT','DISTRICT','DISTRICT','DISTRICT','DISTRICT','D','D']
    HdrParty = ['PARTY','PARTY','PARTY','PARTY','PARTY','PARTY','PARTY','PARTY']
    HdrVotes = ['GENERAL RESULTS','GENERAL RESULTS','GENERAL','GENERAL','GENERAL ','GENERAL ','GENERAL VOTES ','GENERAL VOTES ']
    HdrIncum = ['INCUMBENT INDICATOR','INCUMBENT INDICATOR','INCUMBENT INDICATOR','INCUMBENT INDICATOR','INCUMBENT INDICATOR (I)', 'INCUMBENT INDICATOR (I)','(I)','(I)']
    
    PartyVals = [['R','D'],['R','D'],['R','D'],['REP','DEM'],['R','D'],['REP','DEM'],['R','D'],['R','D']]
     
    fecfiles = ['2000congresults.csv','2002congresults.csv','2004congresults.csv','2006congresults.csv',\
        '2008congresults.csv','2010congresults.csv','2012congresults.csv','2014congresults.csv'] 
      
    d = dict()
    for i in range(len(fecfiles)):
        fn = fecfiles[i]
        yr = fecyrs[i]
        f = open('/home/gswarrin/research/gerrymander/data/' + fn,'r')
        # print "starting year: ",yr
        cnt = 0
        for line in f:
            l = line.split('\t')
            if cnt == 0:
                hdrs = [x for x in l]
                cnt += 1
                # print hdrs
                continue
            if len(l) <= hdrs.index(HdrVotes[i]):
                continue
            state = l[hdrs.index(HdrSt[i])]
            if state in ['CT','NY'] or state not in stateabbrev: # dealing with combined parties is a pain
                continue
            # print "                    ",state
            dist = l[hdrs.index(HdrDist[i])]
            voteval = l[hdrs.index(HdrVotes[i])].replace(',', '')
            if voteval == '' or voteval == ' ' or voteval == 'n/a' or voteval == '#':
                continue
            if voteval[0] == '[':
                voteval = voteval[1:-1]
            if voteval[0] in ['U','W']: # for unopposed/withdrew, some have spaces at end...
                votes = 1000
            else:
                votes = int(voteval)
            party = l[hdrs.index(HdrParty[i])].rstrip()
            if l[hdrs.index(HdrIncum[i])] == '(I)':
                incum = True
            else:
                if len(l[hdrs.index(HdrIncum[i])]) > 1:
                    print "incum: ",l[hdrs.index(HdrIncum[i])]
                incum = False
            key = '_'.join([repr(yr),state,'11',dist])
            cnt += 1
            if votes > 0 and dist.isdigit() and party in PartyVals[i]:
                if key in d.keys():
                    if party == PartyVals[i][1]:
                        d[key][0] += votes
                    else:
                        d[key][1] += votes
                else:
                    if party == PartyVals[i][1]:
                        d[key] = [votes,0]
                    else:
                        d[key] = [0,votes]
        f.close()
        # print "Done with year: ",yr,cnt
    ncnt = 0
    ans = []
    # incum = 0 # information on whether an incumbent
    # do this extra step since otherwise I don't know who won or not.
    # seems like I could do this later when things get pushed into records, though.
    for k in d.keys():
        tyr,tstate,tchamber,tdist = k.split('_')
        if d[k][0] > d[k][1]:
            dwinner = True
            rwinner = False
        else:
            dwinner = False
            rwinner = True
        # tmpd = ['fec' + repr(ncnt),'_'.join([tyr,tstate,tchamber]),tdist,d[k][0],'100',incum,dwinner]
        # tmpr = ['fec' + repr(ncnt),'_'.join([tyr,tstate,tchamber]),tdist,d[k][1],'200',incum,rwinner]
        tmpd = ['_'.join([tyr,tstate,tchamber]),tdist,d[k][0],'Dem',incum,dwinner]
        tmpr = ['_'.join([tyr,tstate,tchamber]),tdist,d[k][1],'Rep',incum,rwinner]
        ans.append(tmpd)
        ans.append(tmpr)
    return ans,[repr(x) for x in fecyrs],stateabbrev # should already have all of these states
    
def compute_adj(cycles):
    """ compute the adjustments due to bias for different weight functions
    """
    # restrict to cycles since others may not have had votes imputed
    adjlist = [[] for j in range(100)] # just pick something big for now
    numgaps = 0
    for c in cycles:
        for k in c.elecs.keys():
            if c.elecs[k].Ndists >= 1:
                numgaps = len(c.elecs[k].numpow)
                for j in range(numgaps):
                    adjlist[j].append([c.elecs[k].Dfrac,c.elecs[k].gaps[j]])

    # now compute slope of regressions
    ans = []
    for j in range(numgaps):
        slope, intercept, r_value, p_value, std_err = stats.linregress([x[0] for x in adjlist[j]], [x[1] for x in adjlist[j]])
        ans.append(slope)
        print "j: ",slope

    # now fill in
    for elec in elections.values():
        elec.adj = [x for x in ans]
        for j in range(len(elec.numpow)):
            elec.adjgaps[j] = elec.gaps[j] - elec.adj[j]*(elec.Dfrac-0.5)
        idx = elec.numpow.index(0)
        elec.egap = elec.gaps[idx] - elec.adj[idx]*(elec.Dfrac-0.5)
        idx = elec.numpow.index(-2)

    def compute_overall(self):
        """ estimate democrats statewide fraction of the vote
        """ 
        doverall = 0
        roverall = 0
        for i in range(self.Ndists):
            actual_votes,dv,rv = self.compute_district_vote(i)
            doverall += dv
            roverall += rv 
        if (doverall + roverall) > 0:
            self.Dfrac = 1.0*doverall/(doverall + roverall)
        else:
            self.valid = False

    # was in init_all
    for c in loc_prior_cycles:
        for k in c.elecs.keys():
            c.elecs[k].compute_overall()       

    for c in loc_recent_cycles:
        for k in c.elecs.keys():
            c.elecs[k].compute_overall()       

    def compute_gaps(self,default_win=15000, default_lose=6000, alpha = 2):
        """ compute e-gap and f-gap
        """
        numarr = [0 for j in range(len(self.numpow))]
        totvotes = 0
        self.Ddists = 0
        # compute efficiency gap
        for i in range(self.Ndists):
            actual_votes,dv,rv = self.compute_district_vote(i)
            if actual_votes:
                self.dcands[i].frac = 1.0*dv/(dv+rv)
                self.rcands[i].frac = 1.0*rv/(dv+rv)
            # else:
            #     print "this should not happen...",self.yr,self.state,i,str(self.dcands[i]),str(self.rcands[i])
            if dv >= rv:
                self.Ddists += 1        
            totvotes = dv + rv

            if totvotes > 0:
                ai = (2.0*(dv*1.0-totvotes*1.0/2))/totvotes
                # print i,ai,totvotes
                for j in range(len(self.numpow)):
                    if self.numpow[j] >= 0: # votes close to 50% are weighed more ("traditional")
                        if ai >= 0:  
                            numarr[j] += pow(ai,self.numpow[j]+1)
                            # print "a",j,numarr[j]
                        else:
                            numarr[j] -= pow(-ai,self.numpow[j]+1)
                            # print "b",j,numarr[j]
                    else:             # votes close to 50% are weighed less 
                        if ai >= 0:
                            numarr[j] -= pow((1-ai),-self.numpow[j]+1)
                            # print "c",j,numarr[j]
                        else:
                            numarr[j] += pow((1+ai),-self.numpow[j]+1)
                            # print "d",j,numarr[j]
                    
        for j in range(len(self.numpow)):
            if self.numpow[j] >= 0:
                self.gaps[j] = (2.0/(self.numpow[j]+2))*(numarr[j]/self.Ndists + 0.5 - self.Ddists*1.0/self.Ndists)
            else:
                self.gaps[j] = (2.0/(-self.numpow[j]+2))*(numarr[j]/self.Ndists - 0.5 + self.Ddists*1.0/self.Ndists)
        # don't do here since we don't know biases yet
        # self.egap = self.gaps[self.numpow.index(0)]

def compute_district_vote(self, i, default_win=15000, default_lose=6000):
    """ put all case here; first return is whether actual values or estimates; i is district number
    """          
    dc = self.dcands[i]
    rc = self.rcands[i]
    if dc is None and rc is None:
        print "weird, this district doesn't exist",i
        return False,0,0
    if rc is None: # unopposed democrat
        return False,default_win,default_lose
    elif dc is None: # unopposed republican
        return False,default_lose,default_win
    elif dc.votes == 0 and rc.votes == 0: # two candidates, but neither has votes listed; shouldn't happen
        if (dc.winner and rc.winner) or (not dc.winner and not rc.winner):
            print "both/no winners", self.yr,self.cyc_state,self.chamber,self.dists[i], str(dc), str(rc)
            return False,0,0
        elif dc.winner:
            return False,default_win,default_lose
        else:
            return False,default_lose,default_win
    return True,dc.votes,rc.votes

