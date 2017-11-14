# 1 - functions for reading data

def read_tsv_2010(fn):
    """
    07.27.17 - added empty string at end since don't know presidential data (yet?)
    """
    f = open('/home/gswarrin/research/gerrymander/data/' + fn,'r')
    cnt = 0
    scnt = 0
    ans = []
    yrs = []
    mmd = [] # for keeping track of who has multi-member districts
    states = []
    myids = []
    for line in f:
        cnt += 1
        if cnt == 1:
            hdrs = line.strip().split('\t')
            continue
        l = line.strip().split('\t')
        
        ############################################################
        # restriction attention to certain lines
        ############################################################
        # TODO: skip senate? (what about nebraska?) - only work with lower house as that's what S-M do
        if l[hdrs.index('v07')] != '9':
            continue
        # v12 - district type (want SMD only, which is coded by a 1)
        if l[hdrs.index('v12')] != '1':
            tmp = [l[hdrs.index('v05')],l[hdrs.index('v02')]]
            if tmp not in mmd:
                mmd.append(tmp)
            continue
        # v16 - election type (want General, coded by a G)
        if l[hdrs.index('v16')] != 'G':
            continue
        # v21 - 100 = Democrat, 200 = Republican, see Study #8907 for detailed listing
        if l[hdrs.index('v21')] not in ['100','200']:
            continue
        # The 2012 data doesn't contain all columns
        # v26 - number of democrats running
        # v27 - number of republicans running
        # v13 - number of winners
        if ('v26' in hdrs and int(l[hdrs.index('v26')]) > 1) or \
           ('v27' in hdrs and int(l[hdrs.index('v27')]) > 1) or \
           ('v13' in hdrs and l[hdrs.index('v13')] != '1'):
            continue

        ############################################################
        # create a new field that identifies each race we care about
        ############################################################
        if l[hdrs.index('v05')] not in yrs:
            yrs.append(l[hdrs.index('v05')])
        if l[hdrs.index('v02')] not in states:
            states.append(l[hdrs.index('v02')])
            
        myid = '_'.join([l[hdrs.index('v05')],l[hdrs.index('v02')], \
                l[hdrs.index('v07')]])
        if myid not in myids:
            myids.append(myid)
        if l[hdrs.index('v23')] == ' ' or l[hdrs.index('v23')] == '': # candidates vote total
            l[hdrs.index('v23')] = 0
        if l[hdrs.index('v21')] == '100':
            party = 'Dem'
        else:
            party = 'Rep'
        # print ".%s." % (l[hdrs.index('v13')])
        # v11 - district identifier
        # v23 - candidate's vote total
        # v29 - total votes cast in election (all candidates)
        # v21 - simplified party code
        # v24 - election winner
        if 'v22' in hdrs and l[hdrs.index('v22')] == '1':
            incum = True
        else:
            incum = False
        if 'v24' in hdrs and l[hdrs.index('v24')] == '1':
            winner = True
        else:
            winner = False
            
        # incum = l[hdrs.index('v22')] # information on whether an incumbent
        if 'v11' in hdrs:
            newdistid = l[hdrs.index('v11')]
        else:
            newdistid = l[hdrs.index('v08')]+l[hdrs.index('v09')]+l[hdrs.index('v10a')]+\
                        l[hdrs.index('v10b')]+l[hdrs.index('v10c')]
        # str = [l[hdrs.index('caseid')],myid,newdistid,\
        #         int(l[hdrs.index('v23')]),l[hdrs.index('v21')],incum,winner]
        str = [myid,newdistid,\
                int(l[hdrs.index('v23')]),party,incum,winner,'']
        ans.append(str)      
        scnt += 1
        # print str
    return ans,yrs,states,mmd

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

def read_jacobson_csv():
    """ for reading in congressional races; 
        07.27.17 - added pres votes at end as either empty string or fraction of 1.
    """
    statelist = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
      
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    cnt = 0
    scnt = 0
    ans = []
    yrs = []
    states = []
    myids = []
    for line in f:
        cnt += 1
        if cnt == 1:
            hdrs = line.strip().split(',')
            continue
        l = line.strip().split(',')
        
        ############################################################
        # create a new field that identifies each race we care about
        ############################################################
        # 0 is year
        # 1 is state and district: 1..50 and 01..n
        # 2 0=Rep incum; 1=Dem incum; 5=both incum (opp parties); 6=Two Dem incum; 7=Two Rep incum
        # 3 1=Dem won; 0=Rep won; 9=Third party/indep won
        # 4 Dem's share of two-party vote
        # 8 unopposed
        # 9 1=District redrawn since last election; 0=not redrawn
        # 12 major party presidential vote

        if l[0] not in yrs:
            yrs.append(l[0])
        stcode = int(l[1][:-2])
        # print stcode
        distcode = l[1][-2:]
        state = statelist[stcode-1]
        if state not in states:
            states.append(state)
        uncontested = False    
        myid = '_'.join([l[0],state,'11']) # '11' means US Congress
        if myid not in myids:
            myids.append(myid)
        if l[4] == ' ' or l[4] == '': # democratic candidate's vote total
            dvotes = 0
            uncontested = True
        else:
            if '.' in l[4]:
                dvotes = int(float(l[4])*10)
            else:
                dvotes = int(l[4])*10
        if l[12] == ' ' or l[12] == '':
            pvotes = ''
        else:
            pvotes = float(l[12])/100

        if dvotes > 1000:
            print "reading jacobson %s %s %s - too many votes!" % (l[0],state,l[1])
        repinc = (l[2] in ['0','5'])
        deminc = (l[2] in ['1','5'])
        dwinner = (l[3] == '1')
        rwinner = (l[3] == '0')
        if not dwinner and not rwinner: # third-party won - ignore it (completely? try to impute?)
            print "Ignoring race completely",myid,state,l[1]
            continue
        redrawn = (l[9] == '1')
        if not uncontested or dwinner:
            ans.append([myid,state+l[1],dvotes,'Dem',deminc,dwinner,pvotes])
        if not uncontested or rwinner:
            ans.append([myid,state+l[1],1000-dvotes,'Rep',repinc,rwinner,pvotes])
        if uncontested:
            print "Uncontested: ",myid,state+l[1],dvotes,dwinner,rwinner
        if (int(l[0]) % 2) == 1:
            print "Odd year: ",myid,state+l[1],dvotes,dwinner,rwinner
        # print(ans[-2])
        # print(ans[-1])    
        scnt += 1
    return ans,yrs,states # ,mmd

def read_2012_state_csv(fn,chamber):
    """ for reading in state races I typed in from Ballotpedia
    07.27.17 - added empty string at end since don't know presidential vote currently
    """
    stlist = ['00','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',
              'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',
              'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'] 

    f = open(fn,'r')
    cnt = 0
    scnt = 0
    ans = []
    yrs = []
    states = []
    myids = []
    for line in f:
        cnt += 1
        if cnt == 1:
            hdrs = line.strip().split(',')
            continue
        l = line.strip().split(',')
        
        ############################################################
        # create a new field that identifies each race we care about
        ############################################################
        # 0 is year
        # 1 is state 
        # 2 is district
        # 3 is D vote
        # 4 is R vote
        # 5 is D Inc
        # 6 is R Inc
        
        if l[0] not in yrs:
            yrs.append(l[0])
        state = l[1]
        if chamber == '9':
            distcode = state + l[2]
        else:
            tmp = stlist.index(state)
            if tmp < 10:
                tmp = '0' + str(tmp)
            else:
                tmp = str(tmp)
            distcode = state + tmp + l[2]
        if state not in states:
            states.append(state)
        uncontested = False    
        myid = '_'.join([l[0],state,chamber]) # '9' means state legislature
        if myid not in myids:
            myids.append(myid)
        if l[3] == ' ' or l[3] == '': # democratic candidate's vote total
            l[3] = '0'
            uncontested = True
        if l[3] == '0' or l[4] == '0':
            uncontested = True
        deminc = (l[5] == '1')
        repinc = (l[6] == '1')
        dwinner = (int(l[3]) > int(l[4]))
        rwinner = (int(l[4]) > int(l[3]))
        if not dwinner and not rwinner: # third-party won - ignore it (completely? try to impute?)
            print "Skipping race %s %s %s because no winner" % (l[0],state,distcode)
            continue
        if not uncontested or dwinner:
            ans.append([myid,distcode,int(l[3]),'Dem',deminc,dwinner,''])
        if not uncontested or rwinner:
            ans.append([myid,distcode,int(l[4]),'Rep',repinc,rwinner,''])
        if uncontested:
            print "Uncontested: ",myid,distcode,l[3],dwinner,rwinner
        # print(ans[-2])
        # print(ans[-1])    
        scnt += 1
    return ans,yrs,states # ,mmd

def read_all_data():
    """ read in all data files
    """
    arr = []
    yrs = []
    states = []
    arr,yrs,states,mmd = read_tsv_2010('34297-0001-Data.tsv') # state races up to 2010
    # arr2,yrs2,states2,mmd2 = read_tsv_2010('SLERs2011to2012_only_2013_05_14.csv') # 2012 state races
    # arr6,yrs6,states6 = read_fec_csv() # 2000 and later congressional - obsoleted by jacobson
    arr6,yrs6,states6 = read_jacobson_csv()
    arr7,yrs7,states7 = read_2012_state_csv('/home/gswarrin/research/gerrymander/data/stateleg2012plus.csv','9')
    arr8,yrs8,states8 = read_2012_state_csv('/home/gswarrin/research/gerrymander/data/cong2016.csv','11')

    # mmd += mmd2
    # arr += arr2
    arr += arr6
    arr += arr7
    arr += arr8
    yrs = sorted(set(yrs + yrs6 + yrs7 + yrs8))
    states = sorted(set(states + states6 + states7 + states8))

    # this keeps track of which state-year pairs contain MMDs
    mmd_dict = dict()
    mmdyrs = sorted(set([x[0] for x in mmd]))
    for y in mmdyrs:
        # if int(y)%2 == 1:
        #     continue
        mmdst = map(lambda z: z[1], filter(lambda x: x[0] == y, mmd))
        mmd_dict[y] = sorted(mmdst)
        # print y,sorted(mmdst)
    return arr,yrs,states,mmd_dict

def set_winner_need_to_impute(elections,race,dist,demwinner,winvotes=1000,losevotes=0,dincum=False,rincum=False):
    """
    """
    if race not in elections.keys():
        print "WARNING: Trying to override data file, but key not present",race,dist
        return
    
    elec = elections[race]
    i = elec.dists.index(dist)
    if demwinner:
        elec.dcands[i] = Candy(race,'','Dem',winvotes,dincum,True)
        elec.rcands[i] = Candy(race,'','Rep',losevotes,rincum,False)
    else:
        elec.dcands[i] = Candy(race,'','Dem',losevotes,dincum,False)
        elec.rcands[i] = Candy(race,'','Rep',winvotes,rincum,True)

def correct_errors(elections):
    """ fix hard-coded errors in data files (leave original data files unchanged!)
    """
    # believe it's a straight error in the file (uncontested race, so no votes listed)
    set_winner_need_to_impute(elections,'2000_VA_11','VAVA4604',True,1000,0,True,False)

    # Rodney Alexander switched parties in 2004 from Dem to Rep - Democrat in 2002 race
    # not sure why no votes listed in file
    set_winner_need_to_impute(elections,'2002_LA_11','LALA1805',True,5028,4972,False,False)

    # 2002 LALA1807 Chris Johns won with 86% against a libertarian - solidly republican since then...
    # WARNING: Imputed winner but value too small:  2002 LA 11 LALA1807 NoName Dem 1000 nan True True 0.469353411591
    # I guess I shouldn't do anything about this one
    
    # 2008 LALA1802 - this was William Jefferson (money in the freezer...), so a strange year
    # not sure why no data in file - listing Jefferson as incumbent, but not sure that makes sense
    # given circumstances
    set_winner_need_to_impute(elections,'2008_LA_11','LALA1802',True,33132,31318,True,False)

    ######################## Still need to address
    # LALA1804 in 1990s - gerrymandered and redrawn....
    # TXTX4309 in 1996 - gerrymandered and redrawn...

#################################################################################
# initialize everything
def init_all():
    """ read everything in and return data
    """
    loc_arra,loc_yrs,loc_states,loc_mmd_dict = read_all_data()
    loc_elections = make_records(loc_arra)
    
    cycstates = []
    for elec in loc_elections.values():
        if elec.cyc_state not in cycstates:
            cycstates.append(elec.cyc_state)
    
    loc_prior_cycles = create_cycles(loc_elections,loc_mmd_dict,True,False,False)
    loc_recent_cycles = create_cycles(loc_elections,loc_mmd_dict,False,True,True)

    for c in loc_prior_cycles:
        for k in c.elecs.keys():
            c.elecs[k].compute_overall()       

    for c in loc_recent_cycles:
        for k in c.elecs.keys():
            c.elecs[k].compute_overall()       

    for elec in loc_elections.values():
        for i in range(elec.Ndists):
            if elec.demfrac[i] >= 1:
                elec.demfrac[i] = 0.95
            
    return loc_arra, loc_yrs, loc_states, cycstates, loc_mmd_dict, \
        loc_elections, loc_prior_cycles, loc_recent_cycles

def write_elections(fn,elections,mmd):
    """ write out all data for elections that aren't part of mmd along with imputation data

    Do I need to worry about things being converted to strings?
    03.13.17: Init version. Can't test until imputation happens again
    """
    f = open('/home/gswarrin/research/gerrymander/' + fn,'w')
    for elec in elections.values():
        if (elec.chamber != '9' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]) and \
           int(elec.yr) >= 1972:
            # general data of an actual election
            l = ['Elec',elec.yr,elec.state,elec.cyc_state,elec.chamber,str(elec.Ndists),str(elec.Ddists),str(elec.Dfrac)]
            f.write(','.join(l) + '\n')
            # data for each district
            for i in range(elec.Ndists):
                # print data about imputation
                if elec.status[i] == 1:
                    print elec.yr,elec.state,elec.chamber,i,elec.status[i],elec.impute_data[i]
                    ttt = map(lambda x: float(x), elec.impute_data[i])
                    implist = "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f" % \
                              (ttt[0],ttt[1],ttt[2],\
                               ttt[3],ttt[4],ttt[5])
                else:
                    implist = ",,,,,"
                demfracstr = "%.3f" % (elec.demfrac[i])
                # 07.27.17 - added presidential vote
                if elec.pvote[i] == '':
                    pvotestr = ''
                else:
                    pvotestr = "%.3f" % (elec.pvote[i])
                f.write('Impute,'  + str(elec.status[i]) + ',' + demfracstr + ',' + implist + ',' + pvotestr + '\n')
                # print out data about democrat candidate
                c = elec.dcands[i]
                cname = c.cname.replace(',',';')
                l = [c.race,cname,c.party,str(c.votes),str(c.incum),str(c.winner),str(c.orig_votes),str(c.is_imputed)]
                f.write('Dem Cand,' + ','.join(l) + '\n')
                # print out data about republican candidate
                c = elec.rcands[i]
                cname = c.cname.replace(',',';')
                l = [c.race,cname,c.party,str(c.votes),str(c.incum),str(c.winner),str(c.orig_votes),str(c.is_imputed)]
                f.write('Rep Cand,' + ','.join(l) + '\n')
    f.close()

def read_elections(fn):
    """ read in info so don't need to fit model each time I restart
    03.13.17: Init version. Can't test until imputation happens again
    """
    elections = dict()
    states = []
    cyc_states = []
    years = []
    # get all lines
    lines = [line.rstrip('\n') for line in open('/home/gswarrin/research/gerrymander/' + fn)]
    idx = 0
    still_going = True
    while idx < len(lines): # WARNING: don't check length restriction down below
        l = lines[idx].split(',')
        # first read in general election data
        if l[0] != 'Elec':
            print "Problem in file at line %d; new election expected" % (idx)
            return
        mykey = '_'.join([l[1],l[2],l[4]])

        if l[1] not in years:
            years.append(l[1])
        if l[2] not in states:
            states.append(l[2])
        if l[3] not in cyc_states:
            cyc_states.append(l[3])

        elections[mykey] = Gerry(l[1],l[2],l[4])
        elec = elections[mykey]
        elec.cyc_state = l[3]
        elec.Ndists = int(l[5])
        elec.Ddists = int(l[6])
        elec.Dfrac = float(l[7])

        idx += 1

        # now go through each district
        for j in range(elec.Ndists):
            l = lines[idx+3*j].split(',')
            # print l
            if l[0] != 'Impute':
                print "Problem in file at line %d; imputation data expected" % (idx+3*j)
                return
            elec.status.append(int(l[1]))
            elec.demfrac.append(float(l[2]))
            if elec.status[-1] == 1:
                elec.impute_data = [float(x) for x in l[3:-1]]
                # 07.27.17 - added in presidential vote
                elec.pvote = l[-1]
            else:
                elec.impute_data = ['','','','','','']

            l = lines[idx+3*j+1].split(',')
            # print l
            if l[0] != 'Dem Cand':
                print "Problem in file at line %d; dem candidate expected" % (idx+3*j+1)
                return
            # print l
            elec.dcands.append(Candy(l[1],l[2],l[3],l[4],l[5],l[6]))
            elec.dcands[j].is_imputed = elec.status[-1]

            l = lines[idx+3*j+2].split(',')
            if l[0] != 'Rep Cand':
                print "Problem in file at line %d; rep candidate expected" % (idx+3*j+2)
                return
            elec.rcands.append(Candy(l[1],l[2],l[3],l[4],l[5],l[6]))
            elec.rcands[j].is_imputed = elec.status[-1]
        idx += (3*elec.Ndists)

    return sorted(years),sorted(states),sorted(cyc_states),elections
