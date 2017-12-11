# Functions for reading data

def read_tsv_2010(fn):
    """ State races through 2010 from Harvard Dataverse ICPSR 34297.

    # Reviewed 11.15.17
    Extracts desired info from state legislative races.
    Note that in this file, only one candidate per line.

    Returns a 4-tuple:
    ans = [
      myid = '1974_AL_9' - year,state and legislative body
      newdistid = '61.' - district identifier - TODO: describe precisely
      votes = candidate's vote totals. 0 if entry in file is empty
      party = 'Dem' or 'Rep'
      incumbent = True or False
      winner = True or False
      '' = This file does not record presidential data. TODO: Check
    ]
    yrs = list of years in file (as strings)
    states = list of states in file (two-letter abbreviations)
    mmd = list of [year,state] pairs for which not all districts are single member
    
    07.27.17 - added empty string at end since don't know presidential data (yet?)

    TODO: Should these be treated differently? Currently dropped from analysis.
    So number of seats will be wrong for these states and years.
    Elections with no major-party candidate:
    2008 AR 39th district - Richard Carroll - Green party (switched to Dem in 2009)
    '1972_CA_914.'
    '1976_CA_911.' 
    '1976_CA_953.' 
    '1976_CA_967.'
    '1980_CA_929.' 
    '1982_CA_934.'
    """
    f = open(homepath + 'data/' + fn,'r')

    cnt = 0  # line of file

    # list of districts with a candidate who is not a Dem or a Rep
    nonDRdists = []
    # list of districts with a candidate who is a Dem or a Rep
    DRdists = []
    # we'll compare these lists to see how many districts don't have either running

    # initializing items we'll be returning
    ans = []    # tuples 
    yrs = []    # years
    states = [] # states
    mmd = []    # for keeping track of who has multi-member districts

    for line in f:
        cnt += 1
        # get the headers of the columns
        if cnt == 1:
            hdrs = line.strip().split('\t')
            continue
        l = line.strip().split('\t')
        
        ############################################################
        # restriction attention to certain lines
        ############################################################
        if int(l[hdrs.index('v05')]) < 1972:
            continue

        # '9' codes for the lower house in each state
        # TODO: Clarify what happens in Nebraska
        if l[hdrs.index('v07')] != '9':
            continue
        # v12 - district type (want SMD only, which is coded by a 1)
        if l[hdrs.index('v12')] != '1':
            # v05 = year
            # v02 = state
            tmp = [l[hdrs.index('v05')],l[hdrs.index('v02')]]
            if tmp not in mmd:
                mmd.append(tmp)
            continue
        # v16 - election type (want General, coded by a G)
        if l[hdrs.index('v16')] != 'G':
            continue
        # v21 - 100 = Democrat, 200 = Republican, see Study #8907 for detailed listing
        # 300 = Fused Dem/Rep
        # 400 = non-major party
        # 500 = non-partisan election
        # 600 = write-in/scattering
        # 700 = unidentified
        if l[hdrs.index('v21')] not in ['100','200']:
            # print "Candidate not Dem or a Rep: ",l[hdrs.index('v05')],l[hdrs.index('v02')],\
            #     l[hdrs.index('v11')],l[hdrs.index('v21')]
            tmpid = '_'.join([l[hdrs.index('v05')],l[hdrs.index('v02')], l[hdrs.index('v07')]]) + \
                    l[hdrs.index('v11')]
            if tmpid not in nonDRdists:
                nonDRdists.append(tmpid)
            continue

        # The 2012 data doesn't contain all columns. Not relevant for <= 2010 currently used.
        # Ignore races in which more than one Democrat or one Republican running.
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
        # keep track of years and states represented in the data
        if l[hdrs.index('v05')] not in yrs:
            yrs.append(l[hdrs.index('v05')])
        if l[hdrs.index('v02')] not in states:
            states.append(l[hdrs.index('v02')])
            
        myid = '_'.join([l[hdrs.index('v05')],l[hdrs.index('v02')], l[hdrs.index('v07')]])
        if l[hdrs.index('v23')] == ' ' or l[hdrs.index('v23')] == '': # candidates vote total
            l[hdrs.index('v23')] = 0
        if l[hdrs.index('v21')] == '100':
            party = 'Dem'
        else:
            party = 'Rep'
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
            
        # create a district id if don't have one in the file
        if 'v11' in hdrs:
            newdistid = l[hdrs.index('v11')]
        else:
            newdistid = l[hdrs.index('v08')]+l[hdrs.index('v09')]+l[hdrs.index('v10a')]+\
                        l[hdrs.index('v10b')]+l[hdrs.index('v10c')]

        DRdists.append('_'.join([l[hdrs.index('v05')],l[hdrs.index('v02')], l[hdrs.index('v07')]])\
                       + l[hdrs.index('v11')])
        str = [myid,newdistid,int(l[hdrs.index('v23')]),party,incum,winner,'']
        ans.append(str)      
    # keep track of which races aren't kept because no major-party candidate
    # first filter out those that will be ignored due to MMD issues
    nonMMDnonDR = filter(lambda x: [x[:4],x[5:7]] not in mmd, nonDRdists)
    # These are races we are completely omitting.
    # TODO: Decide if we want to include as unknown winner for imputation purposes.
    for x in filter(lambda x: x not in DRdists, nonMMDnonDR):
        print "No major-party candidates, not MMD - skipping race: ",x
    return ans,yrs,states,mmd

##########################
def read_jacobson_csv(fn,ignore_pre_1972=True):
    """ Congressional races from 1946 - 2014. From Jacobson.

    # Reviewed 11.15.17
    07.27.17 - added pres votes at end as either empty string or fraction of 1.
    Returns 3-tuple
    ans = [

    ]
    yrs = list of years in data file
    states = list of states in data file

    Nothing is returned for MMDs since we're outlawed in 1967 
      and we're restricting to elections since 1972.
    In file, each election is encoded in a single line (unlike with 34297).
    So output has to split into two lines.

    Congressional races in which 3rd party won (so election ignored):
    1972 MA 9th - Joe Moakley won (ended up caucusing with Dems FWIW)
    1990-2004 VT 1st - Bernie Sanders won
    2000 VA 5th - Virgil Goode won Independent
    """
    # jacobson stores states as indices in this list (shifted down by one)
    # TODO: rewrite to use global state indices
    statelist = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
      
    f = open(homepath + 'data/' + fn,'r')

    cnt = 0     # line in file
    ans = []    # initialize list of tuples
    yrs = []    # initialize year list we'll be returning
    states = [] # initialize state list we'll be returning
    myids = []  # ids i'm creating

    for line in f:
        cnt += 1
        # read in headers
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

        if ignore_pre_1972 and int(l[0]) < 1972:
            continue

        # read in year, state and district
        if l[0] not in yrs:
            yrs.append(l[0])
        stcode = int(l[1][:-2])
        distcode = l[1][-2:]
        state = statelist[stcode-1]
        if state not in states:
            states.append(state)

        uncontested = False    
        myid = '_'.join([l[0],state,'11']) # '11' means US Congress; 1972_AL_11
        if myid not in myids:
            myids.append(myid)
        if l[4] == ' ' or l[4] == '': # democratic candidate's vote total
            dvotes = 0
            uncontested = True
        else:
            if '.' in l[4]:           # vote expressed as a percentage -> convert to out of 1000
                dvotes = int(float(l[4])*10)
            else:
                dvotes = int(l[4])*10
        if l[12] == ' ' or l[12] == '':  # democratic presidential vote as percentage
            pvotes = ''
        else:
            pvotes = float(l[12])/100

        if dvotes > 1000:             # this shouldn't happen since starting with percentages
            print "reading jacobson %s %s %s - too many votes!" % (l[0],state,l[1])

        # 5 = two incumbents from opposing parties
        # 0,1 = Rep and Dem incumbents, respectively
        # other cases treated as no incumbents. TODO: See how many of these there are.
        repinc = (l[2] in ['0','5'])  
        deminc = (l[2] in ['1','5'])
        dwinner = (l[3] == '1')       # Dem winner
        rwinner = (l[3] == '0')       # Rep winner
        if not dwinner and not rwinner: # Third-party won. TODO: Count how many. Currently ignored completely.
            print "No major party winner - skipping race",myid,l[1]
            continue
        redrawn = (l[9] == '1')       # whether district has been redrawn since previous election
        # If contested, then will print two lines (one for Dem, one for Rep)
        # If uncontested, then will only print a line for the winner (if no winner, see above).
        if not uncontested or dwinner:
            ans.append([myid,state+l[1],dvotes,'Dem',deminc,dwinner,pvotes])
        if not uncontested or rwinner:
            ans.append([myid,state+l[1],1000-dvotes,'Rep',repinc,rwinner,pvotes])
        # TODO: Flag some other way? There are a lot of these.
        # if uncontested:
        #     print "Uncontested: ",myid,state+l[1],dvotes,dwinner,rwinner
        if (int(l[0]) % 2) == 1:
            print "Odd year: ",myid,state+l[1],dvotes,dwinner,rwinner
    return ans,yrs,states # ,mmd

####################################
def read_2012_state_csv(fn,chamber):
    """ For reading in state races I typed in from Ballotpedia

    # Reviewed 11.15.17
    07.27.17 - added empty string at end since don't know presidential vote currently
    In file, each race is one line. Races are output as two lines (if contested).

    ans = [
    - myid = '2014_NC_11'
    - distcode = AL0102 or WI1
    - votes obtained by candidate
    - Party: 'Dem' or 'Rep'
    - Whether candidate is incumbent
    - Whether candidate is winner
    - presidential vote (unknown)
    ]
    years = list of years seen
    states = list of states seen
    """
    f = open(homepath + 'data/' + fn,'r')
    cnt = 0
    ans = []         # tuple we're returning
    yrs = []         # years seen
    states = []      # states seen
    myids = []       # id's for each district
    for line in f:
        cnt += 1
        # read in column headers
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
        # TODO: Check if appropriate format.
        # In particular, does it match <= 2010 format? Does it ever need to?
        if chamber == '9':            # state legislature
            distcode = state + l[2]   # eg, WI1
        else:
            tmp = stlist.index(state)
            if tmp < 10:
                tmp = '0' + str(tmp)
            else:
                tmp = str(tmp)
            distcode = state + tmp + l[2] # eg, AL0104
        if state not in states:
            states.append(state)

        uncontested = False    
        myid = '_'.join([l[0],state,chamber]) 
        if myid not in myids:
            myids.append(myid)
        if l[3] == ' ' or l[3] == '': # democratic candidate's vote total
            l[3] = '0'
            uncontested = True
        if l[3] == '0' or l[4] == '0':
            uncontested = True
        deminc = (l[5] == '1')        # democratic incumbent
        repinc = (l[6] == '1')        # republican incumbent
        dwinner = (int(l[3]) > int(l[4]))  # democratic winner
        rwinner = (int(l[4]) > int(l[3]))  # republican winner
        if not dwinner and not rwinner:    # third-party won. TODO: How to handle.
            print "Skipping race %s %s %s because no Dem/Rep winner" % (l[0],state,distcode)
            continue
        if not uncontested or dwinner:
            ans.append([myid,distcode,int(l[3]),'Dem',deminc,dwinner,''])
        if not uncontested or rwinner:
            ans.append([myid,distcode,int(l[4]),'Rep',repinc,rwinner,''])
        # TODO: Handle differently? There are a lot of these.
        # if uncontested:
        #     print "Uncontested: ",myid,distcode,l[3],dwinner,rwinner
    return ans,yrs,states # ,mmd

####################
def read_all_data():
    """ read in all data files

    # Reviewed 11.15.17
    """
    arr = []
    yrs = []
    states = []

    arr,yrs,states,mmd = read_tsv_2010('34297-0001-Data.tsv') # state races up to 2010 from Harvard dataverse
    arr6,yrs6,states6 = read_jacobson_csv('HR4614.csv',True)  # congressional races from Jacobson. Up to 2014.
    arr7,yrs7,states7 = read_2012_state_csv('stateleg2012plus.csv','9') # State legislature 2012,14,16
    arr8,yrs8,states8 = read_2012_state_csv('cong2016.csv','11')        # Congress 2016

    arr += arr6
    arr += arr7
    arr += arr8
    yrs = sorted(set(yrs + yrs6 + yrs7 + yrs8))
    states = sorted(set(states + states6 + states7 + states8))

    # this keeps track of which state-year pairs contain MMDs
    mmd_dict = dict()
    mmdyrs = sorted(set([x[0] for x in mmd]))
    for y in mmdyrs:
        mmdst = map(lambda z: z[1], filter(lambda x: x[0] == y, mmd))
        mmd_dict[y] = sorted(mmdst)
    return arr,yrs,states,mmd_dict

####################################################################################
def override_race(elections,race,dist,demwinner,winvotes=1000,losevotes=0,dincum=False,rincum=False):
    """ Override the data in a particular race.

    # Reviewed 11.15.17
    Only for recognized errors in original data files.
    TODO: Recheck.
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

    # Reviewed 11.15.17
    This is by no means a comprehensive list I am sure
    """
    # believe it's a straight error in the file (uncontested race, so no votes listed)
    override_race(elections,'2000_VA_11','VAVA4604',True,1000,0,True,False)

    # Rodney Alexander switched parties in 2004 from Dem to Rep - Democrat in 2002 race
    # not sure why no votes listed in file
    override_race(elections,'2002_LA_11','LALA1805',True,5028,4972,False,False)

    # 2002 LALA1807 Chris Johns won with 86% against a libertarian - solidly republican since then...
    # WARNING: Imputed winner but value too small:  2002 LA 11 LALA1807 NoName Dem 1000 nan True True 0.469353411591
    # I guess I shouldn't do anything about this one
    
    # 2008 LALA1802 - this was William Jefferson (money in the freezer...), so a strange year
    # not sure why no data in file - listing Jefferson as incumbent, but not sure that makes sense
    # given circumstances
    override_race(elections,'2008_LA_11','LALA1802',True,33132,31318,True,False)

    # TODO: Look into these - have note that something is wrong.
    # LALA1804 in 1990s - gerrymandered and redrawn....
    # TXTX4309 in 1996 - gerrymandered and redrawn...

#################################################################################
# read data and impute votes
def init_all():
    """ read everything in, impute uncontested races and return data

    # Reviewed 11.15.17
    """
    # read in data from various files and place in a consistent format
    loc_arra,loc_yrs,loc_states,loc_mmd_dict = read_all_data()
    # put into records that are easier to work with.
    loc_elections = make_records(loc_arra)
    
    # list of possible state names taking into account variations used to
    # keep track of different district plans within the same decade
    cycstates = []
    for elec in loc_elections.values():
        if elec.cyc_state not in cycstates:
            cycstates.append(elec.cyc_state)
    
    # create_cycles includes the step of imputing votes
    # grab everything up through 2010.
    loc_prior_cycles = create_cycles(loc_elections,loc_mmd_dict,True,False,False)
    # grab everything since 2012.
    loc_recent_cycles = create_cycles(loc_elections,loc_mmd_dict,False,True,True)

    # make sure any legal votes go down to reasonable amounts.
    # TODO: Why am I picking 0.95 here?
    # TODO: Print ones that need to be reset in this way?
    for elec in loc_elections.values():
        for i in range(elec.Ndists):
            if elec.demfrac[i] >= 1:
                elec.demfrac[i] = 0.95
            
    return loc_arra, loc_yrs, loc_states, cycstates, loc_mmd_dict, \
        loc_elections, loc_prior_cycles, loc_recent_cycles

#################################################################################
#################################################################################
#################################################################################

#################################################################################
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
