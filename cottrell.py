#######################################################
# for playing around with chen-cottrell simulation data
#######################################################

import os.path

statelist = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    

def get_state_actual_probs(yr,st):
    """ read in who won the district versus mccain share of vote
    """
    statelist = ['blah','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']    
      
    f = open('/home/gswarrin/research/gerrymander/data/HR4614.csv','r')
    expd = 0.0
    cnt = 0
    tot = 0
    ans = []
    for line in f:
        cnt += 1 
        if cnt == 1:
            hdrs = line.strip().split(',')
            continue
        l = line.strip().split(',')
        
        #  3 is who won (0=Rep; 1=Dem)
        # 12 dem's share of presidential vote - uses most recent election
        # 2 is incumbency (0=Rep; 1=Dem)
        # let's make incumbency positive for republicans
        if int(l[0]) == yr and l[12] != '' and int(l[3]) <= 1 and \
           int(l[1][:-2]) == statelist.index(st):
            tot += 1
            ans.append(float(l[12])/100)
    f.close()
    # print yr,tot," expected: ",expd
    return ans


def cottrell_total_seats(betav,betac):
    """ try to figure out how many total seats switched
    """
    act = []
    sim = []
    for st in statelist:
        fn = '/home/gswarrin/research/gerrymander/data/ChenCottrell/' + st + 'simul.txt'
        if os.path.isfile(fn):
            # print st
            f = open(fn,'r')
            l = cottrell_fig2('',st,betav,betac)
            act.append([st,l[0]])
            sim.append([st,l[1]])
    diff = [sim[i][1] - act[i][1] for i in range(len(act))]
    diffr = map(lambda x: floor(x) if x > 0 else floor(x)+1, diff)
    # simr = map(lambda x: floor(x) if x > 0 else floor(x)+1, [x[1] for x in sim])
    # actr = map(lambda x: floor(x) if x > 0 else floor(x)+1, [x[1] for x in act])
    print "sim expd: ",np.sum([x[1] for x in sim])
    print "act expd: ",np.sum([x[1] for x in act])
    # rounded version
    print "dif expd: ",np.sum(diffr)

def get_expd_seats(vals,betav,betac):
    """ estimate number of seats based only on votes
    """
    return np.sum(map(lambda x: 1/(1+np.exp(-(betav*x + betac))), vals))

# print get_expd_seats([0.2,0.24,0.28,0.45,0.47,0.53,0.55,0.72,0.76,0.8],17.27,-9.09)
# print get_expd_seats([0.2,0.24,0.28,0.45,0.47,0.49,0.49,0.76,0.82,0.8],17.27,-9.09)
# print get_expd_seats([0.2,0.28,0.34,0.45,0.47,0.49,0.49,0.72,0.76,0.8],17.27,-9.09)

# print get_declination('',[0.2,0.24,0.28,0.45,0.47,0.53,0.55,0.72,0.76,0.8])*5
# print get_declination('',[0.2,0.24,0.28,0.45,0.47,0.49,0.49,0.76,0.82,0.8])*5
# print get_declination('',[0.2,0.28,0.34,0.45,0.47,0.49,0.49,0.72,0.76,0.8])*5

# BW
# print "BW:"
# cottrell_total_seats(17.27,-9.09)

# CC
# print "CC:"
# cottrell_total_seats(21.08,-10.93)

# cottrell_fig2('cottrell-fig2-FL-BWfit','FL',17.27,-9.09)
# cottrell_fig2('cottrell-fig2-FL-CCfit','FL',21.08,-10.93)

# cottrell_fig2('cottrell-fig2-NC-BWfit','NC',17.27,-9.09)
# cottrell_fig2('cottrell-fig2-NC-CCfit','NC',21.08,-10.93)

# cottrell_fig2('cottrell-fig2-PA-BWfit','PA',17.27,-9.09)
# cottrell_fig2('cottrell-fig2-PA-CCfit','PA',21.08,-10.93)

# cottrell_fig2('cottrell-fig2-GA-BWfit','GA',17.27,-9.09)
# cottrell_fig2('cottrell-fig2-GA-CCfit','GA',21.08,-10.93)

# cottrell_fig2('cottrell-fig2-AZ-BWfit','AZ',17.27,-9.09,makepic=False)
# cottrell_fig2('cottrell-fig2-AZ-CCfit','AZ',21.08,-10.93,makepic=False)

def seats_cottrell_one_state(st,slo=1.05,inter=-0.035):
    """ 
    07.27.17 - figure out difference in declination between actual and enacted
    use presidential vote to guess legislative, then use S-declination
    """
    fn = st + 'simul.txt'
    f = open('/home/gswarrin/research/gerrymander/data/ChenCottrell/' + fn,'r')
    # list of vote fractions for each of 200 simulations in the state
    simul_votes = [[] for j in range(205)] 
    # S-declination for each simulation
    Sdec = []
    for i,line in enumerate(f):
        l = line.rstrip().split('\t')
        l2 = map(lambda x: float(x), l[1:]) # skip simulation number
        tmp = 0
        for j in range(len(l2)):
            simul_votes[i].append(1-l2[j])

        fa = get_declination('',[inter + slo*x for x in simul_votes[i]])
        if abs(fa) < 2:
            Sdec.append(fa*multfact*len(simul_votes[i]))

    # get expected number of seats in the state from actual district plan
    actual_plan = sorted(get_state_actual_probs(2012,fn[:2]))
    
    fa2 = get_declination('',[inter + slo*x for x in actual_plan])
    if abs(fa2) < 2:
        Sact = fa2*multfact*len(actual_plan)
    else:
        return False,[]
    
    print st," sim: %.2f act: %.2f" % (np.mean(Sdec),Sact)
    return True,[np.mean(Sdec),Sact] # 

def seats_cottrell(slo=1.05,inter=-0.035):
    """ try to figure out how many total seats switched
    """
    act = []
    sim = []
    dff = []
    for st in statelist:
        if st == 'NV':
            continue
        fn = '/home/gswarrin/research/gerrymander/data/ChenCottrell/' + st + 'simul.txt'
        if os.path.isfile(fn):
            # print st
            f = open(fn,'r')
            bo,l = seats_cottrell_one_state(st,slo,inter)
            if bo:
                act.append(l[1])
                sim.append(l[0])
                dff.append(round(l[1]-l[0]))
    print act
    print sim
    print "Actual %.1f   Sim: %.1f   Dff: %.1f" % (sum(act),sum(sim),sum(dff))
