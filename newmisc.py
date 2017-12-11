# def get_seat_change(d,betav,betac,p):
#     """ see how packing and cracking affect estimated number of seats
#     """
#     fig = plt.figure(figsize=(8,8))

#     # add some random error back in 
#     rerr = list(0.057 * np.random.randn(10000))
#     rcp = list(np.random.uniform(0,1,10000))
#     sarr = []
#     est = []
#     nest = []
#     npack = 0

#     allvote = []
#     for yr in d.keys():
#         for st in ['PA']: # d[yr].keys():
#             vote = d[yr][st]
#             # if len(vote) < 20:
#             #     continue
#             allvote.extend(vote)

#             # print vote
#             # compute expected number of seats
#             # plt.scatter([x[0] for x in vote],[x[1] for x in vote],s=3)
#             # newy = map(lambda i: vote[i][1]-(0.13 + 0.8*vote[i][0])+rerr.pop(), range(len(vote)))
#             newy = map(lambda i: (0.13 + 0.8*vote[i][0])+rerr.pop(), range(len(vote)))
#             # newy = map(lambda i: vote[i][1]+rerr.pop(), range(len(vote)))
#             # compute expected number of seats

#             # try to pack/crack
#             rvote = sorted([x[1] for x in vote])
#             if len(filter(lambda x: x < 0.5,rvote)) >= 2 and \
#                len(filter(lambda x: x > 0.5,rvote)) >= 2 and \
#                rcp.pop() < p:
#                 # if len(vote) > 4 and min([x[1] for x in vote]) < 0.5 and max([x[1] for x in vote]) > 0.5:
#                 # print sorted([x[1] for x in vote])
#                 isok,newl = pack_or_crack(sorted([x[1] for x in vote]),crack=True) # (rcp.pop() < 1))
#                 # convert to presidential
#                 if isok:
#                     npack += 1
#                     newz = map(lambda i: (0.13 + 0.8*newl[i])+rerr.pop(), range(len(newl)))
#                     ea = [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newy]
#                     nea = [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newz]

#                     for i in range(len(newy)):
#                         print newy[i],newz[i],ea[i],nea[i]
#                         # print [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newy]
#                     # print [1/(1+np.exp(-(betac[yr]+betav*t))) for t in newz]
#                     # get new estimate
#                     est.append(sum([1/(1+np.exp(-(betac[yr]+betav*t))) for t in newy]))
#                     nest.append(sum([1/(1+np.exp(-(betac[yr]+betav*t))) for t in newz]))
#                     sarr.append(len(filter(lambda x: x > 0.5, [x[0] for x in vote])) + np.random.randn(1)[0]*0.1)

#     # print stats.pearsonr([x[0] for x in allvote],[x[1] for x in allvote])
#     # print stats.linregress([x[0] for x in allvote],[x[1] for x in allvote])
#     # plt.plot([0,1],[0.1,0.1+0.86],color='black')
#     newy = map(lambda i: allvote[i][1]-(0.13 + 0.8*allvote[i][0]), range(len(allvote)))
#     # plt.scatter([x[0] for x in allvote],newy,color='green')
#     # print "stddev: ",np.std(newy)
#     # plt.plot([0,1],[0.13,0.13+0.8],color='black')

#     plt.scatter(sarr,est,s=5)
#     # print "total estd: ",sum(est),len(allvote)
#     # print "new estd: ",sum(nest),len(allvote)

#     # this may be appropriate viewpoint, but I am treating leg vote as indep variable....
#     # mydata = odr.Data([x[0] for x in allvote], [x[1] for x in allvote])
#     # myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
#     # myoutput = myodr.run()
#     # myoutput.pprint()
#     # print myoutput.sum_square

#     plt.savefig('/home/gswarrin/research/gerrymander/seat-est')
#     plt.close()
#     return npack,sum(est),sum(nest)

# def chart_dec_total(elections,mmd,chamber):
#     """
#     """
#     yrarr = []
#     arr = []
#     xarr = []
#     yarr = []
#     for yr in range(1972,2018,2):
#         cnt = 0
#         hasone = False
#         for elec in elections.values():
#             if elec.chamber == chamber and int(elec.yr) == yr and elec.Ndists >= 4 and \
#                (chamber == '11' or elec.yr not in mmd.keys() or elec.state not in mmd[elec.yr]):
#                 hasone = True
#                 tmp = count_extra_seats(elec.demfrac)
#                 cnt += tmp
#                 nseats = get_declination('',elec.demfrac)*elec.Ndists*1.0/2
#                 xarr.append(nseats)
#                 yarr.append(tmp)
#                 print "Elec %s %s %s %d % .3f % .3f" % (elec.yr,elec.state,elec.chamber,elec.Ndists,nseats,tmp)
#         if hasone:
#             print yr,cnt
#             arr.append(cnt)
#             yrarr.append(yr)
#     plt.figure(figsize=(12,8))
#     plt.plot(yrarr,arr,'ro-')
#     plt.savefig('/home/gswarrin/research/gerrymander/pics/seatsovertime' + chamber)
#     plt.close()

#     plt.figure(figsize=(12,8))
#     plt.scatter(xarr,yarr)
#     plt.savefig('/home/gswarrin/research/gerrymander/pics/seatsovertime-scatter' + chamber)
#     plt.close()
