tmp = []
cnt = 0
# for k in mmd_dict.keys():
#     print k,mmd_dict[k]
for e in electionsa.values():
    if 2002 <= int(e.yr) <= 2010 and e.chamber == '11' and e.state not in mmd_dict[e.yr]:
        for j in range(e.Ndists):
            if e.demfrac[j] > 0.5:
                tmp.append(e.demfrac[j])
            else:
                tmp.append(1-e.demfrac[j])
        # tmp += sum(e.demfrac)
        cnt += e.Ndists
print "%.3f" % np.median(tmp)
