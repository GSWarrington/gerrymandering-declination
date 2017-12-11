datapath = '/home/gswarrin/talks/2017/gerrymandering/data/'

def plot_judge(fn):
    """
    """
    ans = [[] for j in range(5)]
    cnt = 0
    num = 0
    f = open(datapath + fn,'r')
    for line in f:
        l = line.rstrip().split(',')
        if cnt > 1:
            for j in range(num):
                for i in range(int(l[1])):
                    ans[j].append(float(l[4+3*j]))
        elif cnt == 0:
            num = int(l[0])
        cnt += 1

    f.close()
    return ans
    print get_declination('',ans)

anscur = plot_judge('superior-current.csv')
ansv2 = plot_judge('superior-2.csv')
ansv3 = plot_judge('superior-3.csv')
ansv4 = plot_judge('superior-4.csv')
ansv5 = plot_judge('superior-5.csv')

# print len(anscur),anscur[0]
# print len(anscur),anscur[1]
# print len(ansv4),ansv4
talk_intro('wi_talk',[Nelections['2010_WI_9'].demfrac,Nelections['2012_WI_9'].demfrac],['2010 WI State','2012 WI State'],True,True,mylab='Dem. fraction')

talk_intro('nc_judge_pres_talk',[anscur[0],ansv5[0]],['Current','Proposed (ver. 5)'],True,True,mylab='Dem. fraction')

talk_intro('nc_judge_pres_3',[anscur[0],ansv2[0],ansv3[0],ansv4[0],ansv5[0]],['Superior Cur P','v2','v3','Superior v4 P','v5'],True,True,mylab='Dem. fraction')
talk_intro('nc_judge_pres',[anscur[0],ansv2[0],ansv3[0],ansv4[0],ansv5[0]],['Superior Cur P','v2','v3','Superior v4 P','v5'],True,True,mylab='Dem. fraction')
talk_intro('nc_judge_gov',[anscur[1],ansv2[1],ansv3[1],ansv4[1],ansv5[1]],['Superior Cur G','v2','v3','Superior v4 G','v5'],True,True,mylab='Dem. fraction')
# fig_eg('asdf','asdf',anscur,True)
