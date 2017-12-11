###################################
# paper generate pictures

load(homepath + 'basic.py')
load(homepath + 'snr.py')
load(homepath + 'paper-pics.py')
load(homepath + 'alpha.py')
load(homepath + 'plotting.py')
load(homepath + 'results.py')
load(homepath + 'paper-info.py')
load(homepath + 'science-stuff.py')

###################################
# pictures in actual paper
###################################
# make_cong_deltaN_histo(Melections)
# fig2_make_heatmap(Melections,Mmmd)
# make_heatmap_combine(Melections,Mmmd)

# Fig. 1 - paper-angle-plot-2014_NC_11.png
# plot_angle('2014_NC_11',Melections)

# Fig. 2 - paperseries.png 
# ml = ['2012_MD_11','2012_PA_11','2012_TX_11','2012_NY_11']
# plot_paper_timeseries(Melections,ml)

# Fig. 3 - lineplot_angle_CHAMBER_YRMIN.png
# make_line_chart(Melections,'2012','11',3)
blah = []
quints = []
make_line_chart_grid(Nelections,'11',Nmmd,True)
# make_line_chart_grid(Melections,'11',False)
make_line_chart_grid(Nelections,'9',Nmmd,True)
# make_line_chart_grid(Melections,'9',False)
make_scatter('tildedelta-Nxdeld2',[x[0] for x in blah], [x[1] for x in blah])
slope, intercept, r_value, p_value, std_err = stats.linregress([x[0] for x in blah],[x[1] for x in blah])
print slope,intercept,r_value,p_value
b2 = sorted([x[1] for x in blah])
print "95% CI: ",b2[int(9.5*len(b2)/10)]

# Fig. 4 - wi-ex-scatter.png
# make_paper_scatter_wi()
# wi_scatter_decade()

# ???
# new_plot_timeseries(Melections)
# make_f_corr_grid('duh',Melections)

# in comparison section
# plot_nc_tx_pic('nctx',Melections)

###################################
# pictures in SI
###################################
# One year, sorted from most republican to least republican (by angle)
# {st,cong}-pg{1,2}paper-declination-grade.png
# switch to 7x5 grid so fits on one page for each
# split_declination_states('cong2016',7,5,'2016','11',Mstates,Melections,Mmmd)
# split_declination_states('st2008',7,5,'2008','9',Mstates,Melections,Mmmd)

# si_corr('corr-zero-infty',Melections)

# grids of line charts
# make_line_chart_grid(Melections,'11')
# make_line_chart_grid(Melections,'9')

###################################
# Other pictures not sure where they'll go
###################################
# fixed state/chamber over all years
# angle-plot-ST-CHAMBER.png
# grid_all_states('',6,4,'11',Mstates,Melections,Mmmd)
# grid_all_states('',6,4,'9',Mstates,Melections,Mmmd)

# compares gap_* and get_declination
# curves-YR.png
# plot_many_alpha

###################################################################################
###################################################################################

load(homepath + 'basic.py')
load(homepath + 'snr.py')
load(homepath + 'paper-pics.py')
load(homepath + 'alpha.py')
load(homepath + 'plotting.py')
load(homepath + 'results.py')
load(homepath + 'paper-info.py')
load(homepath + 'science-stuff.py')
load(homepath + 'unpack.py')

Output_png = False

sns.reset_orig()
# sns.set_style("ticks")
# fig0_create('fig0-dec-def','2014_NC_11',Nelections)
talk_fig0_create('talk_NC','2014_NC_11',Nelections)

sns.reset_orig()
# fig_discuss('fig-discuss',Nelections)
sns.reset_orig()
# fig_variations('fig-var',Nelections,Ncyc,Nmmd)
sns.reset_orig()
# sns.set_style("ticks")
# fig1_create('fig1-dec-ex','2014_NC_11',Nelections)
sns.reset_orig()
# fig2_make_heatmap_e('fig2-heatmap-e',Nelections,Nmmd)
sns.reset_orig()
## fig3_deltae('fig3-deltae',Nelections,Ncyc,Nmmd)
# fig_deltae_only('fig-deltae-only',Nelections,Ncyc,Nmmd)

##################
# for some reason, generating both at same time causes trouble
sns.reset_orig()
# figS34_linechart('figS4-stateline',Nelections,Ncyc,Nmmd,'9')
sns.reset_orig()
# figS34_linechart('figS3-congline',Nelections,Ncyc,Nmmd,'11')

sns.reset_orig()
# figS12_split_dec_states('figS1-cong2016',7,5,'2016','11',Nstates,Nelections,Nmmd)
sns.reset_orig()
# figS12_split_dec_states('figS2-st2008',7,5,'2008','9',Nstates,Nelections,Nmmd)
sns.reset_orig()
# figS5_wi_scatter(Nelections,Nmmd)

# print get_declination('',Nelections['2012_WI_9'].demfrac)*math.log(99)/2
# make_extreme_table(Nelections,Nstates,Nmmd,'11')

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

load(homepath + 'basic.py')
load(homepath + 'snr.py')
load(homepath + 'paper-pics.py')
load(homepath + 'alpha.py')

load(homepath + 'plotting.py')
load(homepath + 'results.py')
load(homepath + 'paper-info.py')
load(homepath + 'science-stuff.py')

# sns.reset_orig()
# sns.set_style("ticks")
#  fig1_create('fig1','2014_NC_11',Nelections)
# fig2_make_heatmap(Nelections,Nmmd)
# fig3_deltae('fig3-de
# mmarr,ddarr = median_mean(Nelections,Nmmd,'11')
# make_histogram("medmean-hist",mmarr)
# make_scatter("medmean-scat",mmarr,ddarr)
# print stats.pearsonr(mmarr,ddarr)
# make_yoy_corr_scatter(Nelections,Nmmd,'11')
# fig_threebad('fig-three',Nelections)

#############################################################
# Pictures for declination introduction
vals = [[0.2,0.24,0.28,0.45,0.47,0.53,0.55,0.72,0.76,0.8],\
        [0.35,0.41,0.44,0.46,0.48,0.52,0.54,0.56,0.59,0.65],\
        [0.2,0.22,0.24,0.26,0.28,0.72,0.74,0.76,0.78,0.80]]
ttls = ['Dem. 50%','Dem. 50%','Dem. 50%']
fig_intro('intro_fig1',vals,ttls,False,False)
# fig_eg('talk_fig1',vals,ttls,False,True)

vals = [[0.23,0.255,0.285,0.32,0.35,0.375,0.405,0.58,0.60,0.61],\
        [0.45,0.47,0.52,0.57,0.61,0.66,0.70,0.75,0.82,0.91]]
ttls = ['Dem. 40%','Dem. 65%','']
fig_intro('intro_fig2',vals,ttls,False,False)
# fig_eg('talk_fig2',vals,ttls,False,True)
# print np.mean(vals[0])
# print np.mean(vals[1])
# print get_declination('',vals[0])
# print get_declination('',vals[1])
# print "------------------"

vals = [[0.2,0.24,0.28,0.45,0.47,0.53,0.55,0.72,0.76,0.8],\
        [0.2,0.24,0.28,0.45,0.47,0.49,0.49,0.76,0.82,0.8],\
        [0.2,0.28,0.34,0.45,0.47,0.49,0.49,0.72,0.76,0.8]]
ttls = ['Dem. 50%','Dem. 50%','Dem. 50%']
fig_intro('intro_fig3',vals,ttls,False,False)     
# fig_eg('talk_fig3',vals,ttls,False,True)     

vals = [[0.31,0.35,0.4,0.45,0.45,0.45,0.49,0.65,0.7,0.75],\
        [0.45,0.47,0.52,0.57,0.61,0.66,0.70,0.75,0.82,0.91]]
ttls = ['Dem. 50%','Dem. 65%','Dem. 50%']        
# print np.mean(vals[0])
fig_intro_mm('intro_fig4',vals,ttls,False,False)

vals = [Nelections['2012_NC_11'].demfrac,\
        Nelections['2016_AZ_11'].demfrac,\
        Nelections['2012_PA_11'].demfrac]
ttls = ['2012 NC','2016 AZ','2012 PA']
fig_intro('intro_fig5_dec',vals,ttls,True,True,mylab='Dem. vote')
# fig_eg('talk_fig5',vals,ttls,False,True,mylab='Dem. vote')

######################################################################
# pics for end of intro
vals = [[0.2,0.24,0.28,0.45,0.47,0.53,0.55,0.72,0.76,0.8],\
        [0.35,0.41,0.44,0.46,0.48,0.52,0.54,0.56,0.59,0.65],\
        [0.2,0.22,0.24,0.26,0.28,0.72,0.74,0.76,0.78,0.80]]
ttls = ['Dem. 50%','Dem. 50%','Dem. 50%']
fig_intro('intro_fig1_dec',vals,ttls,True,True)

vals = [[0.23,0.255,0.285,0.32,0.35,0.375,0.405,0.58,0.60,0.61],\
        [0.45,0.47,0.52,0.57,0.61,0.66,0.70,0.75,0.82,0.91]]
ttls = ['Dem. 40%','Dem. 65%','']
fig_intro('intro_fig2_dec',vals,ttls,True,True)
# print np.mean(vals[0])
# print np.mean(vals[1])

vals = [[0.2,0.24,0.28,0.45,0.47,0.53,0.55,0.72,0.76,0.8],\
        [0.2,0.24,0.28,0.45,0.47,0.49,0.49,0.76,0.82,0.8],\
        [0.2,0.28,0.34,0.45,0.47,0.49,0.49,0.72,0.76,0.8]]
ttls = ['Dem. 50%','Dem. 50%','Dem. 50%']
fig_intro('intro_fig3_dec',vals,ttls,True,True)     

vals = [[0.31,0.35,0.4,0.45,0.45,0.45,0.49,0.65,0.7,0.75],\
        [0.45,0.47,0.52,0.57,0.61,0.66,0.70,0.75,0.82,0.91]]
ttls = ['Dem. 50%','Dem. 65%','Dem. 50%']        
print np.mean(vals[0])
fig_intro('intro_fig4_dec',vals,ttls,True,True)

vals = [Nelections['2008_WI_9'].demfrac,\
        Nelections['2010_WI_9'].demfrac,\
        Nelections['2012_WI_9'].demfrac,\
        Nelections['2014_WI_9'].demfrac]
ttls = ['2008 WI','2010 WI','2012 WI','2014 WI']
fig_web('wi_pics4',vals,ttls,True,True,mylab='Dem. vote')

vals = [Nelections['2012_WI_9'].demfrac,\
        Nelections['2014_WI_9'].demfrac,\
        Nelections['2016_WI_9'].demfrac]
ttls = ['2012 WI','2014 WI','2016 WI']
# fig_intro('wi_picsii',vals,ttls,True,True,mylab='Dem. vote')

load(homepath + 'science-stuff.py')
vals = [[0.2,0.24,0.28,0.45,0.47,0.53,0.55,0.72,0.76,0.8],\
        [0.35,0.41,0.44,0.46,0.48,0.52,0.54,0.56,0.59,0.65],\
        [0.2,0.22,0.24,0.26,0.28,0.72,0.74,0.76,0.78,0.80]]
ttls = ['Whig 50%','Whig 50%','Whig 50%']
fig_eg('talk_fig1',vals,ttls,False,False)

#########################################################################
#########################################################################

# don't think these two are needed
# load(homepath + 'unpack.py')
# load(homepath + 'seats-simulation.py')
load(homepath + 'science-stuff.py')
load(homepath + 'seatspack.py')

vals1 = [[0.2,0.24,0.28,0.45,0.47,0.53,0.55,0.72,0.76,0.8],\
        [0.35,0.41,0.44,0.46,0.48,0.52,0.54,0.56,0.59,0.65],\
        [0.2,0.22,0.24,0.26,0.28,0.72,0.74,0.76,0.78,0.80]]
ttls1 = ['','','']

vals2 = [[0.2,0.24,0.28,0.45,0.47,0.53,0.55,0.72,0.76,0.8],\
        [0.2,0.24,0.28,0.45,0.47,0.49,0.49,0.76,0.82,0.8],\
        [0.2,0.28,0.34,0.45,0.47,0.49,0.49,0.72,0.76,0.8]]
ttls2 = ['','','']

##############################################################
# generate data for Table 1
# seat_shifts(Nelections)
   
# Fig 1,2 - intro pictures explaining how declination works
# fig_intro('seats_fig1',vals1,ttls1,False,False,mylab='Dem. fraction')
# fig_intro('seats_fig2',vals2,ttls2,False,False,mylab='Dem. fraction')
# Fig 3 - definition of declination via angles
# fig0_create('seats_dec','2014_NC_11',Nelections,seatsbool=True)

# fig 4 - showing how well declination counts manual packing and cracking
# create_mpandc_pic('even-mpandc',Nelections,Nmmd,'11')
# fig 5 - is from xfig
# fig 6 - showing how poorly logistic regression does
# cotd = {1972: -5.98, 1976: -7.81, 1980: -7.4, 1984: -6.39, 1988: -7.24, 1992: -8.47,
#   1996: -9.51, 2000: -8.67, 2004: -8.51, 2008: -8.46, 2012: -9.06} # 17.22

# initial linear fit
gamma0d0 = {1972: 0, 1976: 0, 1980: 0, 1984: 0, 1988: 0, 1992: 0, 1996: 0, 2000: 0, 2004: 0, 2008: 0, 2012: 0}
gamma1d1 = {1972: 1, 1976: 1, 1980: 1, 1984: 1, 1988: 1, 1992: 1, 1996: 1, 2000: 1, 2004: 1, 2008: 1, 2012: 1}
# fitted linear
# gamma0d = {1972: 0.19, 1976: 0.29, 1980: 0.23, 1984: 0.18, 1988: 0.26, 1992: 0.25,
#            1996: 0.24, 2000: 0.21, 2004: 0.18, 2008: 0.16, 2012: 0.08}
# gamma1d = {1972: 0.37, 1976: 0.39, 1980: 0.43, 1984: 0.46, 1988: 0.39, 1992: 0.52,
#            1996: 0.62, 2000: 0.59, 2004: 0.62, 2008: 0.69, 2012: 0.85}

betaint = {1972:   -3.55, 1976:   -7.91, 1980:   -5.49, 1984:   -6.19, 1988:   -5.58,\
  1992:   -7.71, 1996:  -11.37, 2000:   -8.22, 2004:  -10.63, 2008:   -9.38, 2012:  -19.18}

betaslo = {1972:   9.6,  1976:  17.3, 1980:  12.7, 1984:  16.0, 1988:  13.1, 1992:  15.8,\
  1996:  20.8, 2000:  16.3, 2004:  21.7, 2008:  18.9, 2012:  37.2}

gammaint = {1972: 0.197, 1976: 0.292, 1980: 0.234, 1984: 0.183, 1988: 0.255, 1992: 0.250,\
 1996: 0.234, 2000: 0.210, 2004: 0.176, 2008: 0.157, 2012: 0.082}

gammaslo = { 1972: 0.362, 1976: 0.391, 1980: 0.427, 1984: 0.451, 1988: 0.395, 1992: 0.527,
 1996: 0.625, 2000: 0.592, 2004: 0.622, 2008: 0.691, 2012: 0.850}
 
twod = {2012: -7.6, 2008: -7.36} # 15.79
# eg_create_mpandc_pic('eg-mpandc',twod, 15.79)
# cottrell_create_mpandc_pic('cottrell-mpandc-0983',cotd, 17.22, 0.09,0.83)
cottrell_create_mpandc_pic('tt-even-cottrell-mpandc-0000', betaint, betaslo, gamma0d0, gamma1d1)
cottrell_create_mpandc_pic('tt-even-cottrell-mpandc-vard', betaint, betaslo, gammaint, gammaslo)

# pres_v_leg('presleg',[2008,2012])
# create_mpandc_pic(Nelections,Nmmd,'9')
# dbl_plot('logex',[2008,2012])
# triple_plot('pvl')

