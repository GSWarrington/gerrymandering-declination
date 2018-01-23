# wherever the files are located
# - there should be a pics subdirectory for picture generation
homepath = '/home/gswarrin/research/gerrymander/'

##################################################
# imports
import pystan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn.apionly as sns; sns.set_context('notebook')
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats

# Index of state in this list corresponds to the convention many of our data files use.
# '00' is added at he beginning to make this correspondence correct.
stlist = ['00','AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY',\
       'LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH',\
       'OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
GLOBAL_MIN_YEAR = 1971

##################################################
load(homepath + 'util.py')
load(homepath + 'metrics.py')                      # functions for computing metrics (e.g., declination, EG)
load(homepath + 'read_data.py')                    # functions for reading in all data
load(homepath + 'model.py')                        # statistical model for imputing values
load(homepath + 'classes.py')                      # various classes for storing information

load(homepath + 'elj-pics.py')                     # make pictures for ELJ paper
load(homepath + 'elj-tables.py')                   # make tables for ELJ paper
load(homepath + 'elj-info.py')                     # make information for ELJ paper
load(homepath + 'validate.py')                     # validate imputation strategy


##################################################
# generate election data from original files

# This generation requires (in part) the original data file from Jacobson
# 
# larr,lyrs,lstates,lcycstates,lmmd,lelecs,lpriorcyc,lreccyc = init_all()
# write_elections('elec-data-jan04.csv',lelecs)

##################################################
#############################
# Read in saved election data
#############################

# The below data files correspond to two different imputation runs
ryrs,rstates,rcycstates,relecs = read_elections('elec-data-dec30.csv')
rmmd = dict()
syrs,sstates,scycstates,selecs = read_elections('elec-data-jan04.csv')
smmd = dict()

##################################################

make_elj_pics(range(1,13),relecs,rstates,rcycstates)
make_elj_tables(range(1,5),relecs,rstates)
make_elj_info([],relecs,selecs,lelecs,ryrs,rstates,lpriorcyc,lreccyc)
make_tiffs()

##################################################
# cross validation our imputation algorithm
# RMS error of about 0.05 on known races

# tot is the number of tests to make
# result quoted in paper has tot=100
cross_validate_lots(lpriorcyc,tot=2)
