'''

for driving simulation runs
this driver is specific to base_version_downvoting_on module

if wanting to record the data of the runs, see TODO below

"settable" globals / control inputs from loaded module:
NUM_RDS = 60  # how many rounds for the simulation
NUM_TRS = 100  # how many trees to start
NUM_AGS = 30  # how many agents to start
(NM_LO, NM_HI, VOT_RAD)  # for agent on-side post polarity preference
    NM_LO   # for agent polarity spread (thin end)
    NM_HI # for agent polarity spread (fat end)
    VOT_RAD # for Agent class upvote allowed polarity radius
MODULE_SIG  # for Tree class, setting in-camp / out-of-camp
# conditions for whether a post (ie polarity) is in-camp or out-of-camp for
# that agent; ca 0.15 is quite picky, 0.25 is moderately picky, 0.5 is somewhat
# loose, 1.0 (and higher) is kind of like no polarity curation at all
PO_ATTN  # attenuation constant for policing (in [0,1]--1 leaves
# existing dynamic; 0 completely stops policing behavior)
AL_ATTN  # attenuation constant for altruism (in [0,1]--1 leaves
# existing dynamic; 0 completely stops altruism behavior)
RE_ATTN  # for general attenuation of comment and vote responses
# to "replies"--ie a general tendency for an agent to comment or vote on a
# post that is a reply to a post of their own
MODULE_SENSITIVITY = 0.01  # for agent class sensitivity (not used in
production)
CHURN_PROP  # proportion of eligible agents that churn,
each round--so each round ca CHURN_PROP*NUM_AGS will churn
RC_BY_RANK--what method reading_choices() uses to select-by-scoring from the
skim list(s)


'''


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import sys
import pandas as pd
import copy
import time
import itertools
import datetime
import base_version_downvoting_on as base_version


# set imported base_version module globals (kludgy)
base_version.PLOTS_ON = False
base_version.VERBOSE = False

# TO SET each batch run:
base_version.RECORD_DATA = False
# TODO: if RECORD_DATA is True, set the filename to start, and set
#  file_prefix (ie file path) to whatever desired in imported
#  base_version_downvoting_on module
base_version.FILENAME_DATA = "...csv"   # file has to exist
# to run without errors (can be an empty file to start)


base_version.NUM_AGS = 60
base_version.NUM_RDS = 100
base_version.NUM_TRS = 150

# e.g. of manual setting (done in loop below, over specified mesh of values):
#base_version.RE_ATTN = 0.80   # 0-low/no propensity for an agent to reply to
# comments on their own posts; 1-high/full (allowed) propensity
#base_version.MODULE_SIG = 0.50
#base_version.PO_ATTN = 1.0
#base_version.AL_ATTN = 1.0
#base_version.NM_LO = 0.025
#base_version.NM_HI = 0.075
#base_version.VOT_RAD = 0.35

# set up mesh of values to try in main driver loop:
vc_sig = [0.10,0.35,0.60]      # for SIG values to try
vc_poa = [0.10,0.25,0.5,1.0]    # policing
vc_ala = [0.10,0.25,0.5,1.0]    # altruism
vc_rea = [0.10,0.25,0.5,1.0]    # replies
vc_lo_hi_rad = [(0.010,0.030,0.20),(0.025,0.075,0.35)]  # tuples to set
# (NM_LO, NM_HI, VOT_RAD) (ie curation scoring, commenting, voting "tightness")
vc_chp = [0.0,0.0025,0.005]  # churn_prop

# set up grand list with all mesh points in it:
perms_list = list(itertools.product(vc_sig,vc_poa,vc_ala,vc_rea,
                                    vc_lo_hi_rad,vc_chp))  # this gives
# tuples for each parameter combination

num_rps = 3    # how many runs to do per grid point
# run simulation

ct = 0

time_start = time.time()
for tup in perms_list:
    base_version.MODULE_SIG = tup[0]
    base_version.PO_ATTN = tup[1]
    base_version.AL_ATTN = tup[2]
    base_version.RE_ATTN = tup[3]
    base_version.NM_LO,base_version.NM_HI,base_version.VOT_RAD = tup[4]
    base_version.CHURN_PROP = tup[5]
    print("running simulation at parameters, SIG=%s, poa=%s, ala=%s, "
          "rea=%s, nm_lo/hi/vr=%s, churn_prop=%s" % tup)
    for _ in range(num_rps):
        base_version.run_simulation(new_sig=tup[0])  # this is ~kludgy,
        # but tells
        # base_version... to update the Tree::SIG value directly (changing
        # MODULE_SIG globally will obviously not filter through to Tree
        # class)
        ct += 1
time_end = time.time()

print("total time elapsed for %s runs (sec): %s" % (ct,time_end-time_start))
print("seconds per run: %s" % ((time_end-time_start)/ct))


