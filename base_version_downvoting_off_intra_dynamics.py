'''

very similar to base_version_downvoting_off, but with diagnostics for polar
skew in __main__

'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import collections  # eg my_dq = collections.deque([])  # for empty deque
import random
import sys
import pandas as pd
import copy
import time
import datetime
import bisect

PLOTS_ON = False     # to turn on plotting features
VERBOSE = False
RECORD_DATA = False      # turns on / off dataframe save of simulation stats
#FILENAME_DATA = "base_21_sessions_1.csv"
FILENAME_DATA = ""

# module-globals:

NUM_RDS = 100  # how many rounds for the simulation
NUM_TRS = 150  # how many trees to start
NUM_AGS = 60  # how many agents to start

NM_LO = 0.025   # for agent polarity spread (thin end)
NM_HI = 0.075   # for agent polarity spread (fat end)
VOT_RAD = 0.35   # for Agent class upvote allowed polarity radius, in vote()

# these following are to set maximal amounts to capture for reading posts:
NRI_BASE = 5
NRI_RAND = 5
NRI_AGNT = 15
POL_AGNT = 10
ALT_AGNT = 5

# Agent.spread--this could be changed here too at the global level; by
# default, when agents are created their vote() and comment() spread score is
# symmetric in the "usual" range [0.025,0.075]--in fact, to affect spread,
# the easiest way is to probably just change NM_LO, NM_HI (which are used in
# creating agents);
# also note, for Agent.spread to make much difference, at least in vote(),
# the main effect is seen going from eg 0.05 to 0.025--the region 0.05 to
# 0.075 is kind of flat re spread's effects (see note(s) in Agent.vote())

MODULE_SIG = 0.35   # for Tree class, setting in-camp / out-of-camp
# conditions for whether a post (ie polarity) is in-camp or out-of-camp for
# that agent (note, this is more for running this script as stand-alone; for
# batch processes via run_simulation(), control of SIG in Tree class can be
# handled directly, in that function
PO_ATTN = 0.5    # attenuation constant for policing (in [0,1]--1 leaves
# existing dynamic; 0 completely stops policing behavior)
AL_ATTN = 0.5     # attenuation constant for altruism (in [0,1]--1 leaves
# existing dynamic; 0 completely stops altruism behavior)
RE_ATTN = 0.5       # for general attenuation of comment and vote responses
# to "replies"--ie a general tendency for an agent to comment or vote on a
# post that is a reply to a post of their own
MODULE_SENSITIVITY = 0.01       # for agent class sensitivity
CHURN_PROP = 0.005      # proportion of elligible agents that churn, each round

RC_BY_RANK = False      # for setting how reading choices are made from the
# skim list(s), in reading_choices() function


class Agent:

    MAX_ROUNDS = 1000   # class variable, limiting allowed # of rounds for
    # the simulation; to ref inside, eg, type(self).MAX_ROUNDS
    SENSITIVITY = MODULE_SENSITIVITY  # class variable, for polarity feedback
    # sensitivity
    # for read posts (like ri and replies); this is approximately the maximal
    # amount of change allowed for the agent's polarity score, per round

    def __init__(self, ID, pol, spd, ev, ib, ni, police, altruism, ts, tm):

        # agent values
        self.id = ID
        self.polarity = pol     # polarity score, in (-1,1)
        self.base_polarity = pol    # saves the polarity instantiated with,
        # while polarity can change as the simulation goes on
        self.spread = spd   # polarity spread; under eg the gaussian
        # method, this will be a value in [0.025,0.075] (ish), with 0.075
        # being a "fat" (loose)
        # distribution, and 0.025 being a "thin" distribution (selective)
        self.expressive_value = ev      # [0,1]
        self.inter_belong = ib      # interaction and belonging, [0,1]
        self.need_to_influence = ni # [0,1]
        self.policing = police  # policing out-of-camp posts, voting [0,1]
        self.altruism = altruism  # reward altruism, commenting on out-of-camp
        # posts, [0,1]

        self.tree_starting = ts   # how likely the agent is to start a tree,
        # per (active) round; note this may not be used--may just source
        # new trees anonymously; if used it should be small (low chance per
        # round of agent creating a tree)
        self.TM = tm    # reward multiplier; this decides how much the
        # (normalized) total reward score for the round feeds back into
        # reading, comment, and voting propensities; this should be in [0,1]

        self.obs_lookback = 5     # really, a class attribute--how many rounds (
        # active
        # or not) to look back to compute recent reply counts and recent
        # extrinsic points

        # total reward;
        # note, this may not even be used--the per-round "total reward"
        # is saved in the extrinsic_points list/array
        self.total_reward = 0.0

        # behaviors
        self.comment_propensity = self.comp_comm(0.0)  # this will be in [0,1]
        self.general_reading = self.comp_genr(0.0)  # this will be in [0,1]
        self.general_voting = self.comp_genv(0.0)  # this will be in [0,1]

        MR = type(self).MAX_ROUNDS # for compact syntax

        # history of observations
        self.reply_counts = np.zeros(MR)  # one entry for each round (active or
        # not); entries in this array are updated via add_reply_ping
        self.extrinsic_points = np.zeros(MR)  # one entry for each round (active
        # or not)
        self.downvotes = np.zeros(MR) # downvotes collected, per round

        # history of total rewards (total, and trailing)
        self.tr_total = np.zeros(MR)  # updated in set_total_reward()
        self.tr_trail = np.zeros(MR)  # updated in set_total_reward()

        # list of tree numbers (indices in tree_list eg) read
        self.trees_read = []

        # reply pings; ie which posts of this agent have unreviewed replies
        # associated with them; a reply will be specified by (tree #,
        # level #, key #)--ie that key # in the level # dictionary in that tree
        self.reply_pings = []

        self.curr_obs = None    # for saving observations for current round:
        # [rc, ep, nrc, nep]

        self.curr_feedback = [None for _ in range(MR)]   # for saving polarity
        # feedback; this is a per-round feedback tuple, based on the posts
        # the agent has read this round (at least ri and replies), of type
        # (polarity, number_read)

    def __str__(self):
        tmp_lst = [self.id, self.polarity, self.base_polarity, self.spread,
                   self.expressive_value, self.inter_belong,
                   self.need_to_influence, self.policing, self.altruism,
                   self.TM]
        return "agent info: " + " ".join([str(xx) for xx in tmp_lst])

    def get_id(self):
        return self.id

    def get_polarity(self):
        return self.polarity

    def get_base_polarity(self):
        return self.base_polarity

    def get_spread(self):
        return self.spread

    def get_reply_pings(self):
        return self.reply_pings

    def get_net_votes(self):
        # returns total net votes (upvotes-downvotes) throughout life of the
        # agent
        return sum(self.extrinsic_points)

    def set_ep(self, tup, rn):
        # set extrinsic points, upvotes-downvotes, for round number rn
        self.extrinsic_points[rn] = tup[0]
        self.downvotes[rn] = tup[1]

    def set_observations(self,rn):
        # for computing the main observables for a given round--
        # rc, ep, nrc, nep;
        # needs to be called each round;
        # rn = round number; this is OK with round number = 0
        if rn > 0:
            rep_cts = self.reply_counts[:rn]
            ext_pts = self.extrinsic_points[:rn]
            rc = sum(rep_cts)
            ep = sum(ext_pts)
            len_bak = min(len(rep_cts),self.obs_lookback)
            nrc = sum(rep_cts[-len_bak:])
            len_bak = min(len(ext_pts),self.obs_lookback)
            nep = sum(ext_pts[-len_bak:])
        else:  # this has been called for the 0th round (at beginning),
            # so all scores must be 0
            rc, ep, nrc, nep = 0, 0.0, 0, 0.0
        self.curr_obs = [rc, ep, nrc, nep]
        return [rc, ep, nrc, nep]

    def set_total_reward(self,round_num,sum_scs):
        # for determining (raw) total reward, this round; expects sum_scs =
        # sums over all existing agents for rc, ep, nrc, nep, to balance
        # the total reward score between replies and extrinsic points;
        # NOTE, this assumes / requires set_observations to have been called
        # for this round, prior to executing this function;
        # needs to be called each round
        if sum_scs[0] == 0: # if no replies at all (historically), just use
            # extrinsic points
            val = self.curr_obs[1]
        else:
            tot_ratio = sum_scs[1] / sum_scs[0]  # sum ep / sum rc
            val = (self.curr_obs[1] + tot_ratio*self.curr_obs[0])/2
        self.tr_total[round_num] = val
        if sum_scs[2] == 0: # if no replies on lookback, use extrinsic pts
            val = self.curr_obs[3]
        else:
            trl_ratio = sum_scs[3] / sum_scs[2]  # sum nep / sum nrc
            val = (self.curr_obs[3] + trl_ratio*self.curr_obs[2])/2
        self.tr_trail[round_num] = val

    def set_propensities(self, max_scs):
        # for determining propensities, this round; expects max_scs = max
        # values this round for rc, ep, nrc, nep;
        # NOTE, this assumes / expects set_observations to have been called
        # for this round already;
        # needs to be called each round IF this agent is active (at least
        # does reading);
        # it is expected coming into this that max_scs are each >= 0.0
        norm_scores = np.zeros(4)
        for ii in range(len(norm_scores)):
            if max_scs[ii] > 0:
                # max(...,0.0) is used since unpopular agents can have negative
                # values in curr_obs for extrinsic points
                norm_scores[ii] = max(self.curr_obs[ii],0.0) / max_scs[ii]
            else:
                norm_scores[ii] = 0.0
            # this puts all the scores,
            # rc, ep, nrc, nep into the range [0,1]
        norm_tr = sum(self.expressive_value*norm_scores +
                self.inter_belong*norm_scores +
                self.need_to_influence*norm_scores)/12
        self.comment_propensity = self.comp_comm(norm_tr)
        self.general_reading = self.comp_genr(norm_tr)
        self.general_voting = self.comp_genv(norm_tr)

    def set_feedback(self, rn, tup):
        # expects round num (0,1,2,...), and a tuple of type
        # (average polarity, number of posts read for feedback)
        self.curr_feedback[rn] = tup

    def comp_feedback(self, rn):
        # computes polarity feedback effect, based on curr_feedback value(s);
        # rn=round number, typically the round the simulation is currently on
        ag_pl = self.polarity   # polarity of this agent
        tup = self.curr_feedback[rn]
        if tup is None:
            raise ValueError  # curr_feedback needs to be filled via
            # set_feedback for this round number before this function is
            # called
        if tup is None:  # softer error check (error raising may be best)
            # nothing to do--no feedback tuple for this round
            return
        fb_pl = tup[0]  # polarity of read-posts (average)
        num_red = tup[1]    # number of read posts
        new_pol = (type(self).SENSITIVITY*(min([num_red,100])/100)*(
                fb_pl-ag_pl) + ag_pl)
        self.polarity = new_pol

    def comp_comm(self, norm_tr):
        # compute comment_propensity; expects normalized total rewards for
        # this round (norm_tr in [0,1]); since TM is also in [0,1] this has a
        # max of 5.75--so can normalize this to be in [0,1]
        return (self.need_to_influence*(1+abs(self.polarity)/2) +
                self.expressive_value*(1+abs(self.polarity)/4) +
                self.inter_belong +
                abs(self.polarity)+
                self.TM*norm_tr) / 5.75

    def comp_genr(self, norm_tr):
        # compute general_reading propensity; expects normalized total
        # rewards for this round (norm_tr in [0,1]); since TM is also in [0,1]
        # this has a max of 5.0--can normalize this accordingly to be in [0,1]
        return (self.need_to_influence+self.expressive_value+
                self.inter_belong+abs(self.polarity)+
                self.TM*norm_tr) / 5.0

    def comp_genv(self, norm_tr):
        # compute general_voting propensity; expects normalized total rewards,
        # norm_tr in [0,1]; since TM is in [0,1], this has max of 6, and can
        # be normalized to lie in [0,1]
        return (self.need_to_influence+self.expressive_value+
                self.inter_belong+abs(self.polarity)+
                self.TM*norm_tr+
                self.policing) / 6.0

    def add_reply_ping(self,coords,rn):
        # this expects the address of a reply, in eg format
        # [tree #, level # (0=root), key #];
        # this also increments the reply count for this round
        self.reply_pings.append(coords)
        self.reply_counts[rn] += 1 # update number of replies received for
        # this round (rn=round number)

    def remove_reply_pings(self,rem):
        # expects list of indices in reply_pings to remove
        self.reply_pings = [self.reply_pings[ii]
                    for ii in range(len(self.reply_pings))
                            if ii not in rem]

    def get_trees_read(self):
        return self.trees_read

    def add_trees_read(self, tr):
        # update list of trees read with iterable / list tr
        if tr is None:
            raise ValueError
        self.trees_read = self.trees_read + [ix for ix in tr
                                if ix not in self.trees_read]

    def vote(self,pol):
        # cast a vote (up vote / down vote) for a post of polarity pol
        # (in [-1,1])
        vis = 1.0   # take "strict" end (narrowest) for polarity (could be
        # adjusted)
        sgm = np.exp(2*(1-vis)*np.log(1.5)+np.log(self.spread))  # std dev of
        # Gaussian, [0,1] scaling

        # NOTE through these voting methods, agent "spread" does not make
        # much of a difference through its ranges, until spread gets quite
        # small--ie close to 0.025 (eg) (vs 0.05 or 0.075)

        # the usual:
        #rnd_drw = stats.norm(0.0, sgm).rvs()  # random draw, in [0,1] scaling
        #rr = 2*abs(rnd_drw)+VOT_RAD

        # compensating for "edge" effects / clipping wrt polars; this also
        # gives more weight to spread differences (and VOT_RAD should be
        # reduced accordingly)
        rnd_drw = stats.norm(0.0, sgm).rvs()  # typical out to +/-0.12
        # ~maxish, |0.08| avg, with spread ca 0.05
        rr = (4 * abs(rnd_drw) + VOT_RAD) * (0.75 + 0.25 * abs(self.polarity))

        # check if polarity of post is outside the allowed radius around agent
        # polarity:
        if pol<self.polarity-rr or self.polarity+rr<pol:
            vt = -1  # downvote
        else:
            vt = 1   # upvote
        #print("result of vote, agent_pol=%s; post_pol=%s: %s" %
        #      (self.polarity,pol,vt))   # DEBUG
        return vt

    def comment(self):
        # returns polarity, in [-1,1] in response to a post
        # this could purely be based on the agent, and not on the post the
        # the comment is in reply to
        vis = 1.0   # take "strict" end (narrowest) for polarity (could be
        # adjusted)
        sgm = np.exp(2*(1-vis)*np.log(1.5)+np.log(self.spread))  # std dev of
        # Gaussian, [0,1] scaling
        rnd_drw = stats.norm(0.0, sgm).rvs()  # random draw, in [0,1] scaling
        cpl = self.polarity+2*rnd_drw  # unclipped polarity, [-1,1] scaling
        if cpl < -1.0:
            cpl = 1.0
        elif cpl > 1.0:
            cpl = 1.0
        return cpl      # will be in [-1,1]


class Post:

    # class constants, for custom log(vis) when vis is low
    # eg for a post with visibility <-5, use 1/16, for [-5,-1),
    # use 1/8; for 0, use 1/4; for 1 use 1/2; for 2 use 3/4
    LGV_LMS = np.array([-5,-1,0,1,2])  # mutable--
    # to use np.argmax
    LGV_VLS = np.exp([1/16,1/8,1/4,1/2,3/4])  # NOTE: all values in inner list
    # should be bounded away from and above 0.0 (so that log_vis is always >
    # 0.0)

    def __init__(self, pol, chl = None, par = None):
        self.polarity = pol  # polarity score for the node, in [-1,1]
        self.parent = par   # integer for parent node / key; note, this should
        # be None if the post is the root / OP in the tree
        self.children = chl     # list of children (integers); note this can
        # either be None, or a list of length 1 or more; ie these are keys
        # in a tree dictionary (for one level below where this post is)
        self.updn = [0.0,0.0]   # [up votes, down votes]
        self.vote_net = 0     # up votes - down votes
        self.log_vis = self.make_log_vis(self.vote_net)  # custom log(vis) value
        # for the post; updated internally through add_upvotes and
        # add_downvotes; this will always be > 0.0
        self.agent_id = None   # which agent (integer ID) wrote the post; note
        # it's possible posts are ~unauthored / anonymous (as possibly roots
        # / OPs of trees)
        self.curation = (None,None)     # curation score, possibly used in Tree
        # class to reflect a post's localized averaged polarity and visibility

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def add_child(self,key_val):
        # adds another child link to this node; this refers to the key value
        # of the next level's dictionary in whatever tree this post belongs to
        if self.children is None: # no children yet; create a list, and put
            # first child in it
            self.children = [key_val]
        else:
            if self.children is not None and key_val in self.children:
                raise ValueError    # make sure key is not already taken
            self.children.append(key_val)

    def make_log_vis(self,vis):
        # given a visibility value (ie upvotes-downvotes), convert it into
        # appropriate value (for vis large enough, log(vis))
        if vis <= max(type(self).LGV_LMS):
            ix = np.argmax(type(self).LGV_LMS>=vis)
            val = type(self).LGV_VLS[ix]
        else:
            val = vis
        return np.log(val)

    def add_upvotes(self,up):
        self.updn[0] += up
        self.vote_net += up
        self.log_vis = self.make_log_vis(self.vote_net)

    def add_downvotes(self,dn):
        self.updn[1] += dn
        self.vote_net -= dn
        self.log_vis = self.make_log_vis(self.vote_net)

    def get_agent_id(self):
        return self.agent_id

    def get_polarity(self):
        return self.polarity

    def get_net_votes(self):
        # return count of up votes minus down votes
        return self.vote_net

    def get_log_vis(self):
        return self.log_vis

    def get_updn(self):
        return self.updn[:]

    def get_curation(self):
        return self.curation

    def set_agent_id(self, aid):
        # expects an (integer) id for agent who authored the post
        self.agent_id = aid

    def set_curation(self,tup):
        self.curation = (tup[0],tup[1])


class Tree:

    # a simple tree structure; consists of a list of dictionaries;
    # the first dictionary in the list is the root post (OP);
    # the second dictionary in the list is all replies to the OP, etc.;
    # dictionary keys are integers (prlly starting at 0):
    # {0: post; 1: post; ...} with "post" objects of Post class;
    # a post with a parent of None will mean it's the root of the tree

    # class constants
    TR_SZ = 50      # trees with post totals > TR_SZ will have visibility
    # chain cutoff dependent on post with maximal net votes; trees with less
    # will have cutoff based on 0 net votes (ie negative net votes are
    # penalized)
    CO_PP = 0.1     # proportion of maximal net votes taken to calculate
    # large tree visibility chain cutoff (ie multiply maximal net votes by
    # this to obtain cutoff value)
    AT_FC = 1/2     # attenuation factor; how much to attenuate curation
    # visibility values for post and all posts downstream if votes below
    # threshold
    SIG = MODULE_SIG      # how picky to be for curation re polarization
    # matching;
    # ca 0.15 is quite picky, 0.25 is moderately picky, 0.5 is somewhat loose,
    # 1.0 (and higher) is kind of like no polarity curation at all

    def __init__(self, in_post = None):
        self.dict_list = []
        self.vot_lst = []       # list of dictionaries, matching keys to
        # dict_list, for ease in tracking post net votes
        self.cur_res_lst = []       # for storing results of curation
        # function(s); each sublist is of form [(lvl,key), vis-prb score]
        # num_posts--track total number of posts in the tree, including root/OP
        # updn--total up votes and down votes over all posts in the tree--
        #   [up votes, down votes]
        self.cur_oth_lst = []       # for storing all the other results of the
        # curation process--this has same structure as cur_res_lst, but holds
        # all other posts in the tree that were not put in cur_res_lst
        if in_post is not None:
            if in_post.get_parent() is not None:
                raise ValueError    # first level is root post, which should
                # have Null for parent
            self.dict_list.append({0: in_post})  # in_post should be a Post
            # object
            self.num_posts = 1
            tmp = in_post.get_updn()
            self.updn = [tmp[0],tmp[1]]
            nt_vt = tmp[0]-tmp[1]
            self.vot_lst.append({0:nt_vt})
        else:   # account for this case, if wanted (do not really have a
            # class method for adding posts once the tree is created with
            # a missing root)
            raise ValueError
            self.dict_list.append({})  # ie root node does not yet exist
            self.vot_lst.append({})
            self.num_posts = 0
            self.updn = [0,0]
        self.vote_trsh = None   # vote threshold for curation

    def __str__(self):
        # print tree
        st = ""
        for lvl in range(len(self.dict_list)):
            st += "Level %s: \n" % lvl
            for key in self.dict_list[lvl]:
                post = self.dict_list[lvl][key]
                aid = post.get_agent_id()
                if aid is None:
                    aid = "None"
                else:
                    aid = str(aid)
                st += (str(key)+": vt: %s" % str(self.vot_lst[lvl][key]) +
                "; pol: %s" % post.get_polarity() + "; aid: %s" %
                       aid + "; chl: "
                                                                    "%s" %
                       post.get_children() + " || ")
            st += "\n"
        return st

    def vote(self, level, post_key, agt):
        # NOTE: this has been modified from base_version_25 and earlier, to
        # "turn off" downvoting
        # ~~~
        # level is the level of the post in the tree; 0=root;
        # post_key is that level's (integer) dictionary key
        # agt is an agent object, used to decide what vote to produce (up/dn);
        # returns id of agent who authored the post
        if level+1 > len(self.dict_list):
            raise ValueError
        post = self.dict_list[level][post_key] # fetches post object
        pol = post.get_polarity()
        pst_vis = post.get_net_votes()
        val = agt.vote(pol)   # -1=downvote, +1=upvote
        if val>0:
            post.add_upvotes(1)
            self.updn[0] += 1
        else:
            pass   # mod for downvote suppression
            #post.add_downvotes(1)
            #self.updn[1] += 1
        if val > 0:
            self.vot_lst[level][post_key] += val  # update post net vote tracker
        # curation list for this post after vote / visibility change; modded
        # for downvote suppression
        pst_aid = post.get_agent_id()
        if pst_aid == agt.get_id():
            print("agent cant vote on their own posts")
            breakpoint()
            raise ValueError
        return (post.get_agent_id(),
                val if val > 0.0 else 0.0, pol)   # returning these values is
        # used for tracking dictionaries, etc in caller; modification
        # here--if vote value val is -1, return 0.0 (downvote "suppression")

    def reply(self, level, post_key, post_child):
        # adds a reply to the given post;
        # level is the level of the post in the tree; 0=root;
        # post_key is that level's (integer) dictionary key;
        # this assumes post_child has polarity set, parent index # set
        # (ie to post_key), and (really) children set to None
        if post_child.get_parent() != post_key:
            raise ValueError
        if level+1 > len(self.dict_list): # the level of the post being
            # replied to has to exist in the tree
            raise ValueError
        elif level+2 > len(self.dict_list):
            # the next level's dictionary has not yet been created, so create
            self.dict_list.append({})
            self.vot_lst.append({})
        post_parent = self.dict_list[level][post_key]  # post object of
        # post being replied to
        key_val = len(self.dict_list[level+1])  # new key value in next-lower
        # level of tree
        post_parent.add_child(key_val)
        self.dict_list[level+1][key_val] = post_child
        self.num_posts += 1
        pst_vis = post_child.get_net_votes()
        self.vot_lst[level+1][key_val] = pst_vis
        return level+1,key_val

    def curation_score(self, ag_id, pol):
        # this goes through dict_list, tailoring
        # curation scores by their visibility and agreement with input polarity
        # ag_id--id of agent (posts that have the same agent id will be
        # ~excluded)
        # pol--polarity to curate the posts on
        # this returns a copy of internal cur_res_lst, amounting to list of
        # sublists that designate post coords, and give overall visibility
        # score: [(level,key), vis_scr], also returns a copy of internal
        # cur_oth_lst, the list of all other posts in tree not in cur_lst--
        # and neither lists will contain posts authored by the agent themselves

        # DEBUG
        '''
        print("starting curation for agent %s, around polarity %s" %
              (ag_id,pol))
        print("vot_lst: %s" % self.vot_lst)
        '''

        lvl = 0
        pst_key = 0
        if len(self.dict_list)==0 or pst_key not in self.dict_list[0]:
            print("empty tree found(?)")
            return []       # nothing to curate; return empty list
        else:
            att_fac = 1.0   # attenuation factor; changes at and downstream
            # of low-popularity posts
            nrm_cst = 0.0   # normalization constant--sum of vis-prob post
            # scores
            self.calc_vote_trsh()  # update threshold, dependent on current
            # state of tree

            # DEBUG
            #print("vote threshold: %s" % self.vote_trsh)

            self.cur_res_lst = []   # clear the list, to store posts filtered
            # for curation
            self.cur_oth_lst = []   # clear the "other" posts storage list,
            # to hold posts that were not pulled for cur_res_lst
            tot_nrm = self.curate_post(lvl,pst_key,att_fac,nrm_cst,ag_id,pol)

            # normalize the probability-visibility value in curation list of
            # posts
            '''
            self.cur_res_lst = [[sbl[0],sbl[1]/tot_nrm] for sbl in
                                self.cur_res_lst]   # this is prlly not necc.--
                                # depends on how caller handles it
            '''

            return copy.deepcopy(self.cur_res_lst), copy.deepcopy(
                self.cur_oth_lst)

    def curate_post(self,lv,pk,attn,nc,ag_id,pol):
        # recursive function, to create visibility probability scores for
        # posts, and to select posts for polarity matching;
        # lv, pk are level and post key in the dict_list;
        # nc is the normalization constant accumulator;
        # this modifies class member cur_res_lst, which has sublists
        # [(level,key),un-normalized attn*lgv value]
        post = self.dict_list[lv][pk]
        net_vts = post.get_net_votes()
        # check net votes for the post, and if below threshold, update
        # attenuation factor
        if self.vote_trsh > 0:
            # consider splitting attenuation factors over threshold vote_trsh,
            # and 0
            if net_vts < self.vote_trsh:
                if net_vts < 0:
                    attn *= type(self).AT_FC
                else:
                    attn *= np.sqrt(type(self).AT_FC)
        else:
            # apply just vote_trsh for attenuation factor
            if net_vts < self.vote_trsh:
                attn *= type(self).AT_FC
        # determine un-normalized probability-visibility score
        pr_vs = attn*post.get_log_vis()
        nc += pr_vs
        # curate for polarity match
        pol_dff = abs(pol-post.get_polarity())
        if self.gaussian_select(pol_dff) and post.get_agent_id() != ag_id:
            # (ensures post not written by this agent)
            self.cur_res_lst.append([(lv,pk),pr_vs])
        elif post.get_agent_id() != ag_id:
            self.cur_oth_lst.append([(lv,pk),pr_vs])

        # DEBUG
        #print("for lv,ky=%s,%s: attn: %s; pr_vs: %s" % (lv,pk,attn,pr_vs))

        # loop through children of post
        chl_lst = post.get_children()  # a list of integers (None if none)--
        # keys in dict_list one level down from this post's level
        if chl_lst is not None:
            for key in chl_lst:  # call function recursively
                nc += self.curate_post(lv+1,key,attn,nc,ag_id,pol)
        return nc

    def calc_vote_trsh(self):
        # calculates where probability visibility net votes threshold is set;
        # posts with vote amounts < this value and posts downstream get a
        # reduction factor to their log(vs)
        max_vts = max([self.vot_lst[ii][key] for ii in range(len(self.vot_lst))
                       for key in self.vot_lst[ii]])  # max net vote score
        # over all posts in tree
        if max_vts < 0:     # handle this?
            pass
        if max_vts > type(self).TR_SZ:
            self.vote_trsh = type(self).CO_PP*max_vts
        else:
            self.vote_trsh = 0

    def gaussian_select(self,xx):
        # uses a "Gaussian shoulder" approach to decide whether to accept or
        # reject a post based on xx=|agent polarity - post polarity|;
        # this uses class attribute SIG, which is the standard deviation for
        # Gaussian--and note the Gaussian acts at half scale (so differences
        # are divided by 2 prior to going to Gaussian
        # returns True if post was accepted, False if not accepted
        tst_val = self.gaussian(xx/2,0.0,type(self).SIG)  # a value in [0,1]
        if random.random() < tst_val:
            return True
        else:
            return False

    def gaussian(self, xx, mu, sg):
        # helper function for gaussian_metric;
        # un-normalized (unit height) Gaussian with mean mu, std dev sig,
        # given input value xx
        return np.exp(-(xx - mu) ** 2 / (2 * sg ** 2))

    def get_post(self,lvl,key):
        # given level in tree (0,1,2..., w/ 0=root), and key (0,1,2,...)
        # for that level's dictionary, return respective post
        return self.dict_list[lvl][key]

    def get_complexity(self):
        # returns value(s) associated with the complexity of the tree;
        # eg return (number of levels, number of posts over all the levels);
        # will be (1,1) for a starting tree
        return (len(self.dict_list), sum([len(x) for x in self.dict_list]))

    def get_level(self,lvl):
        # given tree level, return whole level dictionary (note this is not
        # really ~safe re tree internal data
        return self.dict_list[lvl]

    def get_popularity(self,net=True):
        # determines the total upvote-downvote score, over all posts in the tree
        # (ie just a sum of this value over all posts);
        # will be 0 for a starting tree
        updn = [0,0]
        for lvl in range(len(self.dict_list)):
            for key in self.dict_list[lvl]:
                tmp_ud = self.dict_list[lvl][key].get_updn()
                updn[0] += tmp_ud[0]
                updn[1] += tmp_ud[1]
        if net:
            return updn[0]-updn[1]
        else:
            return updn[:]

    @classmethod
    def change_sig(cls,new_sig):
        cls.SIG = new_sig


class PostContainer:

    # convenience class, for consolidating posts and their scoring tuples
    # NOTE: this could make post_tup_score its own method
    def __init__(self, cor=None, povs=None, lcs=None, pos=None, neg=None,
                 po=None):
        self.coord = cor  # coordinate of post: [tree #, level #, key]
        self.raw_povs = povs  # (polarity, log(vis)) for the individual post
        # at coordinates coord
        self.raw_loc_scores = lcs  # locality scores: (polarity, log(vis))
        self.nrm_povs = None    # for normalized visibility (pol,log(vis))
        # tuples, based off individual post
        self.nrm_loc_scores = None  # locality score tuples, with normalized vis
        self.pos = pos  # scoring value (float) assigned based on polar
        # agreement
        self.neg = neg  # scoring value (float) assigned based on polar
        # opposition
        self.police = po    # scoring value (float) assigned specifically
        # for policing

    def norm_lc_sc(self,cc):
        # normalizes visibility portion of raw locality score, based on cc;
        # visibility scores < 0 will be normalized to 0
        if cc < 0:
            raise ValueError
        self.nrm_loc_scores = (self.raw_loc_scores[0],
                               max(cc*self.raw_loc_scores[1],0))

    def norm_povs(self,cc):
        # normalizes visibility portion of individual post score, based on cc;
        # visibility scores < 0 will be normalized to 0
        if cc < 0:
            raise ValueError
        self.nrm_povs = (self.raw_povs[0],max(cc*self.raw_povs[1],0))


def gaussian(xx,mu,sg):
    # helper function for gaussian_metric;
    # un-normalized (unit height) Gaussian with mean men, std dev sig,
    # given input value xx
    return np.exp(-(xx-mu)**2/(2*sg**2))


def gaussian_metric(xx,spd,vis):
    # uses a "Gaussian shoulder" approach to return a metric value on the
    # separation distance xx between the Gaussian peak (at eg the agent's
    # polarity) and the post polarity
    # xx--distance between agent and post polarity--should be >0, and on scale [
    # -1,1]
    # spd--internal parameter for Gaussian spread width; eg in [0.025,0.075]
    # vis--normalized visibility score--visibility score adjusts Gaussian shape;
    # returns a score value in [0,1]
    sgm = np.exp(2*(1-vis)*np.log(1.5)+np.log(spd))  # std dev of Gaussian,
    # [0,1] scaling
    return (1+vis)*gaussian(xx/2, 0.0, sgm)/2


def post_tup_score(in_pol, in_spd, ploc_pol, ploc_vis):
    # (note, consider absorbing this into PostContainer class?)
    # computes the score to assign to a given post locality tuple;
    # this expects:
    # in_pol--this is the polarity value to filter on/around, in (-1,1)
    # in_spd--this is the spread parameter for the distribution for
    # input polarity; an internal parameter, custom for eg Gaussian
    # ploc_pol--locality polarity score of the post, in [-1,1]
    # ploc_vis--locality visibility score of the post, *normalized* in [0,1]
    # see eg reddit_sim_partisan.odt, "(*) a method for scoring";
    # this returns a score, in [0,1]
    xx = abs(in_pol-ploc_pol)
    return gaussian_metric(xx,in_spd,ploc_vis)


def post_police_score(in_pol, in_spd, ppo, pvs):
    # function to score based on individual post polarity and visibility (vs
    # values averaged over parent and children, as with locality scores); used
    # for policing, where agent polarity is typically "flipped"
    # this expects:
    # in_pol--this is the polarity value to filter on/around, in (-1,1)
    # in_spd--this is the spread parameter for the distribution for
    # input polarity; an internal parameter, custom for eg Gaussian
    # ppo--polarity for single post
    # pvs--visibility for single post, *normalized*
    # see eg reddit_sim_partisan.odt, "(*) a method for scoring";
    # this returns a score in [0,1]
    xx = abs(in_pol-ppo)
    return gaussian_metric(xx,in_spd,1-pvs)
    # for policing--lower visibility (favored) will now have narrower polarity
    # conditions;  have "flipped" visibility score --vis->1-vis; this favors
    # lower visibility posts, where the effect of a downvote should be greater


def locality_score(coord, tree_list):
    # for computing locality score of a given input of post coordinates
    # coord--[tree #, level #, post key value]
    # tree_list--the list holding the simulation tree objects
    # output: 2 tuples, each of type (polarity score, visibility score),
    # the 1st being the individual post's values, the 2nd tuple is the
    # locality score; note both visibility scores (individual post, and
    # locality) are really log(vis) scores (kept track of in Post's log_vis
    # attribute), and are guaranteed to be > 0.0
    loc_lst = []  # for storing locality score tuples, (polarity,
    # visibility) pull visibility (log(vis)) and polarity scores for each
    # "surrounding" post
    # for the post itself:
    tree = tree_list[coord[0]]
    tmp_post = tree.get_post(coord[1], coord[2])
    po, vs = tmp_post.get_polarity(), tmp_post.get_log_vis()
    loc_lst.append((po,vs))
    # parent:
    par_key = tmp_post.get_parent()
    if par_key is not None:
        par_post = tree.get_post(coord[1] - 1, par_key)
        loc_lst.append((par_post.get_polarity(), par_post.get_log_vis()))
    # children:
    '''
    chl_lst = tmp_post.get_children()
    if chl_lst is not None:
        for chl_key in chl_lst:
            chl_post = tree.get_post(coord[1] + 1, chl_key)
            loc_lst.append(
                (chl_post.get_polarity(), chl_post.get_log_vis()))
    '''
    # process loc_lst tuples for overall polarity and visibility score
    sum_of_vis = sum([tup[1] for tup in loc_lst])

    # compute visibility-weighted average of polarity; note this assumes
    # the log_vis values are all > 0.0 (strictly bounded away from it)
    pol_score = (sum([tup[0]*tup[1] for tup in loc_lst]) /
                     sum([tup[1] for tup in loc_lst]))
    vis_score = sum_of_vis / len(loc_lst)
    return (po,vs), (pol_score, vis_score)


def reading_choices(pc_cur_skim, pc_oth_skim, agent, aph=None):
    # consolidated function for parsing skim posts into what the agent
    # reads, in each of 3 baskets (lists) of PC objects: ri_posts, po_posts,
    # al_posts (--returns these 3 PC object lists)
    # pc_cur_skim--posts for ri bucket, curated for polarity ~alignment with
    # agent (use "pos" scores)
    # pc_oth_skim--posts for po and al buckets; randomly selected from the
    # tree(s)
    # note, as of latest the number of posts in cur and oth should be approx
    # equal (we can draw a higher proportion from curated cur, since it's
    # already been ~filtered somewhat, and a lower proportion from oth, for
    # filtering purposes, so that overall get about 2:1 between ri posts and
    # po/al posts (combined))
    # agent--the agent object
    # aph--if not None, this is a debugging tracking dictionary, by agent id,
    # for keeping track of total polarity choices (skim and read) for ri

    # DEBUG
    #agd = 4
    #if agent.get_id()==agd:
    #    print("for agent %s, len cur=%s; len oth=%s" % (agd,
    #            len(pc_cur_skim),len(pc_oth_skim)))

    ri_posts, po_posts, al_posts = [], [], []
    if len(pc_cur_skim) < 2 and len(pc_oth_skim) < 2:
        # this could be handled different ways; if < 2 posts, the agent can't
        # really make a choice, so eg kick back to caller
        if VERBOSE:
            print("< 2 skim posts for agent %s" % agent.get_id())
        return ri_posts, po_posts, al_posts
    # how many posts to draw from pc_..._skims total:
    prp_skm = (0.3+0.3*agent.general_reading, 0.3*agent.policing+
               0.15*agent.altruism)  # for random number generation
    num_ri = sum(np.array([random.random() for _ in
                            range(len(pc_cur_skim))]) <= prp_skm[0])
    num_po_al = sum(np.array([random.random() for _ in
                            range(len(pc_oth_skim))]) <= prp_skm[1])
    if num_ri < 1 and num_po_al < 1:
        # DEBUG
        #if agent.get_id()==agd:
        #    print("too few posts first pass")
        return ri_posts, po_posts, al_posts
    # num_ri and num_po_al should be in ratio as set in prp_skm--eg
    # approx. 0.3+0.3*general_reading : 0.3*policing+0.15*altriusm;
    # because pc_cur_skim and pc_oth_skim may not be the same size
    # (especially if curation is strict), can "correct" drift from the
    # desired proportion, by adjusting the appropriate of the two values
    # downward
    if num_ri>0.5 and num_po_al>0.5:
        num_ri, num_po_al = prop_correct(num_ri, num_po_al, (prp_skm[0],
                                            prp_skm[1]))

    num_ri = int(np.round(num_ri))
    num_po_al = int(np.round(num_po_al))

    # determine pmf for post-type draws (po, or al):
    tmp_arr = np.array([POL_AGNT*agent.policing,
                        ALT_AGNT*agent.altruism])
    pmf_drw = tmp_arr / sum(tmp_arr)
    cdf_drw = np.cumsum(pmf_drw)
    cdf_drw[-1] = 1.0   # to be sure covers up to 1.0
    # do num_drw random draws, and save counts in ri, po, al
    ct_ar = np.zeros(2)
    for jj in range(num_po_al):
        # which reading bucket to select a post for (in order: po, al)
        tmp = random.random()
        ix = np.where(tmp<=cdf_drw)[0][0]    # lowest index in cdf_drw
        # satisfying the inequality
        ct_ar[ix] += 1
    # consolidate all the resulting post counts into a single array--
    # [num_ri, num_po, num_al]
    ct_ar = np.array([num_ri, ct_ar[0], ct_ar[1]])

    # limit reading posts to maximum values (so agents can't read arbitrarily
    # many)
    mx_vs = [NRI_BASE+random.random()*NRI_RAND+agent.general_reading*NRI_AGNT,
             POL_AGNT*agent.policing, ALT_AGNT*agent.altruism]
    ct_ar = np.array([min(tup) for tup in zip(ct_ar,mx_vs)]).astype(int)

    # ri readings
    if ct_ar[0] > 0:
        ix_ls = list(range(len(pc_cur_skim)))  # which indices in pc_cur_skim
        # are available
        num_ri = ct_ar[0]
        agent_pos_scs = [(x.pos,ii) for ii, x in
                  enumerate(pc_cur_skim) if ii in ix_ls]  # pull just "pos""
        # scores;

        prb_dst = np.array([tup[0] for tup in agent_pos_scs])
        if not RC_BY_RANK and sum(prb_dst>0.0)>=num_ri:
            if num_ri == 0:
                rnd_ixs = []
            else:
                # choose by score-based pmf:
                gen = np.random.default_rng()  # generator for distribution-based
                # random selection
                prb_dst = prb_dst/sum(prb_dst)
                rnd_ixs = gen.choice(ix_ls, num_ri, replace=False,
                           p=prb_dst)
        else:
            # choose by rank:
            tup_rnk = sorted(agent_pos_scs,key=lambda x: x[0],reverse=True)
            rnd_ixs = [tup[1] for tup in tup_rnk[:num_ri]]

        ri_posts = [pc_cur_skim[ix] for ix in rnd_ixs]
        for pst in ri_posts:   # DEBUG:
            tup = pst.nrm_loc_scores   # (pol, nrmlzed vis)
            lst = aph[agent.get_id()]  # (mutable) list of tuples
            ix = lst.index(tup)     # if this throws an error, something is
            # wrong
            tmp = lst[ix][1]
            if tmp==0:
                tmp=-1e-12  # ~kludge--adjust to small negative value
            else:
                tmp=-tmp
            lst[ix] = (lst[ix][0],tmp)  # "flipping" visibility value
            # (in [0,1]) signals this post was read (for debugging)
    ix_ls = list(range(len(pc_oth_skim)))  # which indices in pc_oth_skim
    # are available;
    # po readings
    if ct_ar[1] > 0:
        num_po = ct_ar[1]
        agent_neg_scs = [(x.police,ii) for ii, x in
                  enumerate(pc_oth_skim) if ii in ix_ls]  # use the
        # individual-post police scores from the PC object;

        prb_dst = np.array([tup[0] for tup in agent_neg_scs])
        if not RC_BY_RANK and sum(prb_dst>0.0)>=num_po:
            if num_po == 0:
                rnd_ixs = []
            else:
                # choose by score-based pmf:
                gen = np.random.default_rng()  # generator for distribution-based
                # random selection
                prb_dst = prb_dst / sum(prb_dst)
                rnd_ixs = gen.choice(ix_ls, num_po, replace=False,
                                     p=prb_dst)
        else:
            # by rank:
            tup_rnk = sorted(agent_neg_scs,key=lambda x: x[0],reverse=True)
            rnd_ixs = [tup[1] for tup in tup_rnk[:num_po]]

        po_posts = [pc_oth_skim[ix] for ix in rnd_ixs]
        ix_ls = [ix for ix in ix_ls if ix not in rnd_ixs]
    # al readings
    if ct_ar[2] > 0:
        num_al = ct_ar[2]
        agent_neg_scs = [(x.neg,ii) for ii, x in
                  enumerate(pc_oth_skim) if ii in ix_ls]

        prb_dst = np.array([tup[0] for tup in agent_neg_scs])
        if not RC_BY_RANK and sum(prb_dst>0.0)>=num_al:
            if num_al == 0:
                rnd_ixs = []
            else:
                # choose by score-based pmf:
                gen = np.random.default_rng()  # generator for distribution-based
                # random selection
                prb_dst = prb_dst / sum(prb_dst)
                rnd_ixs = gen.choice(ix_ls, num_al, replace=False,
                                     p=prb_dst)
        else:
            # by rank
            tup_rnk = sorted(agent_neg_scs,key=lambda x:x[0],reverse=True)
            rnd_ixs = [tup[1] for tup in tup_rnk[:num_al]]

        al_posts = [pc_oth_skim[ix] for ix in rnd_ixs]
    # DEBUG:
    #print("for agent %s, num [ri, po, al]: %s, %s, %s" %
    #      (agent.get_id(),ct_ar[0],ct_ar[1],ct_ar[2]))

    # DEBUG (spot check specific agent #)
    #if agent.get_id()==agd:
    #    print("agent %s, tried %s ri and %s po/al; ct_ar=%s" % (agd,
    #            num_ri,num_po_al,ct_ar))

    return ri_posts, po_posts, al_posts


def rc_gen_rnd(agt_scs, num, ix_ls):
    # for posts from skim_pare list of PostContainer objects;
    # expects list of scores in agt_scs
    # expects number of random draws in num
    # expects assoc. list of values (eg list indices) to select from in ix_ls
    gen = np.random.default_rng()
    # selection will be based on a pdf derived from agent_pos_scores
    tmp_min = min(agt_scs)
    tmp_vls = np.array(agt_scs)
    if tmp_min < 0:
        tmp_vls = tmp_vls - tmp_min + 1e-10
    p_weights = tmp_vls / sum(tmp_vls)  # create discrete pmf
    rnd_ixs = gen.choice(ix_ls, num, replace=False, p=p_weights)
    return rnd_ixs


def agent_comment(pc_ls, tr_ls, agent, ag_dc, rd_nm, atr_sbl, rep_com = None):
    # for creating agent comments in a tree
    # pc_ls--list of PC objects (posts) to comment on
    # tr_ls--tree list (complete)
    # agent--agent object
    # ag_dc--agent dictionary
    # rd_nm--round number of simulation
    # atr_sbl--appropriate (ri) sublist from agent_track_dct, for debugging
    # rep_com--flag--if "reply" then attenuate comments to
    # replies deep in the tree
    for pc in pc_ls:  # go through "parent" posts to reply to
        tree = tr_ls[pc.coord[0]]
        parent_post = tree.get_post(pc.coord[1], pc.coord[2])
        # have the potential to decline to comment if the
        # reply is (eg) embedded deep in the tree:
        if rep_com is not None and rep_com=="reply":
            if pc.coord[1] <= 2:  # don't attenuate till lower in the tree--
                # suppose levels 0,1,2, don't attenuate
                prb = 1.0
            else:
                prb = np.sqrt(1 / (pc.coord[1] - 1))  # this gives probability
                # 0.707 at level 3, 0.577 at level 4, 0.5 at level 5, etc.
            if random.random() > prb:
                if VERBOSE:
                    print("reply deep in tree; declined to comment")  # DEBUG
                continue
        pol = agent.comment()  # returns polarity for new post by this
        # agent;
        # create reply post object, with parent post coordinates
        reply_post = Post(pol=pol, chl=None, par=pc.coord[2])
        reply_post.set_agent_id(agent.get_id())
        # emplace the reply post in the tree
        pst_lvl, pst_key = tree.reply(pc.coord[1], pc.coord[2], reply_post)
        post_agent_id = parent_post.get_agent_id()
        # ping the agent (of the parent post) with the reply
        if post_agent_id is not None:
            # DEBUG; if wanting to prevent agents commenting on own posts
            if post_agent_id == agent.get_id():
                breakpoint()
                raise ValueError
            ag_dc[post_agent_id].add_reply_ping([pc.coord[0], pst_lvl,
                                     pst_key], rd_nm)
            #ag_dc[post_agent_id].add_reply_ping(pc.coord[:], rd_nm)
        # DEBUG--update agent_track_dct; icc
        update_ag_tr_lst(atr_sbl, parent_post.get_polarity())


def agent_vote(pc_ls, tr_ls, agent, ep_dc, atr_sbl):
    # for an agent voting in a tree
    # pc_ls--list of PC objects (posts) to vote on
    # tr_ls--list of trees
    # agent--agent object
    # ep_dc--ep_dict (for debugging)
    # atr_sbl--sublist of agent_track_dct (for debugging)
    for pc in pc_ls:
        tree = tr_ls[pc.coord[0]]       # tree index
        pid, val, pol = tree.vote(pc.coord[1],pc.coord[2],agent)
        # coord[1] is tree lvl (0,1,...); coord[2] is lvl key for post;
        # pid is agent id of post being voted on
        if pid is not None:  # ie eg not a root / anonymous OP of tree
            tup = ep_dc[pid]
            aa = tup[0] + val  # net votes
            bb = tup[1] + (1.0 if val < 0.0 else 0.0)  # count downvotes
            ep_dc[pid] = (aa, bb)
        # DEBUG--update agent_track_dct; icv
        update_ag_tr_lst(atr_sbl, pol)


def create_tree(pol = None):
    # creates a new tree; suppose it makes an anonymous OP, and that's it
    if pol is None:
        pol = 2*random.random()-1  # random val in [-1,1]
    post = Post(pol, chl=None, par=None)  # agent_id is None by default
    return Tree(post)


def rand_skim_choice(num_to_choose, cor_vis_lst, tree_num):
    # helper function for creating skim list(s) from Tree curation_choice
    # return lists
    # num_to_choose--how many random values to draw (without replacement)
    # cor_vis_lst--curation return list, with elements [(level,key),log(vis)]
    # tree_num--what number tree to assign to full coordinates for output
    # out_lst--returned list; has elements (tree num, level, key),
    # the result of random selection on cor_vis_lst
    out_lst = []
    gen = np.random.default_rng()
    tmp_wts = [sbl[1] for sbl in cor_vis_lst]  # un-normalized
    # visibility weights in cor_vis_lst
    p_weights = [val / sum(tmp_wts) for val in tmp_wts]
    rnd_ixs = gen.choice(len(cor_vis_lst), num_to_choose, replace=False,
                         p=p_weights)  # randomly chosen indices
    # in cur_lst, weighted by vis scores
    out_lst = [(tree_num, cor_vis_lst[ix][0][0],
                 cor_vis_lst[ix][0][1]) for ix in
               rnd_ixs]
    #out_lst = [[(tree_num,cor_vis_lst[ix][0][0],
    #             cor_vis_lst[ix][0][1]),cor_vis_lst[ix][1]] for ix in
    #           rnd_ixs]
    return out_lst


def prop_correct(num_1, num_2, prp):
    # given two values, num_ri and num_po_al (both expected >= 1), and
    # prp a tuple representing desired proportion (eg (2,1) as 2:1, wanting
    # num_1 about twice that of num_2), adjust
    # the values downward as necessary to approximate the proportion better
    # if needed
    ppn = prp[0]/prp[1]
    ppn_lo = ppn - 0.15*ppn*random.random()   # random 15% variation
    ppn_hi = ppn + 0.15*ppn*random.random()
    tst_prp = num_1/num_2
    if tst_prp > ppn_hi:
        # num_2 is limiting
        num_1 = ppn_hi*num_2
    elif tst_prp < ppn_lo:
        # num_1 is limiting
        num_2 = num_1/ppn_lo
    return num_1, num_2


def trees_avg_posts(tr_ls):
    # for list of tree objects, tr_ls, returns average number of posts per
    # tree
    return np.average(np.array([tree.get_complexity()[1] for tree in tr_ls]))


def show_trees(tr_ls):
    # for quick view / stats on the trees in a list of tree objects
    cmt_lst = [tree.get_complexity() for tree in tr_ls]  # this will be a list
    # of tuples: (tree depth, number of posts)
    avg_dpt = np.average([tup[0] for tup in cmt_lst])
    avg_ptc = np.average([tup[1] for tup in cmt_lst])
    print("average (tree depth, post count) for %s trees: %s, %s" %
          (len(tr_ls),avg_dpt,avg_ptc))


def show_tree_details(tr_ls):
    # post-level view for (selected) trees, drawn from list of tree objects
    # tr_ls
    for tree in tr_ls:
        tc = tree.get_complexity()
        if tree.get_post(0,0).updn == [0,0] and tc[0]==1:
            # tree has no upvotes/downvotes, and only the OP
            continue
        for ii in range(min(tc[0],2)): # eg look at up to 1st 2 levels
            lv_dc = tree.get_level(ii)
            lv_ar = [(lv_dc[key].get_polarity(),lv_dc[key].updn) for
                key in lv_dc]
            print("for level %s, (polarity, (upvotes,downvotes))" % ii)
            print(lv_ar)


def show_tree_votes(tr_ls, disp=True):
    # shows total upvotes and total downvotes, over all tree objects in
    # list tr_ls
    vts = np.array([0.0,0.0])     # holds upvotes, downvotes
    for tree in tr_ls:
        for tr_dc in tree.dict_list:   # go through levels in the tree
            for key in tr_dc:  # go through all posts at this level
                vts += np.array(tr_dc[key].updn)
    if disp:
        print("over all trees, (upvotes,downvotes) = (%s,%s); upvote "
              "proportion: %s"
          "" % (*tuple(vts),vts[0]/sum(vts) if sum(vts)>0 else "NA"))
    return vts


def trees_polarity_plot(tr_ls, tre_ixs = None):
    # consider a tree's polarity score distribution, over all posts, and
    # displays a stacked bar chart of the topmost-post-count trees,
    # broken down by (binned) polarity counts
    # expects a list of tree objects (tr_ls)
    # optional tre_ixs argument can contain labels for the
    # trees in tr_ls (so can keep tree labels the same, one call to the next,
    # as from a changing active trees list of indices)--ie tr_ls and tre_ixs
    # are of the same length, with tre_ixs[0] containing label for tree object
    # at tr_ls[0], and so on
    num_bns = 4 # number of bins for stacked bars
    bns = np.linspace(-1,1,num=num_bns+1)
    num_top_trs = 30     # how many of the topmost post count trees to plot

    # create a list of binned polarity counts, one for each tree in tr_ls
    frq_lst = []
    for tree in tr_ls:
        pol_lst = []
        for lvl in range(tree.get_complexity()[0]):
            lv_dc = tree.get_level(lvl)
            for key in lv_dc:
                post = lv_dc[key]
                pol_lst.append(post.get_polarity())
        frq_lst.append(np.histogram(pol_lst,bns)[0].astype(float))  # note if
        # pol_lst is empty, this will just be an array with num_bns zeros

    # sort by total post count, in descending order
    if tre_ixs is not None:
        tmp = sorted(zip(tre_ixs,frq_lst), key=lambda x: sum(x[1]),
                     reverse=True)
        tmp = tmp[:num_top_trs]
        cn_ls = [tup[0] for tup in tmp]
        srt_frl = [tup[1] for tup in tmp]
    else:
        cn_ls = None
        srt_frl = sorted(frq_lst, key=lambda x: sum(x), reverse=True)
        srt_frl = srt_frl[:num_top_trs]

    # convert to dataframe, for plotting;
    # create columns:
    col_ars = []
    for jj in range(len(bns) - 1):  # for each histogram bin
        col_ars.append([sb_ar[jj] for sb_ar in srt_frl])
    # column names:
    cl_ns = [str(x) for x in [jj + 1 for jj in range(len(col_ars))]]
    zp_ar = zip(cl_ns, col_ars)
    if cn_ls is not None:
        df = pd.DataFrame({tup[0]: tup[1] for tup in zp_ar},index=cn_ls)
    else:
        df = pd.DataFrame({tup[0]: tup[1] for tup in zp_ar})

    # plot
    cmp = plt.get_cmap('PiYG') # 'PuOr'; nope--('winter') ('BrBG')
    lmt = 1/(2*num_bns)
    colors = cmp(np.linspace(lmt,1-lmt,num_bns))
    df.plot.bar(stacked=True,color=colors)
    plt.xlabel("tree index")
    plt.ylabel("frequency bins")
    plt.title("top tree polarities (dec tot psts; incl OP)")
    plt.show()

    if tre_ixs is None:
        return frq_lst  # returned for purposes of trees_data; for it to work
        # with trees_data, cannot have "special" subset of tree_list trees,
        # but include all trees in tree_list
    else:
        return None


def trees_complexity_plot(tr_ls, tre_ixs = None):
    # consider a tree's number of posts per level, for eg 1st 5 or 10 levels;
    # displays a stacked bar chart of the topmost-post-count trees,
    # broken down by level-post counts
    # expects a list of tree objects (tr_ls)
    # optional tre_ixs argument can contain the indices / labels of the
    # trees in tr_ls (so can keep tree labels the same, one call to the next,
    # as from a changing active trees list of indices)
    num_lvs = 4 # number of levels to look at (levels 1, 2, ...; ignoring root
    # level, which is always trivially 1 post
    num_top_trs = 10     # how many of the topmost post count trees to plot

    # create a list of level-post counts, one for each tree in tr_ls
    frq_lst = []
    for tree in tr_ls:
        psc_lst = np.zeros(num_lvs)
        for lvl in range(1,min(tree.get_complexity()[0],num_lvs+1)):  # lvl will
            # go 1,2,3,...
            lv_dc = tree.get_level(lvl)
            psc_lst[lvl-1] = len(lv_dc)
        frq_lst.append(psc_lst)

    # sort by total post count, in descending order
    if tre_ixs is not None:
        tmp = sorted(zip(tre_ixs,frq_lst), key=lambda x: sum(x[1]),
                     reverse=True)
        tmp = tmp[:num_top_trs]
        cn_ls = [tup[0] for tup in tmp]
        srt_frl = [tup[1] for tup in tmp]
    else:
        cn_ls = None
        srt_frl = sorted(frq_lst, key=lambda x: sum(x), reverse=True)
        srt_frl = srt_frl[:num_top_trs]

    #print(srt_frl)      # DEBUG

    # convert to dataframe, for plotting;
    # create columns:
    col_ars = []
    for jj in range(num_lvs):  # for each level in the tree
        col_ars.append([sb_ar[jj] for sb_ar in srt_frl])
    # column names:
    cl_ns = [str(x) for x in [jj + 1 for jj in range(len(col_ars))]]
    zp_ar = zip(cl_ns, col_ars)
    if cn_ls is not None:
        df = pd.DataFrame({tup[0]: tup[1] for tup in zp_ar},index=cn_ls)
    else:
        df = pd.DataFrame({tup[0]: tup[1] for tup in zp_ar})

    df.plot.bar(stacked=True)
    plt.xlabel("tree")
    plt.ylabel("level post counts")
    plt.title("top tree complexities (dec tot psts; no OPs)")
    plt.show()


def tree_uniformity_scores_2(tr_ls):
    # adds more scoring methods, over version from _19.py
    # expects list of trees, and returns measure(s) of homophily;
    # can return out_dct, with hom_cto=low_posts; tsh_cnt=number of trees
    # that made the post count cutoff; avg_pol; pln_prp; smp_prp--these last
    # 3 as the "usual" homophily measures; note if no sufficiently complex
    # trees, then these last three are set to np.nan's
    low_posts = 10  # how many posts a tree needs to have to be
    # considered for the uniformity scoring
    # make lists to hold homophily scores:
    score_list = []     # holds scores / tuples, one for each tree that meets
    # the low_posts cutoff
    ct = 0
    for ii, tree in enumerate(tr_ls):
        num_lvs, num_pts = tree.get_complexity()
        if num_pts < low_posts:
            continue
        ct += 1
        pol_lst = []        # for polarity of post
        lgv_lst = []        # for log visibility value of post
        for lvl in range(num_lvs):
            lv_dc = tree.get_level(lvl)
            for key in lv_dc:
                post = lv_dc[key]
                pol_lst.append(post.get_polarity())
                lgv_lst.append(post.get_log_vis())
        avg_pol = abs(np.mean(pol_lst))  # average polarity over all posts in
        # the tree (abs val taken after)
        pol_lo = sum([1.0 for pl in pol_lst if pl < 0.0])  # count number of
        # posts with polarity < 0.0
        pol_hi = len(pol_lst)-pol_lo		# number of posts with polarity
        # >= 0.0
        plain_prop = max(pol_lo/(pol_lo+pol_hi),pol_hi/(pol_lo+pol_hi)) #
        # plain proportion--in [0,1]--1 being maximally polarized by >0 / <0
        # condition
        purity_prop = max((pol_lo+pol_hi+2)/(pol_lo+1),
                          (pol_lo+pol_hi+2)/(pol_hi+1)) # this is prlly not
        # very useful (plain_prop, of the 2, is prlly much better, as for one
        # it's normalized)
        sm_lo = sum([abs(pl) for pl in pol_lst if pl < 0.0])
        sm_hi = sum([abs(pl) for pl in pol_lst if pl >= 0.0])
        sum_pol_prop = max(sm_lo/(sm_lo+sm_hi),sm_hi/(sm_lo+sm_hi))  # like
        # plain_prop, but this includes polarities (vs simple frequency); in
        # [0,1], with 1=maximal polarization
        #sum_pol_prop = max((sm_lo+sm_hi+1.0)/(sm_lo+0.5),
        #                   (sm_lo+sm_hi+1.0)/(sm_hi+0.5))  # combination of pol
        # measure and ratio between the two sides
        rav_pol = np.mean(pol_lst)      # plain average polarity (no absolute
        # value)
        score_list.append((len(pol_lst),avg_pol,plain_prop, #purity_prop,
                               sum_pol_prop,rav_pol))  # len(pol_lst)
        # is
        # just
        # the number of posts in the tree
    out_dct = {"hom_cto":low_posts, "tsh_cnt":ct, "aav_pol":np.nan,
               "pln_prp":np.nan, "smp_prp":np.nan, "raw_pol":np.nan}
    if ct == 0:
        print("zero trees at post threshold %s for homophily check" % low_posts)
    else:
        co_lists = list(zip(*score_list))
        avg_per = [np.mean(sub_lst) for sub_lst in co_lists]
        print("***average homophily scores for %s trees with >= %s posts:***" %
              (ct, low_posts))
        print("avg_abs pol: %s; plain prop: %s; sum pol prop: "
              "%s; raw_avg pol: %s" % tuple(avg_per[1:]))
        out_dct["aav_pol"] = avg_per[1]  # this can also help detect for
        # homophily within trees--it will be near 0.0 if most trees are "well
        # mixed" and near 1.0 if most trees are ~purely homophilic
        out_dct["pln_prp"] = avg_per[2]
        out_dct["smp_prp"] = avg_per[3]
        out_dct["raw_pol"] = avg_per[4]  # the raw average polarity, over all
        # the trees that made the complexity / post count cut

    return out_dct


def tree_uniformity_scores(tr_ls):
    # original, simple averages, from eg _19.py
    # expects list of trees, and returns measure(s) of homophily
    low_posts = 10  # how many posts a tree needs to have to be
    # considered for the uniformity scoring
    # make lists to hold homophily scores:
    ps_1 = []
    ps_2 = []
    as_1 = []
    as_2 = []
    ct = 0
    for ii, tree in enumerate(tr_ls):
        num_lvs, num_pts = tree.get_complexity()
        if num_pts < low_posts:
            continue
        ct += 1
        pol_lst = []        # for polarity of post
        lgv_lst = []        # for log visibility value of post
        for lvl in range(num_lvs):
            lv_dc = tree.get_level(lvl)
            for key in lv_dc:
                post = lv_dc[key]
                pol_lst.append(post.get_polarity())
                lgv_lst.append(post.get_log_vis())
        plain_avg = np.mean(pol_lst)
        plain_score_1 = abs(plain_avg)    # ie |avg tree polarity|--close to 1
        # means polarized-homophilic; near 0 could nonetheless be homophilic,
        # just around polarity 0, etc.
        ps_1.append(plain_score_1)
        plain_score_2 = np.std(pol_lst)   # standard deviation--the lower this
        # is, the more homophilic (less variation among polarities in the tree)
        ps_2.append(plain_score_2)
        # filter for posts with log(vis)
        # greater than some threshold--
        # from Post class: for 0, use 1/4; for 1 use 1/2; for 2 use 3/4 (ie 
        # these are translations from upvote-downvote scores to log(vis) (I 
        # think)--so to filter for posts with at least 1 upvote-downvote net,
        # use 1/2 cutoff for log(vis));
        # note this then does not do a by-visibility weighted average, 
        # but just uses a visibility cutoff
        mod_pls = [pol_lst[ii] for ii in range(len(pol_lst)) if lgv_lst[
            ii]>=1/2]

        # DEBUG; to show what a "unpopular" but ~complex tree might look like:
        #if len(mod_pls)==0:
        #    print("debug lgv_lst: %s" % lgv_lst)

        if len(mod_pls) > 0:
            adj_score_1 = abs(np.mean(mod_pls))
            as_1.append(adj_score_1)
            adj_score_2 = np.std(mod_pls)
            as_2.append(adj_score_2)
    if ct == 0:
        print("zero trees at post threshold %s for homophily check" % low_posts)
    else:
        print("average homophily scores for %s trees with >= %s posts:" %
              (ct, low_posts))
        print("plain avg: %s; plain std: %s; wtd avg: %s; wtd std: %s" %
              (np.mean(ps_1),np.mean(ps_2),np.mean(as_1) if len(as_1)>0 else
               "NA",
            np.mean(as_2) if len(as_2)>0 else "NA"))


def tree_uniformity_scores_OLD(tr_ls):
    # NOTE: the chi-square approach to determining homophily in a tree
    # had some shortcomings (for one, I think, around tree size / number of
    # posts; also, eg bins [5,0,5,0] may produce the same score as [5,5,0,0],
    # etc)
    # scores trees for uniformity re polarity;
    # consider a tree's posts' polarities (possibly weighted by visibility,
    # or log of visibility (guarding away from <1))--how uniformly are the
    # polarities distributed? eg with 5 posts, {-1,-0.5,0.0,0.5,1.0} would be
    # very uniformly distributed, while same with {1,1,1,1,1} would be very
    # non-uniformly distributed
    tr_pv_ls = [None]*len(tr_ls)       # for returning per-tree uniformity
    # p-values (None if not computed for that tree)
    low_posts = 10      # how many posts a tree needs to have to be
    # considered for the uniformity scoring
    num_bns = 8 # number of bins for stacked bars
    bns = np.linspace(-1,1,num=num_bns+1)

    # create a list of binned polarity counts, one for each sufficiently
    # populous tree in tr_ls
    unm_lst = []   # list for holding measures of tree uniformities
    tr_ct = 0       # for diagnostic / DEBUG
    for ii,tree in enumerate(tr_ls):
        num_lvs, num_pts = tree.get_complexity()
        if num_pts < low_posts:
            continue
        pol_lst = []
        wts_lst = []        # for storing value weights (if wanted)
        for lvl in range(num_lvs):
            lv_dc = tree.get_level(lvl)
            for key in lv_dc:
                post = lv_dc[key]
                vis = max(1.0,1+post.get_net_votes())
                # option here to weight the tree post polarities by (eg)
                # log of visibilities (guarded from <1):
                pol_lst.append(post.get_polarity())
                wts_lst.append(np.log(vis))
        # histogram, including optional weights (each polarity score eg gets
        # a weight of log(vis) (frequency multiplier))
        hst = np.histogram(pol_lst,bins=bns,weights=wts_lst)[0].astype(float)
        if sum(abs(hst)) < 1e-10:
            continue    # histogram has all 0 values / frequencies
        tr_ct += 1
        mn_fq = sum(hst)/num_bns    # expected frequency, per cell
        chi2, pval = stats.chisquare(hst,[mn_fq]*num_bns)
        # DEBUG
        #print("debug chi-sq: %s, %s" % (mn_fq,hst))
        unm_lst.append((chi2,pval))
        tr_pv_ls[ii] = pval
    # DEBUG
    #print("raw tree uniformity scores (chi sq, p-value): %s" % unm_lst)
    if len(unm_lst) > 0:
        pv_tk = np.zeros(3)
        for tup in unm_lst:
            if tup[1]<0.10:
                pv_tk[2] += 1
                if tup[1]<0.05:
                    pv_tk[1] += 1
                    if tup[1]<0.01:
                        pv_tk[0] += 1
        pv_tk /= len(unm_lst)
        print("proportion of %s trees with >= %s posts with p-values (<0.01, "
              "<0.05, "
              "<0.10): (%s, %s, %s)" % (tr_ct,low_posts,*pv_tk))
    else:
        print("no trees met %s cutoff in tree_uniformity_scores" % low_posts)
    return tr_pv_ls   # list of same length as tr_ls, with p-val from chi-sq
    # for any trees / indices for which computation was done


def trees_lvl_cmp(tr_ls):
    # for eg trees_data, for trees_list, returns a list of tuples, each tuple
    # the number of posts at (level 1, level 2, ..., level nlvl), with level 1
    # being the 1st non-root/non-OP level of the tree
    nlvl = 5        # number of non-root levels to record for each tree
    lvs_cmp = []
    for ii in range(len(tr_ls)):
        tree = tr_ls[ii]
        nm_lv = tree.get_complexity()[0] # number of levels in the tree,
        # including root
        tmp_sbl = [len(tree.get_level(jj)) for jj in range(1,nm_lv)
                   if jj <= nlvl]
        if len(tmp_sbl) < nlvl:
            tmp_sbl = tmp_sbl + [0]*(nlvl-len(tmp_sbl))
        lvs_cmp.append(tmp_sbl)
    return lvs_cmp, nlvl


def agents_plot(tr_ls,ky_ls):
    # for agents, plots stacked bar plot, according to
    # agents' upvote/downvote scores, sorted by total votes per agent;
    # note this only considers a list of trees, and takes the total upvotes
    # / downvotes from all posts in all trees in the list (id'd with agent_id)
    # ky_ls--list of available agents
    num_top_ags = 10        # how many most-voted-on-agents to include in the
    # plot
    ag_vt_dc = {key:[0,0] for key in ky_ls}   # dictionary for holding agent
    # upvote / downvote scores,
    # keyed by agent_id; of form, agent_id: [upvotes, downvotes]
    for tree in tr_ls:  # go through trees
        for lvl in range(1,tree.get_complexity()[0]): # go through levels in
            # the tree, ignoring 0th level (assuming the root / OPs are
            # anonymous--ie no particular agent has created them)
            lv_dc = tree.get_level(lvl)
            for key in lv_dc:   # go through posts in tree level dictionary
                if key not in ag_vt_dc:
                    continue
                post = lv_dc[key]
                ag_id = post.get_agent_id()
                if ag_id in ag_vt_dc:
                    av = post.updn      # [upvotes, downvotes] for this post
                    ag_vt_dc[ag_id] = [ag_vt_dc[ag_id][0]+av[0],
                                       ag_vt_dc[ag_id][1]+av[1]]
    frq_lst = [(key,ag_vt_dc[key]) for key in ag_vt_dc]
    # sort by total vote count, in descending order
    srt_frl = sorted(frq_lst, key=lambda x: sum(x[1]), reverse=True)
    srt_frl = srt_frl[:num_top_ags]

    # convert to dataframe, for plotting;
    # create columns:
    col_ars = []
    for jj in range(2):  # for upvotes, downvotes
        col_ars.append([sb_ar[1][jj] for sb_ar in srt_frl])
    # column names:
    cl_ns = ["upvotes","dnvotes"]
        #[str(x) for x in [jj + 1 for jj in range(len(col_ars))]]
    zp_ar = zip(cl_ns, col_ars)
    df = pd.DataFrame({tup[0]: tup[1] for tup in zp_ar}, index=[x[0] for
                                            x in srt_frl])

    df.plot.bar(stacked=True)
    plt.xlabel("agent index")
    plt.ylabel("up/dn votes")
    plt.title("top agent vote totals (dec ord)")
    plt.show()

    return ag_vt_dc


def agents_polarity_track(track_dict, net_vot_dct, ags_avl, plot_on):
    # for plotting agent polarity ~concordance with the polarity of the
    # posts the agents comment on or vote on
    # expects agent_track_dct, which contains: [ [perf pol, base pol],
    # [avg icc, #icc], [avg icv, #icv], [avg occ, #occ], [avg ocv, #ocv],
    # [rav c, #c], [rav v, #v] ]    # ic=in-camp, oc=out-of-camp; c=comment,
    # v=vote
    # ags_avl--available (non-retired) agents to do the computations over
    # plot_on--True if wanting plot
    # eg for each agent plot 3 adjacent bars, each in range [-1,1], with
    # positions of in-camp comments and votes, out-of-camp comments and votes,
    # reply comments and votes
    num_top_ags = min(len(track_dict),10)    # how many agents to display (pick
    # topmost eg)

    # create summary dictionary, by agent id; elements are of type
    # {agent_id : [[pol now, base pol], pol ic, pol oc, pol rp]}
    sum_dct = {}
    srt_lst = []    # list to help sort agents by amount of activity
    pol_dsp = {}    # measures average polarity disparity, by agent id, for ri
    # posts--ie disparity between agent (performative) polarity and the average
    # commenting and voting (from skim'd posts in reading bucket) polarities;
    # note this will cover all agents in track_dict (not just "top" most active)

    for key in ags_avl:  # over all agents in agents_avail
        agt_ifo = track_dict[key]
        pr_1 = agt_ifo[0]   # [polarity now, base polarity]
        pr_tt = []
        for ii in range(3):
            pair_pair = agt_ifo[2*(ii+1)-1:2*(ii+1)+1] # slice elements are
            # floats, so OK re protecting original list / mutability
            tt_vl = 0
            tt_ct = 0
            for jj in range(2):
                sbl = pair_pair[jj]
                if sbl[0] is not None:
                    tt_vl += sbl[1]*sbl[0]
                    tt_ct += sbl[1]
            if tt_ct==0:
                out_val = None
            else:
                out_val = tt_vl/tt_ct  # compute average polarity,
                # over comments and votes
            pr_tt.append(out_val)
            if ii==0:       # first pair_pair is for reward intent
                # record the tuple (agent performative polarity,
                # average ri comment and vote polarity)
                pol_dsp[key] = (track_dict[key][0][0],out_val)
        sum_dct[key] = [pr_1] + pr_tt
        srt_lst.append([key,sum([agt_ifo[kk][1] for kk in range(1,7)])]) #
        # total the sum of all comment/vote counts for the agent

    # decide most active agents, and pare sum_dct to only their keys
    srt_lst = sorted(srt_lst, key=lambda x: x[1], reverse=True)
    srt_lst = srt_lst[:num_top_ags]  # list of keys, in decreasing order of
    # popularity
    for key in sum_dct.copy():
        if key not in [x[0] for x in srt_lst]:
            del(sum_dct[key])

    if plot_on:

        # plotting
        bar_height = 2  # Total height of the bars from -1 to 1
        bar_pos = np.arange(1,len(sum_dct)+1,1)  # X positions for the bar pairs
        width = 0.2  # Width of the bars

        # Create the figure and axis
        fig, ax = plt.subplots()

        # Plot bars:
        # Bars start at -1 and have a height of 2 (so they go from -1 to 1)
        ax.bar(bar_pos - width, bar_height, width, color='lightgreen',
                     label='Green Bars', bottom=-1)
        ax.bar(bar_pos, bar_height, width, color='lightcoral',
               label='Red Bars', bottom=-1)
        ax.bar(bar_pos + width, bar_height, width, color='cornflowerblue',
               label='Blue Bars', bottom=-1)

        # Midpoints of the bars (now at 0 since bars go from -1 to 1)
        midpoint = 0

        # add horizontal lines:
        for ii,bp in enumerate(bar_pos):
            sbl = sum_dct[srt_lst[ii][0]]   # will have [[curr pol, base pol],
            # avg ic,
            # av_oc, av_rp];
            # add cross-bar group horizontal lines:
            # agent current polarity
            ax.hlines(sbl[0][0], bp-3*width/2, bp+3*width/2, colors='black',
                      linewidth=2)
            # agent base polarity
            ax.hlines(sbl[0][1], bp-3*width/2, bp+3*width/2, colors='black',
                      linewidth=2, linestyle='--')
            # Add intra-bar horizontal lines:
            # For green bars
            pt = sbl[1]
            ax.hlines(pt, bp - 3*width/2, bp-width/2, colors='white',
                      linewidth=2)
            # For red bars
            pt = sbl[2]
            ax.hlines(pt, bp-width/2, bp+width/2, colors='white',
                      linewidth=2)
            # For blue bars
            pt = sbl[3]
            ax.hlines(pt, bp+width/2, bp+3*width/2, colors='white',
                      linewidth=2)

        # Set plot limits and labels
        agt_key_ord = [srt_lst[ii][0] for ii in range(num_top_ags)]  # ie key
        # values for agents in the plot, in order from left to right
        ax.set_ylim(-2, 2)  # Adjust the y-axis to make the lines visible
        ax.set_xticks(bar_pos)
        ax.set_xticklabels(agt_key_ord)
        plt.title("pols, top agents (dec, cast c/v totals; #=net vts)")

        for ii,br_xx in enumerate(bar_pos):
            plt.text(br_xx,bar_height/2,net_vot_dct[agt_key_ord[ii]],ha='center',
                     va='bottom')

        # DEBUG
        '''
        print("agent polarities, for bar plot:")
        for ii in range(len(srt_lst)):
            key = srt_lst[ii][0]
            print("agent id: %s; %s" % (key,sum_dct[key]))
        '''

        plt.show()

    # pol_dsp should have all possible agent keys (from agent_track_dct),
    # but could have (perf polarity, None) as entries
    tp_ls = [abs(pol_dsp[key][0]-pol_dsp[key][1]) for key in pol_dsp
                     if pol_dsp[key][1] is not None]
    avg_dsp = np.mean(tp_ls) if len(tp_ls)>0 else "NA"

    if VERBOSE:
        print("average disparity, avg ri post comment/vote vs agent "
              "polarity: %s" % avg_dsp)

    return pol_dsp, avg_dsp  # return the dictionary keyed by agent id in
    # agents_avail
    # with the displacement tuple, difference between agent perf polarity and
    # average ri read score


def agents_activity_track(track_dict, atv_ags, plot_on):
    # for plotting agent activity vs agent polarity;
    # expects agent_tract_dct, which contains: [ [perf pol, base pol],
    # [avg icc, #icc], [avg icv, #icv], [avg occ, #occ], [avg ocv, #ocv],
    # [rav c, #c], [rav v, #v] ]    # ic=in-camp, oc=out-of-camp; c=comment,
    # v=vote
    # atv_ags--active agents (ie not retired)

    act_lst = []  # contains tuples of type (agent polarity, agent activity)
    ior_lst = []    # tuples, one per agent, with counts (#ic tot,
    # #oc tot, #rp tot)--ie how many comments+votes were made by each agent
    # in each of ~ri, po/al, reply categories
    for key in atv_ags:
        ls_sl = track_dict[key]  # fetch agent's tracker list of sublists
        tot_act = sum([sl[1] for sl in ls_sl[1:]])
        act_lst.append((ls_sl[0][0],tot_act))
        ior_lst.append((ls_sl[1][1]+ls_sl[2][1], ls_sl[3][1]+ls_sl[4][1],
                        ls_sl[5][1]+ls_sl[6][1]))

    uzl = list(zip(*act_lst))

    if plot_on:
        plt.scatter(x=uzl[0],y=uzl[1])
        plt.title("agent polarity vs total activity")
        plt.show()

    if VERBOSE:
        print("per-agent distribution of ri, po/al, reply comment+vote: %s"
          % ior_lst)
    zp_il = list(zip(*ior_lst))
    out_tup = tuple(np.mean(zp_il[ii]) for ii in range(3))
    if VERBOSE:
        print("per-agent distro averages: %s; %s; %s" % out_tup)
    return out_tup


def agents_activity_track_2(tr_ls, track_dict, agn_ret):
    # for binning agents by polarity values, then checking activity levels
    # and net upvotes (eg); code has been used from agents_plot() and
    # agents_activity_track()
    # tr_ls--tree_list
    # track_dict--agent_track_dct; which contains: [ [perf pol, base pol],
    # [avg icc, #icc], [avg icv, #icv], [avg occ, #occ], [avg ocv, #ocv],
    # [rav c, #c], [rav v, #v] ]    # ic=in-camp, oc=out-of-camp; c=comment,
    #  v=vote
    # agn_ret--list of retired agent ids
    # intermediary agent_stats: {agent_id: [agent perf polarity, agent base
    # polarity,total upvotes,
    # total downvotes, total comments made,total votes cast]}
    # returns:
    #   number of agents per polarity bin
    #   average (per-agent) activity per polarity bin: [upvotes,downvotes,
    #       total comments made,total votes cast,polarity drift]

    agent_stats = {}
    num_bns = 6     # how many bins to lump agent polarity scores into

    # pull comment and vote counts from agent_track_dct
    for ag_id in track_dict:
        if ag_id in agn_ret:
            continue
        ls_sl = track_dict[ag_id]  # fetch agent's tracker list of sublists
        tot_com = ls_sl[1][1]+ls_sl[3][1]+ls_sl[5][1]
        tot_vot = ls_sl[2][1]+ls_sl[4][1]+ls_sl[6][1]
        agent_stats[ag_id] = {"per_pol":ls_sl[0][0], "bas_pol":ls_sl[0][1],
                    "up_vt":0, "dn_vt":0, "tot_com":tot_com,
                    "tot_vot": tot_vot}

    # pull upvotes/downvotes from tree_list
    for tree in tr_ls:  # go through trees
        for lvl in range(1,tree.get_complexity()[0]): # go through levels in
            # the tree, ignoring 0th level (assuming the root / OPs are
            # anonymous--ie no particular agent has created them)
            lv_dc = tree.get_level(lvl)
            for key in lv_dc:   # go through posts in tree level dictionary
                post = lv_dc[key]
                ag_id = post.get_agent_id()
                if ag_id not in agn_ret:
                    av = post.updn      # [upvotes, downvotes] for this post
                    agent_stats[ag_id]["up_vt"] += av[0]
                    agent_stats[ag_id]["dn_vt"] += av[1]

    # bin agents by polarity scores, and record total upvotes, downvotes,
    # comments, votes for each bin
    bin_lms = np.linspace(-1.0,1.0,num_bns+1)[1:]  # right-end limits for the
    # polarity score bins
    bin_stats = [[0,0,0,0,0] for _ in range(num_bns)]
    bin_cts = [0 for _ in range(num_bns)]       # how many agents in each
    # polarity bin
    for ag_id in agent_stats:
        agt_pol = agent_stats[ag_id]["per_pol"]
        bn_ix = bisect.bisect_left(bin_lms,agt_pol)  # index for bin this
        # polarity score belongs in--0,1,...,num_bns-1
        bin_cts[bn_ix] += 1
        bin_stats[bn_ix][0] += agent_stats[ag_id]["up_vt"]
        bin_stats[bn_ix][1] += agent_stats[ag_id]["dn_vt"]
        bin_stats[bn_ix][2] += agent_stats[ag_id]["tot_com"]
        bin_stats[bn_ix][3] += agent_stats[ag_id]["tot_vot"]
        bin_stats[bn_ix][4] += (agent_stats[ag_id]["per_pol"]-
                                agent_stats[ag_id]["bas_pol"])  # measures
        # agent polarity drift (eg base polarity=0.5, performance
        # polarity=0.75, then the drift is 0.75-0.50 = +0.25)

    avg_bin_stats = [[bin_stats[jj][kk]/bin_cts[jj] if bin_cts[jj] > 0 else
                      np.nan
                      for kk in range(5)] for
                     jj in range(num_bns)]
    return bin_cts, avg_bin_stats


def agents_polarity_history(agent_pol_hst, ag_dc, ags_avl):
    # for debugging; uses entries in agent_pol_hst dictionary to plot
    # skim list choices and, of those, actual reading choices from skim list
    # for reward intent (only); color map used for (normalized) associated
    # visibility scores;
    # agent_pol_hst--dictionary, by agent id, with lists of tuples,
    # (pol,vis), from skim list, with visibility normalized, and vis
    # "flipped" (ie <0) for entries that were actually read;
    # ag_dc--agent dictionary, for agent polarities
    # ags_avl--available agents for study (not retired)
    num_tpa = 5     # max number of top agents to plot
    key_lst = []        # list of keys with activity levels, for getting
    # "top" agents in some sense (say in # posts read)
    for key in ags_avl:  # agent_pol_hst dictionary keys / agent ids should
        # always "cover" agents_avail keys
        if key is None: # DEBUG
            print("issue with keys in agent_pol_hst...?")
            raise ValueError
        key_lst.append((key,len([tup for tup in agent_pol_hst[key]
                                 if tup[1]<0.0])))
    key_lst = sorted(key_lst, key=lambda x: x[1], reverse=True)  # agent key
    # with most read posts will be 1st, then 2nd most read, etc.
    num_tpa = min(num_tpa,len(key_lst))
    key_lst = key_lst[:num_tpa]

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    cmp = plt.get_cmap('winter')

    jit_val = 0.05   # how much horizontal jitter
    for ii,tup in enumerate(key_lst):
        key = tup[0]
        tup_lst = agent_pol_hst[key]  # this will be agent key's list of
        # (pol, vis) tuples;
        # make jitter "base" of x values:
        xx_vs = (ii+1)*np.ones(len(tup_lst)) + np.random.uniform(
            -jit_val, jit_val, size=len(tup_lst))
        clr_lst = [abs(tup[1]) for tup in tup_lst] # ie use
        # colormap to color points by visibility score
        plt.scatter(xx_vs, [tup[0] for tup in tup_lst], alpha=0.6,
                color=cmp(clr_lst), label=("Set %s" % ii))
        ix_rd = [ii for ii, tup in enumerate(tup_lst) if tup[1]<0.0] #
        # indices in tup_lst associated with read tuples
        tup_sbl = [tup_lst[ix] for ix in ix_rd]
        plt.scatter([xx_vs[ix] for ix in ix_rd], [tup[0] for tup in tup_sbl],
                    s=100, alpha=0.4, color='red')
        ht = ag_dc[key].get_polarity()
        ax.hlines(ht, (ii+1)-0.1,(ii+1)+0.1, colors='black', linewidth=2)
    ax.set_xticks(range(1,len(key_lst)+1))
    ax.set_xticklabels([tup[0] for tup in key_lst])
    plt.show()


def create_agent(id):
    # creates an agent, randomly, with potential for feedback between polarity
    # score and other values;
    # to instantiate an agent, need:
    # ID, pol, spd, ev, ib, ni, police, altruism, ts, tm
    pol = 2 * random.random() - 1   # polarity score

    tmp = stats.norm(0.0,0.15).rvs()
    if tmp<-0.5:    # clip to within [-0.5,0.5]
        tmp = -0.5
    elif tmp>0.5:
        tmp = 0.5
    spd = (tmp+0.5)*(NM_HI-NM_LO)+NM_LO  # polarity spread; low end is a
    # thinner
    # spread and high end is fatter; this affects eg vote() and comment()
    # functions (how "tight" the agent is relative to their own polarity value)
    ni = gen_nrm()
    # feed back polarity into need to influence
    dst = 1-ni
    dst = dst - abs(pol)*dst/2
    ni = 1-dst
    ev = gen_nrm()
    # feed back polarity into expressive value
    dst = 1-ev
    dst = dst - abs(pol)*dst/2
    ev = 1-dst
    ib = gen_nrm()
    police = gen_nrm()
    # feed back polarity into police
    dst = 1-police
    dst = dst - abs(pol)*dst/4
    police = 1-dst
    police *= PO_ATTN
    altruism = gen_nrm()
    altruism *= AL_ATTN
    ts = 0.0    # tree starting--maybe not used
    tm = gen_nrm()      # total rewards multiplier
    # feed back polarity into tm
    dst = 1-tm
    dst = dst - abs(pol)*dst/4
    tm = 1-dst
    return Agent(ID=id, pol=pol, spd=spd, ev=ev, ib=ib, ni=ni, police=police,
                 altruism=altruism, ts=ts, tm=tm)


def gen_nrm(mean = 0.5, std = 0.125):
    # helper function, to generate normal in [0,1], with clipping
    res = np.random.normal(mean,std)
    if res < 0:
        res = 0.0
    elif res > 1:
        res = 1.0
    return res


def show_agents(ag_dc):
    # given agent dictionary, ag_dc, produce stats on agents
    # NOTE: this is not set up for agent churn--if only wanting non-retired
    # agents, need to modify this function
    pol_lst = []    # polarization list, per-agent
    tr = 0.0     # total reward
    cm_pr = 0.0     # comment propensity
    gr_pr = 0.0     # reading propensity
    gv_pr = 0.0     # voting propensity
    trs_red = 0.0       # trees read
    unr_rep = 0.0       # unread replies (outstanding reply pings)
    for key in ag_dc:
        agent = ag_dc[key]
        pol_lst.append(agent.polarity)
        tr += sum(agent.extrinsic_points)
        cm_pr += agent.comment_propensity
        gr_pr += agent.general_reading
        gv_pr += agent.general_voting
        trs_red += len(agent.trees_read)
        unr_rep += len(agent.reply_pings)
    num_ags = len(ag_dc)
    print("stats for %s agents:" % num_ags)
    print("mean and stdev polarity: %s, %s" %
          (np.average(pol_lst),np.std(pol_lst)))
    print("avg total reward: %s; avg comment prop: %s; avg gen read: %s; avg "
          "gen vote: %s" % (tr/num_ags, cm_pr/num_ags, gr_pr/num_ags,
                            gv_pr/num_ags))
    print("avg trees read: %s; avg unread replies: %s" % (trs_red/num_ags,
                                                          unr_rep/num_ags))


def update_ag_tr_lst(sublst, in_pl):
    # helper function for updating agent_track_dct;
    # expects mutable 2-element sublist from agent_track_dct, and
    # input polarity score; modifies mutable sublist in place
    if sublst[0] is None:
        sublst[0] = in_pl
        sublst[1] += 1
    else:
        tp_ct = sublst[1]
        new_avg = (tp_ct * sublst[0] +
                   in_pl) / (tp_ct + 1)
        sublst[0] = new_avg
        sublst[1] += 1


def tree_probs(sel_lst, tree_ratings):
    # for agent selection of which trees to read, favoring more complex trees;
    # this considers tree complexity (number of posts), and popularity (upvote-
    # downvote total), and bases a score with equal weighting on both these
    # values;
    # tree_ratings contains tuples, (ix, total # posts, total score)
    # sel_lst contains which ix's in tree_ratings to use
    # returns a probability distribution over indices in sel_lst;
    # for starter trees, tot_pos can be "at worst" an array of 1's
    tot_pos = np.array([tup[1] for tup in tree_ratings
                        if tup[0] in sel_lst]).astype(float)  # number of posts;
    # for starter trees, tot_scr can be "at worst" an array of 0's; note
    # net scores for trees can also be negative
    tot_scr = np.array([tup[2] for tup in tree_ratings if tup[0] in
                            sel_lst]).astype(float)  # net upvotes-downvotes;
    # normalize each
    tmp = sum(tot_pos)
    tot_pos /= tmp

    # DEBUG
    if len(tot_scr)==0:
        breakpoint()

    tmp = min(tot_scr)
    if tmp < 0:  # if any tree scores are < 0, then shift to make those
        # scores all positive (lowest shifted to be 0)
        tot_scr -= tmp
    # all values in tot_scr at this point should be >= 0
    tmp = float(sum(tot_scr))  # at simulation start, tot_scr could be all 0's
    if tmp > 0.5:  # avoid possible rounding issues (like tmp = 1e-15)
        tot_scr /= tmp
    else:  # if all scores are ~0, then just use uniform distribution
        tot_scr = np.ones(len(tot_scr))/len(tot_scr)
    prb_out = (tot_pos+tot_scr)/2  # average the 2 (normalized) score
    # arrays (will remain normalized)

    # decide on any smoothing functions for final pmf
    prb_out = np.sqrt(prb_out)
    return prb_out/sum(prb_out)


def proc_ret_agents(ag_dc, rt_ls):
    # for data recording for agents that churn (if any);
    # ag_dc--agent dictionary
    # rt_ls--list of retired/churned agents
    # eg return polarity frequency bins of the retired agents
    num_bns = 3
    bin_lms = np.linspace(-1.0, 1.0, num_bns + 1)[
              1:]  # right-end limits for the
    # polarity score bins
    bin_cts = [0 for _ in range(num_bns)]
    for ag_id in rt_ls:     # for each retired agent id
        agt_pol = ag_dc[ag_id].get_base_polarity()
        bn_ix = bisect.bisect_left(bin_lms, agt_pol)  # index for bin this
        # polarity score belongs in--0,1,...,num_bns-1
        bin_cts[bn_ix] += 1
    return bin_cts


def agent_make_data(key,agent,agents_data,pol_dsp,
                    agent_track_dct,ag_vt_dc,ag_rs_ct):
    # to record data to agent_data dictionary (keyed by agent id)
    # key--agent id
    # agent--agent object
    # agents_data--dictionary for storing per-agent-id data dictionaries
    # *all the following are dictionaries, by agent id:*
    # pol_dsp--(agent performative polarity, avg ri comment and vote
            # polarity)
    # agent_track_dct--[ [perf pol, base pol], [avg icc, #icc], [avg icv,
    # #icv], [avg occ, #occ], [avg ocv, #ocv], [rav c, #c], [rav v, #v] ]
    # ic=in-camp, oc=out-of-camp; c=comment, v=vote
    # ag_vt_dc--[upvotes,downvotes]
    # ag_rs_ct--how many rounds agent was active in the simulation
    tmp_dct = {}
    tmp_dct["per_pol"] = agent.get_polarity()
    tmp_dct["bas_pol"] = agent.base_polarity
    tmp_dct["spread"] = agent.spread
    tmp_dct["ni"] = agent.need_to_influence
    tmp_dct["ev"] = agent.expressive_value
    tmp_dct["ib"] = agent.inter_belong
    tmp_dct["po"] = agent.policing
    tmp_dct["al"] = agent.altruism
    tmp_dct["tm"] = agent.TM
    if key in pol_dsp:
        tmp_dct["rir_avg"] = pol_dsp[key][1] # tuple, (agent performative
        # polarity, average ri comment and vote polarity)--just record the
        # avg ri comment/vote polarity
    else:   # handle / warn if wanted
        raise KeyError
    tmp_lst = agent_track_dct[key]
    tmp_lst = [xx for sbl in tmp_lst[1:] for xx in sbl] # flatten sublists
    # in tmp_lst[1:]
    trk_lab = ["apc_ic", "nmc_ic", "apv_ic", "nmv_ic", "apc_oc", "nmc_oc",
               "apv_oc", "nmv_oc", "apc_rp", "nmc_rp", "apv_rp", "nmv_rp"]
    if len(tmp_lst) != len(trk_lab):
        raise ValueError
    sub_dct = {trk_lab[ii]:tmp_lst[ii] for ii in range(len(trk_lab))}
    tmp_dct.update(sub_dct)
    vot_lab = ["up","dn"]
    sub_dct = {vot_lab[ii]:ag_vt_dc[key][ii] for ii in range(2)} # upvotes
    # downvotes
    tmp_dct.update(sub_dct)
    tmp_dct["num_rac"] = ag_rs_ct[key]      # number of rounds active
    agents_data[key] = tmp_dct


def run_simulation(new_sig=None):

    if new_sig is not None:
        Tree.change_sig(new_sig)  # update SIG in Tree class

    agent_proportion = 0.5  # what proportion of agents from total to draw as
    # active for each round of the simulation;

    # for tree retirement
    co_mn, co_sd = 7.5*NUM_AGS*0.75, (7.5*NUM_AGS*0.75)/8  # for use
    # when retiring trees; suppose each agent makes an average of 5 to 10 posts
    # per tree they "read," and expect ca 3/4 of the agents to have read a tree
    # before it's retired by the other retirement mechanism (of most agents
    # reading the tree)

    # set up initial trees
    tree_list = [create_tree() for _ in range(NUM_TRS)]
    active_trees = list(range(len(tree_list)))  # indices of active trees in
    # tree_list

    # set up initial agents
    agent_dict = {ii:create_agent(ii) for ii in range(NUM_AGS)}
    active_agents = []  # ids of active agents in
    # agent_dict (ie keys)

    # for retiring agents
    agent_engage = []       # for agent churn / retirement, in-loop tracker of
    # per-agent engagement at this point in the simulation
    agent_ret_tsh = 10      # number of active rounds cutoff for an agent to be
    # considered for churn/retirement
    agent_id_ret = []       # list of ids of retired agents
    agents_avail = []       # list of ids of non-retired
    # agents (complementary to agent_id_ret, wrt agent_dict keys/ids)
    ag_rt_tk = []  # DEBUG, for agent churn diagnostics
    ch_pl = 0.0  # DEBUG, for tracking average polarity of churned agents

    # data recording (for eg dataframes / csv files)
    session_data = {}
    agents_data = {}
    trees_data = {}
    tr_ex_dc = {}   # helper for trees_data, for recording some tree information
    # during the simulation; of format, tree_list index key: [number of rounds,
    # is_retired]


    # DEBUG
    trees_retired = 0
    agent_pol_hst = {key: [] for key in agent_dict}    # for debugging; this will
    # provide a total of all polarity scores in all skim buckets for this agent,
    # via tuples of type (polarity, normalized visibility), with the sign of
    # the visibility (in [0,1]) "flipped" if that polarity-post was chosen by
    # the agent for reading (so eg (0.3,0.5)=skimmed but not read, and
    # (0.3,-0.5)=skimmed and read)

    net_vot_dct = {}        # DEBUG, for double checking per-agent vote totals
    agent_track_dct = {}    # DEBUG, for tracking agent polarity agreement with
    # posts commented on and voted on for reward intent (ri) only
    ag_rs_ct = {}       # DEBUG; agent rounds counter--counts number of rounds
    # for each agent
    for key in agent_dict:  # DEBUG; initialize agent_track_dct
        agent = agent_dict[key]
        agent_track_dct[key] = ([[agent.get_polarity(),agent.base_polarity]] +
                            [[None,0] for _ in range(6)])
        # format is [ [perf pol, base pol], [avg icc, #icc], [avg icv, #icv],
        # [avg occ, #occ], [avg ocv, #ocv] ], where ic. = in-camp; oc. = out-of
        # -camp; ..c=comment; ..v=vote; also, tack on [ [rav c, #c], [rav v, #v] ]
        # for separately tallying reply comments and votes
        ag_rs_ct[key] = 0

    # for agent polarity tracking
    ini_pol = [agent_track_dct[key][0][0] for key in agent_track_dct]
    ini_pol_sav = np.histogram(ini_pol, np.linspace(-1, 1, 7))[0]


    # ***
    # outer loop, over simulation rounds
    # ***

    # outer loop, over rounds--round_num = 0, 1, ...
    for round_num in range(NUM_RDS):

        if not VERBOSE:
            if round_num%10==0:
                print("round number: %s" % round_num) # DEBUG
                print("%s trees made so far" % len(tree_list))
        else:
            print("round number: %s" % round_num)  # DEBUG
        if VERBOSE:
            print("%s trees made so far" % len(tree_list))
        #show_trees(tree_list)   # DEBUG
        #show_agents(agent_dict)     # DEBUG

        num_trees_read = 0  # counter per-round, of total number of trees read,
        # over all agents; used to estimate steady-state tree generation

        # record active trees this round (for trees_data):
        for ix in active_trees:
            if ix in tr_ex_dc:
                tr_ex_dc[ix][0] += 1
            else:
                tr_ex_dc[ix] = [1,False]    # False=not retired (yet

        # ***
        # agent loop preparations
        # ***

        agents_avail = [key for key in agent_dict if key not in agent_id_ret]  #
        # which agents are not retired (by id)

        # determine per-agent total reward, and current propensities
        # (comment, vote, etc.); this is computed per-round, for all agents
        # (regardless if they've posted/been active this round or not)

        # rate the trees indexed in active_trees, eg for tuples
        # (tree_list index, total # posts, total score (upvotes-downvotes))
        tree_ratings = [(ix, tree_list[ix].get_complexity()[1],
                      tree_list[ix].get_popularity(net=True)) for ix in
                     active_trees]

        # determine normalization constants, for use with propensity scores:
        max_scs = [0.0, 0.0, 0.0, 0.0]  # list of maximal scores this round,
        # respectively for [rc, ep, nrc, nep]
        sum_scs = [0.0, 0.0, 0.0, 0, 0]  # list of sums of scores over all agents,
        # respectively for [rc, ep, nrc, nep]
        for key in agent_dict:
            agent = agent_dict[key]
            vals = agent.set_observations(round_num)  # returns [rc, ep, nrc, nep]
            for kk in range(4):
                if max_scs[kk] < vals[kk]:
                    max_scs[kk] = vals[kk]
                sum_scs[kk] += vals[kk]
        # note max_scs at this point are each at least 0.0

        # go through each agent, and set (a) total reward (raw), (b) total reward
        # normalized, (c) propensity scores
        for key in agent_dict:
            agent = agent_dict[key]
            agent.set_total_reward(round_num, sum_scs)
            agent.set_propensities(max_scs)  # max_scs are used to normalize
            # propensities (this round) (and will each be >= 0.0)

        # decide on active agents for this round
        num_sam = int(agent_proportion*len(agents_avail))
        active_agents = random.sample(agents_avail, num_sam)  # keys
        # (ids) of active agents

        # zero upvote-downvote per-agent array for this round (index = agent_id)
        ep_dict = {key:(0.0,0.0) for key in agent_dict.keys()} # for storing
        # agent upvote-downvote totals earned this round; tuple:
        # (net votes, downvotes)

        # ***
        # loop over agents
        # ***

        for key in active_agents:

            agent_now_id = key
            #print("agent %s start" % agent_now_id)     # DEBUG
            ag_rs_ct[key] += 1  # agent participated in this round

            agent = agent_dict[key]

            # assume eg have an agent_dict, with agent objects to loop over; the
            # agent_id is the key of the agent object in agent_dict;
            # assume round_num is the number of the current round (0,1,2,...)

            # ***
            # create skim posts list
            # ***

            # flow for initial skimming for a given agent
            # tree_list--this can be all the tree objects, ever created
            # active_trees--this can be the indices in tree_list of active (eg
            # non-defunct) trees in tree_list;
            # coordinate lists--[tree #, level #,
            #         # key val], for initial review set for this agent:
            skim_cur_list = []   # list of coordinates for curation filtered
            # (for reward intent reading)
            skim_oth_list = []  # list of coordinates for random sample of posts
            # from tree (for policing and altruism readings)

            # agent randomly selects trees to read, based on tree visibility
            # / size / popularity
            sel_lst = [ix for ix in active_trees if ix not in
                       agent.get_trees_read()]
            num_trees = min(max(1, int(np.round(np.random.normal(3.0,1.0)))),
                            len(sel_lst))
            gen = np.random.default_rng()

            # DEBUG
            if len(sel_lst)==0:
                print("at round %s, ran out of trees to read for agent %s; "
                      "consider "
                      "increasing the number of trees created each round" %
                        (round_num, agent_now_id))
                raise ValueError

            p_weights = tree_probs(sel_lst, tree_ratings)
            # randomly select indices in sel_lst for trees to read
            sll_ixs = gen.choice(len(sel_lst),num_trees,replace=False,p=p_weights)
            tre_ixs = [sel_lst[ix] for ix in sll_ixs]
            num_trees_read += num_trees     # running count, over all agents
            #print("trees to read: %s" % tre_ixs)  # DEBUG

            # go through trees, selecting posts for skim list
            for tree_num in tre_ixs:
                tree = tree_list[tree_num]  # tree object
                cur_lst, oth_lst = tree.curation_score(agent.get_id(),
                                                agent.get_polarity())
                # this will be 2 lists, each a list of sublists
                # of type [(level,key), vis score], the first amounting to all
                # "acceptably close" posts in the tree, that correspond sufficiently
                # well with agent's own polarity score, and posts that were not
                # written by this agent (different id than agent); the second is
                # a list of all other posts in the tree (that did not make the
                # curation "cut"; posts written by this agent are also excluded)

                # how many posts max to target for getting from this tree:
                tmp_num = sum([np.random.normal(7.5, 2.0) for _ in
                               range(int(3 + np.round(np.random.random())))])
                # assume a 2:1 ratio for fetched posts for cur:oth skim lists;
                # note that during simulation start, both lists are likely to
                # fall short of tmp_num target, especially cur_lst, because it
                # tries to match polarities (more selective than oth_lst):
                tmp_num_cl = int(min(len(cur_lst),tmp_num/3))  # since these are
                # curated, can afford to take a higher proportion of them and
                # still have "good" selection (in reading_choices)--so lower the
                # amount needed from earlier versions (ie would be 2*tmp_num/3)
                tmp_num_ol = int(min(len(oth_lst),tmp_num/3))

                if tmp_num_cl > 0:   # for reward intent
                    skim_cur_list += rand_skim_choice(tmp_num_cl, cur_lst,
                                                tree_num)
                if tmp_num_ol > 0:    # for policing and altruism
                    # randomly select uniformly (without visibility score
                    # consideration)
                    rnd_ixs = random.sample(range(len(oth_lst)),tmp_num_ol)
                    skim_oth_list += [(tree_num,oth_lst[ix][0][0],
                     oth_lst[ix][0][1]) for ix in
                        rnd_ixs]

            # pass-through here is skim_cur_list and skim_oth_list, lists of
            # post coords; "cur" is for polarity curated posts, for reward intent
            # bucket, and "oth" is for policing and altruism buckets; it may
            # be possible these lists are empty (depending on whether root posts /
            # OPs can
            # be authored by existing agents, or they start ~anonymously)
            #if len(skim_list)==0:  # DEBUG
            #    print("no posts in skim set for agent %s" % agent.get_id())
            # DEBUG
            # note, when simulation starts, there may not be many posts / coords
            # in either list (maybe as few as number of trees skimmed)
            #print("cur len=%s; oth len=%s" % (len(skim_cur_list),
            #                                  len(skim_oth_list)))

            # compute locality scores for skim_..._lists, and store bundled with
            # skim_..._lists coordinates, using convenience class PostContainer
            # objects
            pc_cur_skim = []
            for coord in skim_cur_list:
                povs, lcs = locality_score(coord, tree_list)  # returns both
                # locality score (as (polarity, visibility) tuple)
                # and the post's individual (polarity, visibility)
                pc_cur_skim.append(PostContainer(cor=coord, povs=povs, lcs=lcs))
            pc_oth_skim = []
            for coord in skim_oth_list:
                povs, lcs = locality_score(coord, tree_list)  # returns both
                # locality score (as (polarity, visibility) tuple)
                # and the post's individual (polarity, visibility)
                pc_oth_skim.append(PostContainer(cor=coord, povs=povs, lcs=lcs))

            # select random subset from reply pings in agent object, and get
            # locality scores
            tmp_rep = agent.get_reply_pings()
            reply_list = []
            if len(tmp_rep) > 0:
                # how many replies to read
                num_rep_red = 5 + agent.general_reading*(2.5+5*np.random.random())
                num_rep_red = min(len(tmp_rep), int(round(num_rep_red)))

                red_ixs = random.sample(range(len(tmp_rep)),num_rep_red)

                # holds all replies that the agent will/has read this round
                reply_list = [tmp_rep[ii] for ii in red_ixs]
                # remove these replies from the agent's ping list
                agent.remove_reply_pings(red_ixs)

            # compute locality scores for reply_list, and store bundled with
            # reply_list coordinates, using convenience class PostContainer objects
            pc_reply = []
            for coord in reply_list:
                _, lcs = locality_score(coord, tree_list)
                pc_reply.append(PostContainer(cor=coord, lcs=lcs))
                # DEBUG--how are agents own posts getting into replies??? (this
                # was totally fixed, pretty sure...)
                tp_tr = tree_list[coord[0]]
                tp_ps = tp_tr.get_post(coord[1],coord[2])
                if tp_ps.get_agent_id() == agent.get_id():
                    breakpoint()
                    raise ValueError

            # normalize visibility values in (polarity, visibility) tuples for both
            # pc_skim and pc_reply (separately); this brings visibility score into
            # [0,1] (posts with negative visibility scores (downvotes > upvotes)
            # will have visibility normalized to 0);
            # note overall, this code segment is ~inelegant; consider putting
            # into a function for pc_skim and for pc_reply (possibly a single
            # function, with "povs" / single flag)

            # for "cur" skim list:
            if len(pc_cur_skim) > 0:
                # individual post scores (skim):
                tmp = max([pc.raw_povs[1] for pc in pc_cur_skim])   # max log(vis)
                # over all posts; note this should always be > 0.0
                for pc in pc_cur_skim:
                    pc.norm_povs(1/tmp)  # this will zero any visibilities < 0
                # locality scores (skim):
                tmp = max([pc.raw_loc_scores[1] for pc in pc_cur_skim])
                for pc in pc_cur_skim:
                    pc.norm_lc_sc(1/tmp)
                    agent_pol_hst[agent.get_id()].append(pc.nrm_loc_scores)  #
                    # DEBUG--this appends all in ri skim list to agent_pol_hst list
                    # for this agent
            # for "oth" skim list:
            if len(pc_oth_skim) > 0:
                # individual post scores (skim):
                tmp = max([pc.raw_povs[1] for pc in pc_oth_skim])   # max log(vis)
                # over all posts; note this should always be > 0.0
                for pc in pc_oth_skim:
                    pc.norm_povs(1/tmp)  # this will zero any visibilities < 0
                # locality scores (skim):
                tmp = max([pc.raw_loc_scores[1] for pc in pc_oth_skim])
                for pc in pc_oth_skim:
                    pc.norm_lc_sc(1/tmp)
            # locality scores (reply):
            if len(pc_reply) > 0:
                tmp = max([pc.raw_loc_scores[1] for pc in pc_reply])
                for pc in pc_reply:
                    pc.norm_lc_sc(1/tmp)


            # pass-through here is list of post coordinates [tree #, level #, key val], of
            # posts skimmed in skim_..._lists, replies "read" in reply_list,
            # and locality score
            # tuples (polarity score, visbility score), for skim_list and reply_list,
            # bundled into lists of objects pc_skim, and pc_reply; normalized visibility
            # scores are in .nrm_loc_scores and .nrm_povs

            # assign agent-specific scores to the locality tuples; these help the agent
            # choose what posts to read, reply to, vote on, police, etc.
            agent_pol = agent.get_polarity()
            agent_spd = agent.get_spread()
            # "positive matches"--these are floats, bounded above by 1 (can be negative),
            # to reflect polar-relative favorability ratings for the posts considered
            for pc in pc_cur_skim:
                pc.pos = post_tup_score(agent_pol, agent_spd, pc.nrm_loc_scores[0],
                pc.nrm_loc_scores[1])
            #for pc in pc_oth_skim:  # maybe not needed--"oth" is for policing and
                # altruism, which focus in "neg" (not "pos") scoring
            #    pc.pos = post_tup_score(agent_pol, agent_spd, pc.nrm_loc_scores[0],
            #    pc.nrm_loc_scores[1])
            for pc in pc_reply:
                pc.pos = post_tup_score(agent_pol, agent_spd, pc.nrm_loc_scores[0],
                pc.nrm_loc_scores[1])
            # "negative matches"--these also floats <= 1 (can be negative), to reflect
            # polar-opposite ratings for posts (altruism and possibly policing);
            # modify agent polarity to "opposite"
            '''
            # alternate handling of ~full neutrals
            if abs(agent_pol) < 0.15:
                mod_pol = (-0.8,0.8)    # then "handle" this in the either functions
                # or functions preparation below (calls to post_tup_score and 
                # post_police_score)--would call the function twice, once with each
                # polarity extreme, then take the max score of the results of the
                # (2) function calls
            elif abs(agent_pol) <= 0.5: ...
            '''
            if abs(agent_pol) <= 0.5:
                mod_pol = -np.sign(agent_pol)*0.5
            else:
                mod_pol = -np.sign(agent_pol)*0.75
            # modify agent spread, to "soften" for opposition filter (ie make
            # spread wider/fatter)
            mod_spd = agent_spd + (NM_HI-agent_spd)/2
            #for pc in pc_cur_skim:  # maybe not needed; "cur" is for ri bucket,
                # and focuses on "pos" (not "neg") scores
            #    pc.neg = post_tup_score(mod_pol, mod_spd, pc.nrm_loc_scores[0],
            #    pc.nrm_loc_scores[1])
            for pc in pc_oth_skim:
                pc.neg = post_tup_score(mod_pol, mod_spd, pc.nrm_loc_scores[0],
                pc.nrm_loc_scores[1])
            for pc in pc_reply:
                pc.neg = post_tup_score(mod_pol, mod_spd, pc.nrm_loc_scores[0],
                pc.nrm_loc_scores[1])

            # note that for policing, locality scores may not be optimal--rather
            # police based on the post by itself (without averaging over eg parent and
            # children)
            for pc in pc_oth_skim:
                pc.police = post_police_score(mod_pol, mod_spd, *pc.nrm_povs)

            # pass-throughs:
            # pc_skim and pc_reply: objects with () post coordinates [tree #, level #,
            # key], () locality scores (polarity, visibility), () pos/neg float scores
            # to each post, weighing both polarity and visibility at the same time

            # ***
            # decide what posts to read (From pc_skim and pc_reply), depending on
            # purpose
            # ***

            # decide what posts in skim_list go into reward intent, altruism, and policing
            # (the posts in reply_list are handled differently--all of these are assumed
            # read, and can just "skip" replies ahead to deciding commenting/voting)
            # eg:
            # reward intent: 10+20*general_reading
            # altruism: 5*altruism
            # policing: 10*policing (say these don't overlap with altruism picks)

            #print("starting with %s skim posts" % len(pc_skim))  # DEBUG

            # DEBUG
            # the PC objects contain nrm_povs, for single / parent posts, and
            # nrm_loc_scores, for locality averages
            '''
            if len(pc_skim)>5:
                pst_sts = [(post.nrm_loc_scores,post.pos) for post in pc_skim]
                print("agent polarity: (%s,%s); total (loc) polarities: %s" %
                      (agent.get_polarity(), agent.get_spread(),
                       sorted(pst_sts, key=lambda x: x[1], reverse=True)))
            '''

            ri_posts, po_posts, al_posts = reading_choices(pc_cur_skim,
                                        pc_oth_skim, agent, agent_pol_hst)

            # DEBUG
            '''
            if len(ri_posts)>0:
                print("resulting (loc) polarities in ri_posts: %s" % [
                    (post.nrm_loc_scores,post.pos)
                                    for post in ri_posts])
            '''

            # pass-throughs: pc_skim, pc_reply--from earlier;
            # ri_posts, po_posts, al_posts--these will be the posts (PostContainer
            # objects) that the agent has "read" (along with pc_reply)

            # check for duplicates between reply_list and the ri/po/al bucket
            set_skim = set([tuple(x.coord) for x in ri_posts+po_posts+al_posts])
            set_reply = set([tuple(x.coord) for x in pc_reply])
            tmp_overlap = list(set_skim & set_reply)  # which coordinates, if any,
            # these two sets have in common (coords in tuple form);
            # favor the ri/po/al list over the reply list:
            if len(tmp_overlap) > 0:        # if the set intersection is not empty
                pc_reply = [x for x in pc_reply if tuple(x.coord) not in tmp_overlap]

            # pass-throughs
            # ri_posts--reward-intent posts; scored via .pos
            # po_posts--policing posts; scored via .police
            # al_posts--altruism posts; scored via .neg
            # pc_reply--posts that have directly replied to a post of the agent;
            # scored via .pos and .neg

            # decide on what posts to comment and vote on; then do so
            # for ri/po/al

            # for vote to comment ratio adjustment as the trees get more complex;
            # both com_mlt and vot_mlt will be in [0,1]
            tr_av_ps = trees_avg_posts([tree_list[ix] for ix in active_trees])
            NRM_TSH = 50        # normalization threshold for average number
            # of posts per tree
            prp = tr_av_ps/NRM_TSH
            if prp > 1.0:
                prp = 1.0
            # for comments--adjust downward as trees get more complex--
            # asymptotically could be ca 0.25
            if agent.comment_propensity <= 0.5:
                slp = 1-prp/2
                com_mlt = slp*agent.comment_propensity
            else:
                slp = 1+prp/2
                com_mlt = slp*agent.comment_propensity-prp/2
            # for voting--this could just be constant adjusted (no variation wrt
            # prp); say center this at around 0.75
            if agent.general_voting <= 0.5:
                slp = 3/2
                vot_mlt = slp*agent.general_voting
            else:
                slp = 1/2
                vot_mlt = slp*agent.general_voting+1/2

            # for comments in ri
            num_ri = sum(np.array([random.random() for _ in
                                range(len(ri_posts))]) <= com_mlt)
            tmp_ri = []
            if num_ri > 0:
                ri_ix = random.sample(range(len(ri_posts)),num_ri)
                tmp_ri = [ri_posts[ii] for ii in ri_ix]
                agent_comment(tmp_ri, tree_list, agent, agent_dict, round_num,
                              agent_track_dct[agent.get_id()][1])
                #print("commenting on %s ri posts" % num_ri)  # DEBUG
            else:   # DEBUG
                pass
                # print("no reward intent posts selected for commenting")

            # for votes in ri
            num_ri = sum(np.array([random.random() for _ in
                                range(len(ri_posts))]) <= vot_mlt)
            tmp_ri = []
            if num_ri > 0:
                ri_ix = random.sample(range(len(ri_posts)),num_ri)
                tmp_ri = [ri_posts[ii] for ii in ri_ix]
                agent_vote(tmp_ri, tree_list, agent, ep_dict,
                           agent_track_dct[agent.get_id()][2])
                #print("voting on %s ri posts" % num_ri)
            else:   # DEBUG
                pass
                # print("no reward intent posts selected for voting")

            # for policing (assume it's only voting)
            num_po = sum(np.array([random.random() for _ in
                                   range(len(po_posts))]) <= vot_mlt)
            tmp_po = []
            if num_po > 0:
                po_ix = random.sample(range(len(po_posts)),num_po)
                tmp_po = [po_posts[ii] for ii in po_ix]
                agent_vote(tmp_po, tree_list, agent, ep_dict,
                           agent_track_dct[agent.get_id()][4])
                #print("voting on %s po posts" % num_po)
            else:   # DEBUG
                pass
                # print("no policing posts selected for voting")

            # for comments in al
            num_al = sum(np.array([random.random() for _ in
                                range(len(al_posts))]) <= com_mlt)
            tmp_al = []
            if num_al > 0:
                al_ix = random.sample(range(len(al_posts)),num_al)
                tmp_al = [al_posts[ii] for ii in al_ix]
                agent_comment(tmp_al, tree_list, agent, agent_dict, round_num,
                              agent_track_dct[agent.get_id()][3])
                #print("commenting on %s al posts" % num_al)
            else:   # DEBUG
                pass
                # print("no altruism posts selected for commenting")

            # for votes in al
            num_al = sum(np.array([random.random() for _ in
                                   range(len(al_posts))]) <= vot_mlt)
            tmp_al = []
            if num_al > 0:
                al_ix = random.sample(range(len(al_posts)),num_al)
                tmp_al = [al_posts[ii] for ii in al_ix]
                agent_vote(tmp_al, tree_list, agent, ep_dict,
                           agent_track_dct[agent.get_id()][4])
                #print("voting on %s al posts" % num_al)
            else:   # DEBUG
                pass
                # print("no altruism posts selected for voting")

            # for comments in replies;
            # choose number of replies to make:
            num_rp = sum(np.array([random.random() for _ in
                                range(len(pc_reply))]) <= RE_ATTN*com_mlt)
            tmp_rp = []
            if num_rp > 0:
                rp_ix = random.sample(range(len(pc_reply)),num_rp)
                tmp_rp = [pc_reply[ii] for ii in rp_ix]
                agent_comment(tmp_rp, tree_list, agent, agent_dict, round_num,
                              agent_track_dct[agent.get_id()][5], rep_com="reply")
                #print("commenting on %s replies" % num_rp)
            else:   # DEBUG
                pass
                # print("no reply posts selected for commenting")

            # for votes in replies
            num_rp = sum(np.array([random.random() for _ in
                                   range(len(pc_reply))]) <= RE_ATTN*vot_mlt)
            tmp_rp = []
            if num_rp > 0:
                rp_ix = random.sample(range(len(pc_reply)),num_rp)
                tmp_rp = [pc_reply[ii] for ii in rp_ix]
                agent_vote(tmp_rp, tree_list, agent, ep_dict,
                           agent_track_dct[agent.get_id()][6])
                #print("voting on %s replies" % num_rp)
            else:   # DEBUG
                pass
                # print("no reply posts selected for voting")

            # compute average polarity from "read" posts, say from ri_posts, and
            # pc_reply (policing and altruism may have less of a polarity feedback
            # effect)
            tot_pol = 0.0
            for pc in ri_posts:
                tot_pol += (tree_list[pc.coord[0]].get_post(pc.coord[1],pc.coord[
                2]).get_polarity())
            for pc in pc_reply:
                tot_pol += (tree_list[pc.coord[0]].get_post(pc.coord[1],pc.coord[
                2]).get_polarity())
            tot_red = len(ri_posts) + len(pc_reply)
            if tot_red > 0: # nothing to do if no posts read in select categories
                agent.set_feedback(round_num, (tot_pol/tot_red,tot_red))
                agent.comp_feedback(round_num)  # compute new polarity score

            # update agent's trees "read"
            agent.add_trees_read(tre_ixs)

            #print("agent %s done" % agent_now_id)     # DEBUG

            # END loop over agents

        #show_agents(agent_dict)     # DEBUG
        #print("agents net vote totals this round: %s" % ep_dict)  # DEBUG; this
        # will show net votes over all agents' posts

        # DEBUG--as checksum for per-agent vote totals
        for key in ep_dict:
            if key not in net_vot_dct:
                net_vot_dct[key] = ep_dict[key][0]
            else:
                net_vot_dct[key] += ep_dict[key][0]

        #show_tree_details(tree_list)
        # DEBUG
        if VERBOSE:
            vts_tup = show_tree_votes(tree_list)      # displays total
        # upvote,
        # downvote
        # over
        # all
        # trees

        # record agent net upvote-downvote values this round (0 by default)
        for key in agent_dict.keys():
            if ep_dict[key][0] != (0.0,0.0):
                agent_dict[key].set_ep(ep_dict[key], round_num)

        # DEBUG
        # update agent polarities in agent_track_dct (from possible feedback
        # changes)
        for key in agents_avail:  # only makes sense for non-retired agents
            agent = agent_dict[key]
            agent_track_dct[key][0][0] = agent.get_polarity()  # ie this fetches
            # performance polarity (vs base polarity)

        # retire trees;
        # can have two mechanisms:
        # (1) a hard-coded complexity limit, based on eg
        # number of posts and/or net upvotes, and
        # (2) agent saturation limit--if the tree has been read by ca 80% of agents
        # then it's likely to be retired (eg)
        rem_trs = []  # list of tree indices in tree_list to remove from active_list
        avg_ppt = 0.0    # track average # of posts per tree
        bef_ret = len(active_trees)     # how many active trees initially
        # total post count:
        for ix in active_trees:
            cutoff = np.random.normal(co_mn,co_sd) # eg simple complexity cutoff and
            # score, based on total number of posts in the tree
            tree = tree_list[ix]
            nm_ps = tree.get_complexity()[1]   # number of posts in tree
            avg_ppt += nm_ps
            if nm_ps > cutoff:  # cutoff after some # of posts total
                rem_trs.append(ix)
        # agent saturation (note this may be computationally expensive):
        read_dict = {}  # dictionary with entries, tree index : agent read count
        for key in agent_dict:
            ts_rd = agent_dict[key].trees_read  # indices in tree_list of trees
            # read by this agent
            for ix in ts_rd:
                if ix in read_dict:
                    read_dict[ix] += 1
                else:
                    read_dict[ix] = 1
        for key in read_dict:   # keys here are indices of (some) trees in tree_list
            cutoff = np.random.normal(0.8,0.05)
            if cutoff < 0.7:
                cutoff = 0.7
            elif cutoff > 0.95:
                cutoff = 0.95
            if read_dict[key]/len(agents_avail) > cutoff:
                if key not in rem_trs:  # if not already removed
                    rem_trs.append(key)   # add "full" tree to trees to remove
        for ix in rem_trs:  # record trees retired in this round for trees_data:
            tr_ex_dc[ix] = [tr_ex_dc[ix][0], True]  # True = is retired
        active_trees = [ix for ix in active_trees if ix not in rem_trs]
        aft_ret = len(active_trees)
        num_ret = bef_ret-aft_ret       # number of trees actually retired, from
        # active_trees
        avg_ppt /= len(active_trees)

        # create new trees, to "replace" trees read by agents;
        # note, tree_list is a list of tree objects; active_trees is a list of
        # indices of active trees in tree_list

        # (i) original method--this relied on a steady state estimate
        #avg_num_trees_totally_read = int(np.round(num_trees_read / len(
        # active_agents)))
        #new_trs = [create_tree() for _ in range(avg_num_trees_totally_read)]

        # (ii) modifed method--just eg replace any trees that get retired; note this
        # may be more practical if agents are more likely to select popular trees--
        # that may ensure enough asymmetry in tree post counts there will be a
        # full spectrum of tree complexities when a tree is retired
        new_trs = [create_tree() for _ in range(num_ret)]

        ln_tl = len(tree_list)
        tree_list += new_trs
        active_trees += list(range(ln_tl, ln_tl+len(new_trs)))

        #print("%s trees created and %s retired at round %s" %
        #      (num_ret, num_ret, round_num))   # DEBUG
        trees_retired += num_ret
        #print("%s posts per tree, on average" % avg_ppt)   # DEBUG

        # DEBUG
        # display active / current trees (not retired trees):
        '''
        if round_num==NUM_RDS-1:
            trees_polarity_plot([tree_list[ix] for ix in active_trees],
                                tre_ixs=active_trees)
            trees_complexity_plot([tree_list[ix] for ix in active_trees],
                                  tre_ixs=active_trees)
            # agents_plot([tree_list[ix] for ix in active_trees])
            # note for agents upvote/downvote split plot, it's prlly better to show
            # over all trees (retired or not)--since all trees do contribute to the
            # agent votes counts
            agents_plot(tree_list)
        '''

        # retire agents
        ell_ags = [key for key in agents_avail if ag_rs_ct[key]>agent_ret_tsh]
        # which agent ids are eligibile--they need to be above activity
        # threshold, and not already be retired
        agent_engage = []
        for key in ell_ags:
            net_vot = agent_dict[key].get_net_votes()  # net upvote-downvote for
            # this agent over their lifetime
            pl = agent_dict[key].get_polarity()  # DEBUG
            ls_sl = agent_track_dct[key]
            tot_com = ls_sl[1][1]+ls_sl[3][1]+ls_sl[5][1]
            log_toc = np.log(tot_com) if tot_com >= 2 else 1.0  # log of total
            # comment count made by this agent
            #agent_engage.append((key,net_vot/log_toc))  # normalizing by log of
            # total comments is optional
            agent_engage.append((key, net_vot / ag_rs_ct[key], pl)) #
            # normalize by number of rounds the agent has been active
        fin_to_ret = 0
        if len(ell_ags) > 0:
            ag_st = sorted(agent_engage,key=lambda x: x[1])  # sorted (id:votes)
            # tuples in increasing order by "votes" score--so least-positive
            # agents are first in the list
            num_to_ret = NUM_AGS*CHURN_PROP
            remdr = num_to_ret - np.round(num_to_ret)
            rnd_efc = 1 if random.random()<remdr else 0
            fin_to_ret = int(num_to_ret)+rnd_efc

            # DEBUG
            if fin_to_ret > 0:
                st_ls = [str(tup[2]) for tup in ag_st]
                print(", ".join(st_ls[:fin_to_ret])
                  + " || " + ", ".join(st_ls[fin_to_ret:]))
            tp_co = int(0.15 * len(ag_st))
            ag_rt_tk.append(np.mean([abs(tup[2]) for tup in ag_st[:tp_co]]))
            ch_pl += sum([abs(tup[2]) for tup in ag_st[:fin_to_ret]])  # sums
            # of abs(pol) of all retired agents (cumulative)

            for ij in range(fin_to_ret):  # record first fin_to_ret elements in
                # sorted ag_st list (ie agent id's)
                agent_id_ret.append(ag_st[ij][0])
        if VERBOSE:
            print("%s agents retired at round %s" % (fin_to_ret,round_num))
        cur_ags = len(agent_dict)
        for ij in range(fin_to_ret):    # create new agents (same number as retired)
            new_id = cur_ags+ij
            agent_dict[new_id] = create_agent(new_id)
            agent_track_dct[new_id] = ([[agent_dict[new_id].get_polarity(),
                                        agent_dict[new_id].base_polarity]] +
                                    [[None, 0] for _ in range(6)])
            agent_pol_hst[new_id] = []
            ag_rs_ct[new_id] = 0  # agent rounds counter dictionary

        # END loop over rounds

    # DEBUG, curation
    '''
    by_np = [tree.num_posts for tree in tree_list]
    ix = np.argmax((np.array(by_np)==max(by_np)))
    print(tree_list[ix])
    cu_ls, ot_ls = tree_list[ix].curation_score(2,0.5)
    print("curation kept:")
    print(cu_ls)
    cl = [tree_list[ix].get_post(*sbl[0]).get_polarity() for sbl in cu_ls]
    print("polarities: %s " % cl)
    '''

    # calculate tree uniformity scores
    #tr_pv_ls = tree_uniformity_scores(tree_list)     # returns a list of same
    # length as tree_list with associated p-val from uniformity / goodness of fit
    # (Chi-squared) whenever trees met minimum complexity;
    # show uniformity / homophily scores:
    hom_dct = tree_uniformity_scores_2(tree_list)
    # hom_dct holds results: hom_cto=low_posts; tsh_cnt=number of trees that made
    # the post count cutoff; aav_pol; pln_prp; smp_prp; raw_pol--these last 3
    # of 4 as the "usual"
    # homophily measures; note if no sufficiently complex trees, then these last
    # three are set to np.nan's

    # show active agent activity by polarity bins
    bin_cts, avg_stats = agents_activity_track_2(tree_list, agent_track_dct,
                                                 agent_id_ret)
    stats_sbl = list(zip(*avg_stats))
    bn_rt_cs = proc_ret_agents(agent_dict, agent_id_ret)  # list of agent counts by
    # polarity; these will all be zero if no agents churned
    if VERBOSE:
        print("active agent count by polarity bin: %s" % bin_cts)
        print("avg active agent activities by polarity:")
        print("upvotes: %s" % list(stats_sbl[0]))
        print("downvotes: %s" % list(stats_sbl[1]))
        print("comments posted: %s" % list(stats_sbl[2]))
        print("votes made: %s" % list(stats_sbl[3]))
        print("polarity drift: %s" % list(stats_sbl[4]))
        # print("avg agent activity by polarity bin (upvt,dnvt,com,vot): %s" %
        # avg_stats)
        print("retired agents polarity frequencies: %s" % bn_rt_cs)

    # DEBUG
    # since trees can be retired, show complexities over all trees (active or not):
    if PLOTS_ON:
        tfq_lst = trees_polarity_plot(tree_list)
        trees_complexity_plot(tree_list)
    # agents_plot([tree_list[ix] for ix in active_trees])
    # note for agents upvote/downvote split plot, it's prlly better to show
    # over all trees (retired or not)--since all trees do contribute to the
    # agent votes counts
    if PLOTS_ON:
        agents_plot(tree_list, agents_avail) # [key for key in agent_dict])

    if VERBOSE:
        print("total net votes for each agent: %s" % net_vot_dct)

    # DEBUG
    # agent activity as a function of polarity; simple scatterplot
    avg_ri_pa_re = agents_activity_track(agent_track_dct, active_agents, PLOTS_ON)
    # agent-wide average distribution of ri, po/al, reply, comment+vote


    # plot agent polarity stats, re agent scores and posts read; fancy bar plot,
    # with horizontal ticks representing values
    pol_dsp, avg_dsp = agents_polarity_track(agent_track_dct, net_vot_dct,
                                             agents_avail, PLOTS_ON) #
    # pol_dsp dict
    # is the polarization disparity between ri read bucket polarization and the
    # performative agent polarization score at simulation end (so this is somewhat
    # slippery, depending on how flexible the agent is); keyed by agent id--
    # (agent performative polarity, average ri comment and vote polarity);
    # avg_dsp--this is average disparity between the agent's polarity and that
    # agent's reward intent / ri comment/vote posts (ie the polarity of the posts
    # the agent is commenting or voting on, with the "intent of reward")

    # check total number of posts over all trees *not including OPs / roots*
    tot_pst = 0
    for tree in tree_list:
        nm_ps = tree.get_complexity()[1]
        if nm_ps > 1:
            tot_pst += nm_ps-1

    if VERBOSE:
        print("total number of comment posts over all trees check (not including "
          "root): %s" % tot_pst)
        # DEBUG
        print("total number of trees retired / created over the whole %s rounds: %s" %
          (NUM_RDS,trees_retired))
        print("total number of agents retired / created over the whole %s rounds: %s" %
          (NUM_RDS,len(agent_dict)-len(agents_avail)))

    # DEBUG
    if PLOTS_ON:
        agents_polarity_history(agent_pol_hst, agent_dict, agents_avail)
    '''
    ag_id = 0
    print("agent_pol_hst for agent %s, pol=%s: %s" % (ag_id,agent_dict[
        ag_id].get_polarity(),agent_pol_hst[
        ag_id]))
    '''

    # DEBUG
    # for agent churn diagnostics
    print(ag_rt_tk)
    ag_ls_dg = []
    # ag_rs_ct--this is the dictionary by agent key, counting # rounds active
    for key in agent_dict:
        agent = agent_dict[key]
        ag_ls_dg.append((sum(agent.extrinsic_points)/ag_rs_ct[key],
                         sum(agent.downvotes)/ag_rs_ct[key],
                         agent.get_polarity()))  #abab
    bin_edges = np.linspace(-1,1,10)
    #hist, _ = np.histogram([tup[2] for tup in ag_ls_dg], bins=bin_edges)
    bin_indices = np.digitize([tup[2] for tup in ag_ls_dg], bins=bin_edges)
    df_av = pd.DataFrame(ag_ls_dg,columns=["net","dnv","pol"],
                         index=bin_indices)
    print(df_av.groupby(df_av.index).mean())
    print("initial polarity frequencies: %s" % str(ini_pol_sav))  # ie this...
    print("final bin counts (end sim, non-retired agents): %s" % str(bin_cts))
    print("avg pol churned agents: %s" % (ch_pl/len(agent_id_ret)))
    #plt.bar(x=range(len(ag_rt_tk)),height=ag_rt_tk)
    #plt.show()

    # record any (remaining) data:

    # for session data
    session_data["id"] = time.time()
    # session-level control parameters:
    session_data["num_rds"] = NUM_RDS
    session_data["num_trs"] = NUM_TRS  # number of trees to start
    session_data["num_ags"] = NUM_AGS  # total number of agents to start
    session_data["agt_prp"] = agent_proportion  # proportion of agents active
    # each round
    session_data["SIG"] = Tree.SIG #MODULE_SIG   # Tree class in-camp
    # "pickiness"
    session_data["po_attn"] = PO_ATTN  # control for policing tendency
    session_data["al_attn"] = AL_ATTN  # control for altruism tendency
    session_data["re_attn"] = RE_ATTN  # control for reply tendency
    session_data["sensitivity"] = MODULE_SENSITIVITY  # how much agent polarity
    # "drift" is allowed, depending on their environment
    session_data["churn_prop"] = CHURN_PROP
    session_data["nm_lo"] = NM_LO
    session_data["nm_hi"] = NM_HI
    session_data["vot_rad"] = VOT_RAD
    session_data["tree_retire"] = True  # whether trees are retired
    session_data["agent_retire"] = CHURN_PROP>0.0    # whether agents are
    # retired/churned

    # session results:
    vts_tup = show_tree_votes(tree_list,disp=False)  # (upvotes, downvotes)
    # totals over all trees
    session_data["tot_up"] = vts_tup[0] # upvotes total over all trees
    session_data["tot_dn"] = vts_tup[1] # downvotes total over all trees
    session_data["avg_ri"] = avg_ri_pa_re[0]  # average reward intent
    # comment+vote activity, over all agents
    session_data["avg_po_al"] = avg_ri_pa_re[1]  # average police/altruism
    # comment+vote activity, over all agents
    session_data["avg_re"] = avg_ri_pa_re[2]  # average reply comment+vote
    # activity, over all agents
    session_data["ri_disp"] = avg_dsp if avg_dsp != "NA" else np.nan   # this is
    # average disparity between the
    # agent's polarity and that agent's reward intent / ri comment/vote posts (
    # ie the polarity of the posts the agent is commenting or voting on, with the
    # "intent of reward"); this I suppose could be as large as 2 (eg agent is -1
    # polarity, while ri posts they interact with are all polarity +1); note,
    # this can be NA, which for dataframe purposes will convert to Pandas
    # standard blank cell, numpy's nan;
    # polarity bin counts (number of agents per polarity bin):
    for jj in range(len(bin_cts)):
        session_data["agc_bin_"+str(jj)] = bin_cts[jj]  # ie agent count bin 0,1,...
    # record homophily measure stats:
    session_data["hom_cto"] = hom_dct["hom_cto"]  # post count cutoff used for
    # trees to be eligible for homophily scoring
    session_data["tsh_cnt"] = hom_dct["tsh_cnt"]   # how many trees from the session
    # met
    # post count threshold
    session_data["aav_pol"] = hom_dct["aav_pol"]   # simple average of
    # intra-tree polarities, which then has the average of those absolute
    # values taken
    session_data["pln_prp"] = hom_dct["pln_prp"]  # simple proportion of
    # majority-side
    # posts (<>0.0 polarity) to total posts, averaged over all trees
    session_data["smp_prp"] = hom_dct["smp_prp"]   # "weighted" form of pln_prp,
    # that takes polarity of posts into account (just sums abs(pol) for
    # polarities on each "side")
    session_data["raw_pol"] = hom_dct["raw_pol"]  # plain average, over all
    # trees that made the complexity cutoff for the homophily measure; this
    # can show if the most popular trees are "one sided" re polarity
    # average agent activity by polarity bin:
    nm_ar = ["upv_bin_","dnv_bin_","com_bin_","vot_bin_","dft_bin"]
    for jj in range(5):
        for kk in range(len(bin_cts)):
            session_data[nm_ar[jj]+str(kk)] = stats_sbl[jj][kk]
    # number of retired agents by polarity bin:
    for jj in range(len(bn_rt_cs)):
        session_data["rt_pl_"+str(jj)] = bn_rt_cs[jj]
    #session_data["tr_cr"] = trees_retired # trees created (these may be equal,
    # for steady state)
    session_data["tr_rt"] = trees_retired  # trees retired
    session_data["ag_rt"] = len(agent_id_ret) # len(agent_dict)-len(
    # active_agents)--OLD--this is not correct--active_agents are those
    # active for just the past round, and eg may always be
    # agent_prop*num_agents; agents retired (and created)
    session_data["tot_com_pst"] = tot_pst   # total number of comments made,
    # across all trees over whole simulation (not including root / OP)

    df_session = pd.DataFrame([session_data])

    # DEBUG
    #print([session_data["agc_bin_"+str(ii)] for ii in range(6)])
    print(session_data)
    print("retirement polarity bins: %s" % str(bn_rt_cs))

    file_prefix = "/home/wot/projects/reddit_simulation/simulation_data/"
    now = datetime.datetime.now()

    file_append = True  # for appending to an existing file--the filename needs
    # to already have been created in the appropriate directory

    if RECORD_DATA:
        if file_append:
            filename = FILENAME_DATA
            try:
                df_in = pd.read_csv(file_prefix+filename)
            except pd.errors.EmptyDataError:
                print("empty file for appending; creating blank dataframe")
                df_in = pd.DataFrame()
            df_tot = pd.concat([df_in, df_session], axis=0, ignore_index=True)
            df_tot.to_csv(file_prefix+filename,index=False)
        else:
            filename = now.strftime("%Y-%m-%d_%H-%M-%S")
            df_session.to_csv(file_prefix+filename+"_session_dat.csv", index=False)

    return bn_rt_cs, ini_pol_sav, bin_cts


if __name__ == '__main__':

    tup_lst = []

    for ii in range(50):
        ret_frq, ini_pol_frq, fin_pol_frq = run_simulation()
        #print(ini_pol_frq)
        #print(fin_pol_frq)
        ll = sum(ini_pol_frq[:2])
        rr = sum(ini_pol_frq[4:])
        l1 = np.log(ll/rr)
        ll = sum(fin_pol_frq[:2])
        rr = sum(fin_pol_frq[4:])
        l2 = np.log(ll/rr)
        #print("%s: %s" % (l1,l2))
        tup_lst.append((ret_frq,ini_pol_frq,fin_pol_frq,(l1,l2)))

    print(tup_lst)