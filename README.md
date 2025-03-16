# reddit_simulation

This is some code for simulating user interactions in a Reddit-like social media environment. The main features of this agent-based model include,
- there is a single dimension of polarization, quantified by values in the range [-1,1], that is assigned to agents and posts
- the basic container of user interaction is a Reddit-like thread in the form of a tree, with an original post (OP) at the tree's root, and zero or more reply posts with successive parent-child relationships
- agents "read" and selectively reply to and/or vote on posts from other agents, or the OP
- agent activity with respect to polarization falls into two categories
  - on-side posts--these are posts with polarities that are "close," under some metric, to the agent's own polarity
  - off-side posts--these are posts with polarities that are "far," under some metric, from the agent's own polarity
- the simulation is conducted in rounds (e.g. 100 rounds), where some subset of active agents (e.g. 60 total agents), are active for that round, and may post replies or upvote / downvote posts in any of the existing trees (the number of trees varies flexibly round-to-round, to ensure there are enough "fresh" trees so that agents do not re-read trees they have already reviewed)

General notes on the code:
- some of the code has not been factored to high standards, and in some areas is a bit sprawled out
- there is a lot of redundancy between the "base_version_downvoting_..." modules, which ideally could be consolidated under a single module with control points for the downvoting on/off and intra-dynamics options

This code is used in an informal study on homophily in social media, available on the author's [blog pages](https://tmwine.github.io/2025/01/30/social-media-Reddit-simulation.html).
