import os, sys, heapq, copy
import numpy as np
from collections import defaultdict

NBEST = 10
ROOT_ID = 0

class Hypo:
    def __init__(self, logp, edges, u, num_roots):
        self.logp = logp
        self.edges = edges
        self.u = u
        self.num_roots = num_roots


def cube_next(lhs_list, rhs_list, visited, priq,
        is_making_incomplete, u, k1, k2, new_uas, is_s_0 = False):
    if len(lhs_list) <= k1 or len(rhs_list) <= k2 or \
            (u, k1, k2) in visited:
        return
    visited.add((u,k1,k2))

    uas_logp = lhs_list[k1].logp + rhs_list[k2].logp
    if is_making_incomplete: # making incomplete hypothesis, adding an edge
        uas_logp += new_uas
        if is_s_0: # s == 0 and is making ('->', 0), must have ROOT relation
            las_logp = uas_logp 
            heapq.heappush(priq, (-las_logp,u,k1,k2, ROOT_ID))
    else:
        heapq.heappush(priq, (-uas_logp,u,k1,k2,None))


def cube_pruning(s, t, kk, memory, mb_parse_probs):
    if s == 0 and kk[0] == '<-': # artificial root can't be governed
        return

    key = (s,t) + kk

    hd, md = (s,t) if kk[0] == '->' else (t,s)
    new_uas = np.log(mb_parse_probs[md,hd])

    if kk[1] == 0:
        u_range = range(s,t)
        u_inc = 1
        ll, rr = ('->',1), ('<-',1)
    elif kk[1] == 1 and kk[0] == '<-':
        u_range = range(s,t)
        u_inc = 0
        ll, rr = ('<-',1), ('<-',0)
    else:
        u_range = range(s+1,t+1)
        u_inc = 0
        ll, rr = ('->',0), ('->',1)

    #print 'cube_pruning:', key, ll, rr

    ## initialize priority queue
    priq = []
    visited = set() # each item is (split_u, k1, k2)
    for u in u_range:
        lhs = (s,u) + ll
        rhs = (u+u_inc,t) + rr
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, 0, 0, new_uas, s==0)

    ## actual cube pruning
    nbest = []
    while len(priq) > 0:
        ### obtain the current best
        neglogp, u, k1, k2, li = heapq.heappop(priq)
        logp = -neglogp
        lhs = (s,u) + ll
        rhs = (u+u_inc,t) + rr
        edges = memory[lhs][k1].edges | memory[rhs][k2].edges
        num_roots = memory[lhs][k1].num_roots + memory[rhs][k2].num_roots
        if li != None:
            edges.add((md,hd,li))
            num_roots += (s == 0)
        ### check if violates
        is_violate = (num_roots > 1)
        j = -1
        for i, hyp in enumerate(nbest):
            #### hypotheses with same edges should have same logp
            if is_violate or hyp.edges == edges or \
                    (i == 0 and hyp.logp - logp >= 5.0):
                is_violate = True
                break
            if hyp.logp < logp:
                j = i
                break
        ### insert
        if is_violate == False:
            new_hyp = Hypo(logp, edges, u, num_roots)
            if j == -1:
                nbest.append(new_hyp)
            else:
                nbest.insert(j, new_hyp)
        if len(nbest) >= 3 * NBEST:
            break
        ### append new to priq
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, k1+1, k2, new_uas, s==0)
        cube_next(memory[lhs], memory[rhs], visited, priq,
                kk[1]==0, u, k1, k2+1, new_uas, s==0)
    memory[key] = nbest[:NBEST]


def eisner_dp_nbest(length, mb_parse_probs):
    memory = defaultdict(list)
    for i in range(0, length+1):
        for d in ('->', '<-'):
            for c in range(2):
                memory[(i,i,d,c)].append(Hypo(0.0, set(), None, 0))

    for t in range(1, length+1):
        for s in range(t-1, -1, -1):
            cube_pruning(s, t, ('<-',0), memory, mb_parse_probs)
            cube_pruning(s, t, ('->',0), memory, mb_parse_probs)
            cube_pruning(s, t, ('<-',1), memory, mb_parse_probs)
            cube_pruning(s, t, ('->',1), memory, mb_parse_probs)
    ## output nbest of memory[(0,length,'->',1)]
    for hyp in memory[(0,length,'->',1)]:
        print (hyp.edges, hyp.logp, hyp.num_roots)


def gen_tree(hyp):
    pass


if __name__ == '__main__':
    # do some unit test
    mb_parse_probs = np.arange(16).reshape((4,4))
    #mb_rel_probs = np.arange(48).reshape((4,4,3))
    #print(mb_parse_probs)
    #print(mb_rel_probs)
    eisner_dp_nbest(3, mb_parse_probs)