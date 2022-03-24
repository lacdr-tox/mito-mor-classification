#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:43:00 2017

@author: yangh
"""
import numpy as np
from sklearn import mixture
def data2gmm(N_components, tempmatrix, Ngmmlist):
    np.random.seed(np.int(0))
    gmm = mixture.GaussianMixture(n_components=N_components, covariance_type='full', init_params='random', random_state=np.int(0), max_iter=1000,verbose=0);
    gmm.fit(tempmatrix)
    bestgmm = gmm
    # Ngmmlist = 10000;
    gmmlist = [gmm]*Ngmmlist
    Ngmmlist = range(Ngmmlist)
    for i in np.array(Ngmmlist):
        np.random.seed(np.int(i+1))
        #np.int(i+1+j*1e6)
        gmmlist[i] = mixture.GaussianMixture(n_components=N_components, covariance_type='full', init_params='random', random_state=np.int(i+1), max_iter=1000,verbose=0);
        gmmlist[i].fit(tempmatrix);
        #print gmm.bic(tempmatrix)
        if gmmlist[i].bic(tempmatrix) < bestgmm.bic(tempmatrix):
            bestgmm = gmmlist[i];
    return  bestgmm, gmmlist
####
import itertools
from multiprocessing import Pool
def data2gmm_s(a,b):
    index = a
    tempmatrix, N_components = b
    gmm = mixture.GaussianMixture(n_components=N_components, covariance_type='full', init_params='random', random_state=np.int(index+1), max_iter=1000,verbose=0);
    gmm.fit(tempmatrix)
    return gmm
def data2gmmstar(a_b):
    return data2gmm_s(*a_b)
def data2gmmp(N_components, tempmatrix, Ngmmlist,verboseF=0):
    #
    # parallized version
    #
    np.random.seed(np.int(0))
    gmm = mixture.GaussianMixture(n_components=N_components, covariance_type='full', init_params='random', random_state=np.int(0), max_iter=1000, verbose=verboseF);
    gmm.fit(tempmatrix)
    bestgmm = gmm
    # Ngmmlist = 10000
    gmmlist = [gmm]*Ngmmlist
    Ngmmlist = range(Ngmmlist)
    #      
    N_clist = Ngmmlist  
    others = [tempmatrix, N_components]
    Ncpu = 55
    pool = Pool(Ncpu)      
    result_multipleprocessing = pool.map(data2gmmstar, itertools.izip(N_clist, itertools.repeat(others, len(N_clist)))) 
    pool.close()
    pool.join()
    # extract the set of outcomes 
    for i in range(len(result_multipleprocessing)):
        gmmlist[i] = result_multipleprocessing[i]    
    gmmbiclist = [gmmlist[i].bic(tempmatrix) for i in range(0, len(result_multipleprocessing))]
    index = np.where(gmmbiclist == np.min(gmmbiclist))
    bestgmm = gmmlist[index[0][0]]
    return  bestgmm, gmmlist