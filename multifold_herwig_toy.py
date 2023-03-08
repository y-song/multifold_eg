#!/usr/bin/env python
# coding: utf-8
# modified based on https://github.com/ericmetodiev/OmniFold/blob/master/OmniFold%20Demo.ipynb

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import root_numpy
import energyflow as ef
import energyflow.archs
import omnifold
import sys
import getopt
import random
import tensorflow as tf

print('Number of arguments:', len(sys.argv), 'arguments.')
seed = (int)(getopt.getopt(sys.argv[1:], "s")[1][0])
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def applyCut(inputDataframe, cut, text=None):
    dataframe = inputDataframe
    nbeforecut = dataframe.shape[0]
    cutDataframe = dataframe.query(cut)
    if text:
        print(text, cutDataframe.shape[0], ' fraction kept: %2.1f' % (
            100.0*float(cutDataframe.shape[0])/nbeforecut))
    return cutDataframe


def flatten(data, column):
    a = []
    for row in range(0, len(data)):
        a.append(data[row][column])
    return np.concatenate(a)


def getFlattenedData(branchNames, dfNames, inputFiles, treeName):
    data = root_numpy.root2array(inputFiles, treeName, branches=branchNames)
    a = []
    for branchName in branchNames:
        a.append(flatten(data, branchNames.index(branchName)))
    return pd.DataFrame(np.transpose(a), columns=dfNames)


def getData(branchNames, dfNames, inputFiles, treeName):
    data = root_numpy.root2array(inputFiles, treeName, branches=branchNames)
    return pd.DataFrame(data, columns=dfNames)


observables1 = ['pt', 'm', 'q', 'rg', 'zg', 'mg']
observabless1 = ['pts', 'ms', 'qs', 'rgs', 'zgs', 'mgs']
inputFiles = ['$HOME/output/run12/all0623_hadcorr05.root',
              '$HOME/output/run12/data0616.root']
treeName = ["MatchedTree", "ResultTree"]

df1 = applyCut(pd.read_csv('/vast/palmer/home.grace/ys668/output/run12/herwig_matched_1208.csv',
               low_memory=False), 'rg > 0 and rgs > 0')
df1.eval('relw = w/2.87272e-05', inplace=True)

df2 = applyCut(getFlattenedData(observables1, observabless1,
               inputFiles[1], treeName[1]), 'rgs > 0')
ptedges = [15, 20, 25, 30, 40, 60]
scaling = [0.96878842, 0.97606573, 0.97070026,
           0.97702706, 0.99731198]  # (1 - fake rate) for nom
w = np.ones(df2.shape[0])
for i in range(0, df2.shape[0]):
    for j in range(0, len(ptedges)-1):
        if (ptedges[j] < df2.iloc[i]['pts'] < ptedges[j+1]):
            w[i] *= scaling[j]
            break
df2['w'] = w

datasets = {'pythia': df1, 'data': df2}
itnum = 10
obs_multifold = ['pt', 'm', 'q', 'rg', 'zg', 'mg']
obs = {}

obs.setdefault('pt', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['pt'+s]),
})

obs.setdefault('q', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['q'+s]),
})

obs.setdefault('rg', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['rg'+s]),
})

obs.setdefault('zg', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['zg'+s]),
})

obs.setdefault('m', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['m'+s]),
})

obs.setdefault('mg', {}).update({
    'func': lambda dset, s: np.asarray(datasets[dset]['mg'+s]),
})

# calculate quantities to be stored in obs
for obkey, ob in obs.items():

    # calculate observable for GEN, SIM, DATA, and TRUE
    ob['genobs'], ob['simobs'] = ob['func'](
        'pythia', ''), ob['func']('pythia', 's')
    ob['dataobs'] = ob['func']('data', 's')

    print('Done with', obkey)

# set up the array of data/simulation detector-level observables
X_det = np.asarray([np.concatenate(
    (obs[obkey]['dataobs'], obs[obkey]['simobs'])) for obkey in obs_multifold]).T
Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(obs['pt']['dataobs'])),
                                                np.zeros(len(obs['pt']['simobs'])))))

# set up the array of generation particle-level observables
X_gen = np.asarray([np.concatenate(
    (obs[obkey]['genobs'], obs[obkey]['genobs'])) for obkey in obs_multifold]).T
Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(obs['pt']['genobs'])),
                                                np.zeros(len(obs['pt']['genobs'])))))

# standardize the inputs
X_det = (X_det - np.mean(X_det, axis=0))/np.std(X_det, axis=0)
X_gen = (X_gen - np.mean(X_gen, axis=0))/np.std(X_gen, axis=0)

# Specify the training parameters
# model parameters for the Step 1 network
model_layer_sizes = [100, 100, 100]  # use this for the full network size
det_args = {'input_dim': len(obs_multifold), 'dense_sizes': model_layer_sizes,
            'patience': 50, 'filepath': 'Step1_{}', 'save_weights_only': False,
            'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}

# model parameters for the Step 2 network
mc_args = {'input_dim': len(obs_multifold), 'dense_sizes': model_layer_sizes,
           'patience': 50, 'filepath': 'Step2_{}', 'save_weights_only': False,
           'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}

# general training parameters
fitargs1 = {'batch_size': 50000, 'epochs': 100,
            'verbose': 1}  # use this for a full training
fitargs2 = {'batch_size': 10000, 'epochs': 100,
            'verbose': 1}  # use this for a full training

# reweight the sim and data to have the same total weight to begin with
ndata, nsim = np.count_nonzero(Y_det[:, 1]), np.count_nonzero(Y_det[:, 0])
wdata = df2['w']
winit = ndata/nsim*np.asarray(df1['relw'])

# apply the OmniFold procedure to get weights for the generation
multifold_ws = omnifold.omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit,
                                 (ef.archs.DNN, det_args), (ef.archs.DNN, mc_args),
                                 fitargs1, fitargs2, val=0.2, it=itnum, trw_ind=-2)

path = '/vast/palmer/home.grace/ys668/analysis/run12/data/unfolding/valnum02_batch50000_10000/csvs/herwig_toy/'

# save output
if (os.path.isfile("jet_noweight.csv")==False):
    df3 = df1[obs_multifold]
    df3.to_csv(path+'jet_noweight.csv',index=False)
    
np.savetxt(path+'it1_seed'+str(seed)+'.csv', multifold_ws[2]/sum(multifold_ws[2]))
np.savetxt(path+'it2_seed'+str(seed)+'.csv', multifold_ws[4]/sum(multifold_ws[4]))
np.savetxt(path+'it3_seed'+str(seed)+'.csv', multifold_ws[6]/sum(multifold_ws[6]))
np.savetxt(path+'it4_seed'+str(seed)+'.csv', multifold_ws[8]/sum(multifold_ws[8]))
np.savetxt(path+'it5_seed'+str(seed)+'.csv', multifold_ws[10]/sum(multifold_ws[10]))
np.savetxt(path+'it6_seed'+str(seed)+'.csv', multifold_ws[12]/sum(multifold_ws[12]))
np.savetxt(path+'it7_seed'+str(seed)+'.csv', multifold_ws[14]/sum(multifold_ws[14]))
np.savetxt(path+'it8_seed'+str(seed)+'.csv', multifold_ws[16]/sum(multifold_ws[16]))
np.savetxt(path+'it9_seed'+str(seed)+'.csv', multifold_ws[18]/sum(multifold_ws[18]))
np.savetxt(path+'it10_seed'+str(seed)+'.csv', multifold_ws[20]/sum(multifold_ws[20]))