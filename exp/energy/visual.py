# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:30:45 2025

@author: user
"""

import pandas as pd
# import mpl_scatter_density
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
# from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# test_dict = {}
# with open('test_%s.txt'%dataset, 'r') as fp:
#     test = fp.readlines()
# for row in test:
#     tmp = row.split("\t")
#     if tmp[0] in test_dict:
#         test_dict[tmp[0]].append(tmp[1])
#     else:
#         test_dict[tmp[0]] = [tmp[1]]
# del test, tmp, row

# metric_dict = {}
# for row in result:
#     tmp = row.split(":")
#     prec = tmp[1].count("*") / 20
#     hitr = tmp[1].count("*") / len(test_dict[tmp[0]])
#     metric_dict[tmp[0]] = [prec,hitr]
# del tmp, row, prec, hitr

# #evaluation metrix
# x = []
# y_pred = []
# y_hitr = []
# for node in train_dict:
#     if node not in metric_dict:
#         continue
#     x.append(len(train_dict[node]))
#     y_pred.append(metric_dict[node][0])
#     y_hitr.append(metric_dict[node][1])
# del node
# plt.scatter(x, y_pred, color='hotpink')
# plt.scatter(x, y_hitr, color='#88c999')

def draw_entropy(dataset, i, ax):
    # data loading
    with open('DiffRec_%s.txt'%dataset.lower(), 'r') as fp:
        result = fp.readlines()
    result = result[1:]
    
    train_dict = {}
    with open('train_%s.txt'%dataset.lower(), 'r') as fp:
        train = fp.readlines()
    for row in train:
        if dataset in ['LastFM','Amazon-Beauty']:
            tmp = row.split("\t")
        elif dataset in ['Douban-Book','Yelp2018', 'Gowalla']:
            tmp = row.split(" ")
        if tmp[0] in train_dict:
            train_dict[tmp[0]].append(tmp[1])
        else:
            train_dict[tmp[0]] = [tmp[1]]
    del train, tmp, row
    
    #DiffRec curves
    eps=1e-6
    entropy_x_DiffRec = []
    entropy_y_DiffRec = []
    train_nodes = list(train_dict.keys())
    with open('reconstruct_DiffRec_%s'%dataset.lower(), 'rb') as fp:
        reconstruct_DiffRec = pickle.load(fp)
    reconstruct_DiffRec = np.maximum(reconstruct_DiffRec, eps) # avoid negative values
    reconstruct_DiffRec = reconstruct_DiffRec / reconstruct_DiffRec.sum(axis=1, keepdims=True)
    for i in range(len(reconstruct_DiffRec)):
        if (reconstruct_DiffRec[i]>1).any() or (reconstruct_DiffRec[i]<-1).any(): # abnormal point
            reconstruct_DiffRec[i] = np.nan_to_num(np.exp(reconstruct_DiffRec[i]))
            reconstruct_DiffRec[i] = reconstruct_DiffRec[i] / reconstruct_DiffRec[i].sum()
        entropy_x_DiffRec.append(-len(train_dict[train_nodes[i]])*1/len(train_dict[train_nodes[i]])*np.log2(1/len(train_dict[train_nodes[i]])))
        entropy_y_DiffRec.append((-reconstruct_DiffRec[i]*np.log2(reconstruct_DiffRec[i])).sum())
        
    #TV_Diff curves
    eps=1e-6
    entropy_x_TV_Diff = []
    entropy_y_TV_Diff = []
    train_nodes = list(train_dict.keys())
    with open('reconstruct_TV_Diff_%s'%dataset.lower(), 'rb') as fp:
        reconstruct_TV_Diff = pickle.load(fp)
    reconstruct_TV_Diff = np.maximum(reconstruct_TV_Diff, eps) # avoid negative values
    reconstruct_TV_Diff = reconstruct_TV_Diff / reconstruct_TV_Diff.sum(axis=1, keepdims=True)
    for i in range(len(reconstruct_TV_Diff)):
        if (reconstruct_TV_Diff[i]>1).any() or (reconstruct_TV_Diff[i]<-1).any(): # abnormal point
            reconstruct_TV_Diff[i] = np.nan_to_num(np.exp(reconstruct_TV_Diff[i]))
            reconstruct_TV_Diff[i] = reconstruct_TV_Diff[i] / reconstruct_TV_Diff[i].sum()
        entropy_x_TV_Diff.append(-len(train_dict[train_nodes[i]])*1/len(train_dict[train_nodes[i]])*np.log2(1/len(train_dict[train_nodes[i]])))
        entropy_y_TV_Diff.append((-reconstruct_TV_Diff[i]*np.log2(reconstruct_TV_Diff[i])).sum())
    
    # BPR curvas
    entropy_x_BPR = []
    entropy_y_BPR = []
    train_nodes = list(train_dict.keys())
    with open('emb_BPR_%s'%dataset.lower(), 'rb') as fp:
        [user_emb, item_emb] = pickle.load(fp)
    reconstruct_BPR = np.dot(user_emb, item_emb.T)
    reconstruct_BPR = np.exp(reconstruct_BPR)
    reconstruct_BPR = reconstruct_BPR / reconstruct_BPR.sum(axis=1, keepdims=True) 
    for i in range(len(reconstruct_BPR)):
        entropy_x_BPR.append(-len(train_dict[train_nodes[i]])*1/len(train_dict[train_nodes[i]])*np.log2(1/len(train_dict[train_nodes[i]])))
        entropy_y_BPR.append((-np.maximum(reconstruct_BPR[i],eps)*np.log2(np.maximum(reconstruct_BPR[i],eps))).sum())
    
    # LightGCN curvas
    entropy_x_LightGCN = []
    entropy_y_LightGCN = []
    train_nodes = list(train_dict.keys())
    with open('emb_LightGCN_%s'%dataset.lower(), 'rb') as fp:
        [user_emb, item_emb] = pickle.load(fp)
    reconstruct_LightGCN = np.dot(user_emb, item_emb.T)
    reconstruct_LightGCN = np.exp(reconstruct_LightGCN)
    reconstruct_LightGCN = reconstruct_LightGCN / reconstruct_LightGCN.sum(axis=1, keepdims=True) 
    for i in range(len(reconstruct_LightGCN)):
        entropy_x_LightGCN.append(-len(train_dict[train_nodes[i]])*1/len(train_dict[train_nodes[i]])*np.log2(1/len(train_dict[train_nodes[i]])))
        entropy_y_LightGCN.append((-np.maximum(reconstruct_LightGCN[i],eps)*np.log2(np.maximum(reconstruct_LightGCN[i],eps))).sum())
    
    sns.scatterplot(ax=ax, x=entropy_x_TV_Diff, y=entropy_y_TV_Diff, c='forestgreen', alpha=0.6)
    sns.scatterplot(ax=ax, x=entropy_x_BPR, y=entropy_y_BPR, c='gold', alpha=0.6)
    sns.scatterplot(ax=ax, x=entropy_x_LightGCN, y=entropy_y_LightGCN, c='lightcoral', alpha=0.6)
    sns.scatterplot(ax=ax, x=entropy_x_DiffRec, y=entropy_y_DiffRec, c='royalblue', alpha=0.6)
    
    data_entropy = pd.DataFrame({'Models':['BPR']*len(entropy_x_BPR)+['LightGCN']*len(entropy_x_LightGCN)+['DiffRec']*len(entropy_x_DiffRec)+['TV-Diff']*len(entropy_x_TV_Diff), \
                                'Original':entropy_x_BPR+entropy_x_LightGCN+entropy_x_DiffRec+entropy_x_TV_Diff, \
                                'Reconstructed':entropy_y_BPR+entropy_y_LightGCN+entropy_y_DiffRec+entropy_y_TV_Diff})
    
    sns.kdeplot(ax=ax, data=data_entropy, x='Original', y='Reconstructed', fill=True, hue='Models', palette=['gold','lightcoral','royalblue','forestgreen'], alpha=0.3, thresh=0.001, levels=50, bw_method=0.3)
    #sns.kdeplot(ax=ax, data=LightGCN_entropy, x='Original', y='Reconstructed', fill=True, hue='Models', palette=['lightcoral'], alpha=0.3, thresh=0.001, levels=50, bw_method=0.3)
    #sns.kdeplot(ax=ax, data=DiffRec_entropy, x='Original', y='Reconstructed', fill=True, hue='Models', palette=['royalblue'], alpha=0.3, thresh=0.001, levels=50, bw_method=0.3)
    #sns.kdeplot(ax=ax, data=TV_Diff_entropy, x='Original', y='Reconstructed', fill=True, hue='Models', palette=['forestgreen'], alpha=0.3, thresh=0.001, levels=50, bw_method=0.3)
    ax.set_xlabel('Original', fontsize=20)
    ax.set_ylabel('Reconstructed', fontsize=20)
    ax.set_title(r'%s'%(dataset), fontsize=25)
    # ax.set_title(r'%s, $\frac{\Delta S_D}{\Delta S_L}=%.3f$, $\frac{\Delta S_U}{\Delta S_L}=%.3f$'\
                    # %(dataset, (np.sum(entropy_y_DiffRec)-np.sum(entropy_x_DiffRec))/(np.sum(entropy_y_LightGCN)-np.sum(entropy_x_LightGCN)), \
                      # (np.sum(entropy_y_TV_Diff)-np.sum(entropy_x_TV_Diff))/(np.sum(entropy_y_LightGCN)-np.sum(entropy_x_LightGCN))), fontsize=13) 
    print(np.sum(entropy_y_BPR)-np.sum(entropy_x_BPR), np.sum(entropy_y_DiffRec)-np.sum(entropy_x_DiffRec), (np.sum(entropy_y_LightGCN)-np.sum(entropy_x_LightGCN)), \
          np.sum(entropy_y_TV_Diff)-np.sum(entropy_x_TV_Diff))
    sns.move_legend(ax, "lower left")
    ax.grid(linestyle='--',alpha=0.5)
    
    return ax, train_dict, data_entropy, reconstruct_DiffRec, reconstruct_LightGCN, reconstruct_TV_Diff, reconstruct_BPR

def draw_energy(dataset, i, ax):
    # data loading
    with open('DiffRec_%s.txt'%dataset.lower(), 'r') as fp:
        result = fp.readlines()
    result = result[1:]
    
    train_dict = {}
    with open('train_%s.txt'%dataset.lower(), 'r') as fp:
        train = fp.readlines()
    for row in train:
        if dataset in ['LastFM','Amazon-Beauty']:
            tmp = row.split("\t")
        elif dataset in ['Douban-Book','Yelp2018', 'Gowalla']:
            tmp = row.split(" ")
        if tmp[0] in train_dict:
            train_dict[tmp[0]].append(tmp[1])
        else:
            train_dict[tmp[0]] = [tmp[1]]
    del train, tmp, row
    
    #DiffRec curves
    energy_x_DiffRec=[]
    energy_y_DiffRec=[]
    train_nodes = list(train_dict.keys())
    with open('reconstruct_DiffRec_%s'%dataset.lower(), 'rb') as fp:
        reconstruct_DiffRec = pickle.load(fp)
    reconstruct_DiffRec /= reconstruct_DiffRec.sum(axis=1, keepdims=True) 
    reconstruct_DiffRec = np.maximum(reconstruct_DiffRec, 0) # avoid negative values
    for i in range(len(reconstruct_DiffRec)):
        energy_x_DiffRec.append(len(train_dict[train_nodes[i]]))
        if (reconstruct_DiffRec[i]>1).any() or (reconstruct_DiffRec[i]<-1).any() or (reconstruct_DiffRec[i].sum()>1): # abnormal points
            reconstruct_DiffRec[i] = np.minimum(reconstruct_DiffRec[i], 1)
            reconstruct_DiffRec[i] = reconstruct_DiffRec[i] / reconstruct_DiffRec[i].sum()
            # energy_y_DiffRec.append(np.minimum((reconstruct_DiffRec[i]>=1/len(train_dict[train_nodes[i]])).sum(), len(train_dict[train_nodes[i]])))
            energy_y_DiffRec.append((reconstruct_DiffRec[i]>=1/len(train_dict[train_nodes[i]])).sum())
            continue
        energy_i = np.minimum((reconstruct_DiffRec[i]>=1/len(train_dict[train_nodes[i]])).sum(), len(train_dict[train_nodes[i]]))
        low_part = reconstruct_DiffRec[i][reconstruct_DiffRec[i]<1/len(train_dict[train_nodes[i]])]
        energy_i += (low_part * (low_part/(1/len(train_dict[train_nodes[i]])))).sum()
        energy_y_DiffRec.append(energy_i)
    
    #Tri-Diff curves
    energy_x_TV_Diff=[]
    energy_y_TV_Diff=[]
    train_nodes = list(train_dict.keys())
    with open('reconstruct_TV_Diff_%s'%dataset.lower(), 'rb') as fp:
        reconstruct_TV_Diff = pickle.load(fp)
    reconstruct_TV_Diff /= reconstruct_TV_Diff.sum(axis=1, keepdims=True) 
    reconstruct_TV_Diff = np.maximum(reconstruct_TV_Diff, 0) # avoid negative values
    for i in range(len(reconstruct_TV_Diff)):
        energy_x_TV_Diff.append(len(train_dict[train_nodes[i]]))
        if (reconstruct_TV_Diff[i]>1).any() or (reconstruct_TV_Diff[i]<-1).any() or (reconstruct_TV_Diff[i].sum()>1): # abnormal points
            reconstruct_TV_Diff[i] = np.minimum(reconstruct_TV_Diff[i], 1)
            reconstruct_TV_Diff[i] = reconstruct_TV_Diff[i] / reconstruct_TV_Diff[i].sum()
            # energy_y_TV_Diff.append(np.minimum((reconstruct_TV_Diff[i]>=1/len(train_dict[train_nodes[i]])).sum(), len(train_dict[train_nodes[i]])))
            energy_y_TV_Diff.append((reconstruct_DiffRec[i]>=1/len(train_dict[train_nodes[i]])).sum())
            continue
        energy_i = np.minimum((reconstruct_TV_Diff[i]>=1/len(train_dict[train_nodes[i]])).sum(), len(train_dict[train_nodes[i]]))
        low_part = reconstruct_TV_Diff[i][reconstruct_TV_Diff[i]<1/len(train_dict[train_nodes[i]])]
        energy_i += (low_part * (low_part/(1/len(train_dict[train_nodes[i]])))).sum()
        energy_y_TV_Diff.append(energy_i)
    
    # BPR curvas
    energy_x_BPR=[]
    energy_y_BPR=[]
    train_nodes = list(train_dict.keys())
    with open('emb_BPR_%s'%dataset.lower(), 'rb') as fp:
        [user_emb, item_emb] = pickle.load(fp)
    reconstruct_BPR = np.dot(user_emb, item_emb.T)
    reconstruct_BPR = np.exp(reconstruct_BPR) #softmax
    reconstruct_BPR = reconstruct_BPR / reconstruct_BPR.sum(axis=1, keepdims=True) 
    for i in range(len(reconstruct_BPR)):
        energy_x_BPR.append(len(train_dict[train_nodes[i]]))
        energy_i = (reconstruct_BPR[i]>=1/len(train_dict[train_nodes[i]])).sum()
        low_part = reconstruct_BPR[i][reconstruct_BPR[i]<1/len(train_dict[train_nodes[i]])]
        energy_i += (low_part * (low_part/(1/len(train_dict[train_nodes[i]])))).sum()
        energy_y_BPR.append(energy_i)    
    
    # LightGCN curvas
    energy_x_LightGCN=[]
    energy_y_LightGCN=[]
    train_nodes = list(train_dict.keys())
    with open('emb_LightGCN_%s'%dataset.lower(), 'rb') as fp:
        [user_emb, item_emb] = pickle.load(fp)
    reconstruct_LightGCN = np.dot(user_emb, item_emb.T)
    reconstruct_LightGCN = np.exp(reconstruct_LightGCN) #softmax
    reconstruct_LightGCN = reconstruct_LightGCN / reconstruct_LightGCN.sum(axis=1, keepdims=True) 
    for i in range(len(reconstruct_LightGCN)):
        energy_x_LightGCN.append(len(train_dict[train_nodes[i]]))
        energy_i = (reconstruct_LightGCN[i]>=1/len(train_dict[train_nodes[i]])).sum()
        low_part = reconstruct_LightGCN[i][reconstruct_LightGCN[i]<1/len(train_dict[train_nodes[i]])]
        energy_i += (low_part * (low_part/(1/len(train_dict[train_nodes[i]])))).sum()
        energy_y_LightGCN.append(energy_i)
    
    sns.scatterplot(ax=ax, x=energy_x_TV_Diff, y=energy_y_TV_Diff, c='forestgreen', alpha=0.6)
    sns.scatterplot(ax=ax, x=energy_x_BPR, y=energy_y_BPR, c='gold', alpha=0.6)
    sns.scatterplot(ax=ax, x=energy_x_LightGCN, y=energy_y_LightGCN, c='lightcoral', alpha=0.6)
    sns.scatterplot(ax=ax, x=energy_x_DiffRec, y=energy_y_DiffRec, c='royalblue', alpha=0.6)
    
    data_energy = pd.DataFrame({'Models':['BPR']*len(energy_x_BPR)+['LightGCN']*len(energy_x_LightGCN)+['DiffRec']*len(energy_x_DiffRec)+['TV-Diff']*len(energy_x_TV_Diff), \
                          'Original':energy_x_BPR+energy_x_LightGCN+energy_x_DiffRec+energy_x_TV_Diff, \
                              'Reconstructed':energy_y_BPR+energy_y_LightGCN+energy_y_DiffRec+energy_y_TV_Diff})
    if dataset=='LastFM':
        bw_weight=0.05
    if dataset=='Amazon-Beauty':
        bw_weight=1.6
    if dataset=='Douban-Book':
        bw_weight=1.6
    if dataset=='Yelp2018':
        bw_weight=2.5
    if dataset=='Gowalla':
        bw_weight=1.6
    
    sns.kdeplot(ax=ax, data=data_energy, x='Original', y='Reconstructed', fill=True, hue='Models', palette=['gold','lightcoral','royalblue','forestgreen'], alpha=0.3, thresh=0.001, levels=50, bw_method=bw_weight)
    ax.set_xlabel("Original", fontsize=20)
    ax.set_ylabel("Reconstructed", fontsize=20)
    ax.set_title(r'%s'%(dataset), fontsize=25)
    # ax.set_title(r'%s, $\frac{\Delta U_D}{\Delta U_L}=%.3f$, $\frac{\Delta U_U}{\Delta U_L}=%.3f$'\
    #                %(dataset, (np.sum(energy_y_DiffRec)-np.sum(energy_x_DiffRec))/(np.sum(energy_y_LightGCN)-np.sum(energy_x_LightGCN)), \
    #                  (np.sum(energy_y_TV_Diff)-np.sum(energy_x_TV_Diff))/(np.sum(energy_y_LightGCN)-np.sum(energy_x_LightGCN))), fontsize=13) 
    print(np.sum(energy_y_BPR)-np.sum(energy_x_BPR), np.sum(energy_y_DiffRec)-np.sum(energy_x_DiffRec), (np.sum(energy_y_LightGCN)-np.sum(energy_x_LightGCN)),\
          (np.sum(energy_y_TV_Diff)-np.sum(energy_x_TV_Diff)))
    sns.move_legend(ax, "upper left")
    ax.grid(linestyle='--',alpha=0.5)

    return ax, train_dict, data_energy, reconstruct_DiffRec, reconstruct_LightGCN, reconstruct_TV_Diff, reconstruct_BPR
    
# canvas initialization
fig, ax = plt.subplots(1, 5, figsize=(25, 5))
plt.subplots_adjust(top=0.887,
bottom=0.122,
left=0.026,
right=0.995,
hspace=0.18,
wspace=0.24)

datasets = ['LastFM','Amazon-Beauty','Douban-Book','Yelp2018','Gowalla']
# datasets=['LastFM']
for i, name in enumerate(datasets):
    ax[i], train_dict, data_energy, reconstruct_DiffRec, reconstruct_LightGCN, reconstruct_TV_Diff, reconstruct_BPR = draw_entropy(name, i, ax[i])
    print('Finished %s.'%name)
plt.show()