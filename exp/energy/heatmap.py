# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:43:42 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

delta_S = {'LastFM':[5140.311601, 9.984452812, 12655.30379, 11994.22176],
           'Amazon-Beauty':[211904.141, 94313.53166, 209897.641, 199614.5004],
           'Douban-Book':[80834.04207, 69731.58894, 109290.6202, 107343.0889],
           'Yelp2018':[154084.5901, 144123.6526, 262136.3401, 243493.7776],
           'Gowalla':[96831.59087, 11392.76274, 271594.1065, 263456.9815]}

delta_U = {'LastFM':[-63501.39908, -58493.3925, -19336.00004, -19899.00004],
           'Amazon-Beauty':[-156721.3639, -138100.7428, -54788.0001, -58681.00026],
           'Douban-Book':[-443534.926, -425646.9068, -171176.0002, -171176.0002],
           'Yelp2018':[-1155675.143, -1136948.963, -393558.0007, -394245.0007],
           'Gowalla':[-731810.2189, -753389.0694, -283230.0007, -285014.0007]}


def get_lower_tri_heatmap(ax, df, dataset, t, i):
    cmap = sns.light_palette("seagreen", as_cmap=True)
    # sns.set(font_scale=1)
    sns.heatmap(df, cmap=cmap, center=1, vmax=2, fmt='.3f', cbar=False,\
            square=True, linewidths=.5, annot=True, ax=ax)
    ax.set_title(r'%s'%(dataset), fontsize=35)
    ax.set_xticklabels(['$\Delta %s_B$'%t, '$\Delta %s_L$'%t, '$\Delta %s_D$'%t, '$\Delta %s_T$'%t], fontsize=30)
    ax.set_xlabel('x', fontsize=30)
    if i==0:
        ax.set_ylabel('y', fontsize=30)
        ax.set_yticklabels(['$\Delta %s_B$'%t, '$\Delta %s_L$'%t, '$\Delta %s_D$'%t, '$\Delta %s_T$'%t], fontsize=30)
    else:
        ax.set_yticklabels([], fontsize=30)
    return ax

fig, ax = plt.subplots(1, 5, figsize=(20, 5))
plt.subplots_adjust(top=1.0,
bottom=0.08,
left=0.045,
right=0.996,
hspace=0.165,
wspace=0.064)
datasets = ['LastFM','Amazon-Beauty','Douban-Book','Yelp2018','Gowalla']
# datasets=['LastFM']
# for i, name in enumerate(datasets):
#     matrix_S = np.zeros((4,4))
#     for j in range(4):
#         for k in range(4):
#             matrix_S[k][j]=delta_S[name][k]/delta_S[name][j]
#     ax[i] = get_lower_tri_heatmap(ax[i], matrix_S, name, 'S',i)
# plt.show()

for i, name in enumerate(datasets):
    matrix_U = np.zeros((4,4))
    for j in range(4):
        for k in range(4):
            matrix_U[k][j]=delta_U[name][k]/delta_U[name][j]
    ax[i] = get_lower_tri_heatmap(ax[i], matrix_U, name, 'U', i)
plt.show()













