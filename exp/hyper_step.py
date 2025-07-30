# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:20:25 2023

@author: simon
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:04:02 2023

@author: simon
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FormatStrFormatter

#plt.rcParams['font.sans-serif'] = ['SimHei']

font = {'family' : 'Calibri',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)


labels = ['2', '5', '10', '40', '50', '100']

r_fm = [0.30893,
0.31327,
0.30985,
0.30981,
0.31234,
0.3113,
]
n_fm = [0.29465,
0.29533,
0.29644,
0.29686,
0.29755,
0.29723,
]

r_ab = [0.14027,
0.14534,
0.14418,
0.14493,
0.14745,
0.14471,
]
n_ab = [0.08141,
0.08281,
0.08332,
0.0832,
0.08516,
0.08287,
]

r_db = [0.18933,
0.19928,
0.20112,
0.20191,
0.20366,
0.20159,
]
n_db = [0.17904,
0.18637,
0.18787,
0.1897,
0.19078,
0.19023,
]

r_ye = [0.06745,
0.06928,
0.07011,
0.06958,
0.06996,
0.07025,
]
n_ye = [0.05627,
0.05726,
0.05812,
0.05782,
0.0581,
0.05809,
]

r_go = [0.20964,
0.21732,
0.21832,
0.21952,
0.21926,
0.21922,
]
n_go = [0.16894,
0.17393,
0.1752,
0.17615,
0.17594,
0.17601,
]

x = np.arange(len(labels))  # the label locations
fig = plt.figure(figsize=(18, 9))
plt.subplots_adjust(top=0.8,
bottom=0.05,
left=0.065,
right=0.93,
hspace=0.355,
wspace=0.48)

gs = gridspec.GridSpec(2, 6)
gs.update(wspace=1.0)
ax1 = plt.subplot(gs[0, 1:3])
ax2 = plt.subplot(gs[0, 3:5])
ax3 = plt.subplot(gs[1, :2])
ax4 = plt.subplot(gs[1, 2:4])
ax5 = plt.subplot(gs[1, 4:])

line1 = ax1.plot(x, r_fm, label='Recall@20', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
line2 = ax2.plot(x, r_ab, label='Recall@20', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
line3 = ax3.plot(x, r_db, label='Recall@20', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
line4 = ax4.plot(x, r_ye, label='Recall@20', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
line5 = ax5.plot(x, r_go, label='Recall@20', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))

ax1_1 = ax1.twinx()
ax2_1 = ax2.twinx()
ax3_1 = ax3.twinx()
ax4_1 = ax4.twinx()
ax5_1 = ax5.twinx()
line6 = ax1_1.plot(x, n_fm, label='NDCG@20', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
line7 = ax2_1.plot(x, n_ab, label='NDCG@20', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
line8 = ax3_1.plot(x, n_db, label='NDCG@20', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
line9 = ax4_1.plot(x, n_ye, label='NDCG@20', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
lineA = ax5_1.plot(x, n_go, label='NDCG@20', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))

ax1.set_ylabel('Recall@20', fontsize=45)
ax3.set_ylabel('Recall@20', fontsize=45)
ax2_1.set_ylabel('NDCG@20', fontsize=45)
ax5_1.set_ylabel('NDCG@20', fontsize=45)

ax1.set_title('LastFM', fontsize=40)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax1_1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax1.set_xticks(x, labels, fontsize=30)
#ax1.set_xticklabels(labels)
ax1.grid(linestyle='--',alpha=0.5)
#ax1.set_ylim([0.15,0.3])

ax2.set_title('Amazon-Beauty', fontsize=40)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2_1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.set_xticks(x, labels, fontsize=30)
#ax2.set_xticklabels(labels)
ax2.grid(linestyle='--',alpha=0.5)
#ax2.set_ylim([0.0,0.008])

ax3.set_title('Douban-Book', fontsize=40)
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax3_1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax3.set_xticks(x, labels, fontsize=30)
#ax3.set_xticklabels(labels)
ax3.grid(linestyle='--',alpha=0.5)
#ax3.set_ylim([0.0,0.03])

ax4.set_title('Yelp2018', fontsize=40)
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax4_1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax4.set_xticks(x, labels, fontsize=30)
#ax4.set_xticklabels(labels)
ax4.grid(linestyle='--',alpha=0.5)
#ax4.set_ylim([0.0,0.03])

ax5.set_title('Gowalla', fontsize=40)
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax5_1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax5.set_xticks(x, labels, fontsize=30)
#ax5.set_xticklabels(labels)
ax5.grid(linestyle='--',alpha=0.5)
#ax5.set_ylim([0.0,0.03])

handles_h, labels_h = ax1.get_legend_handles_labels()
handles_d, labels_d = ax1_1.get_legend_handles_labels()
fig.legend(handles_h+handles_d, labels_h+labels_d, loc='upper center', ncol=4, fontsize=40)

plt.subplot_tool()
plt.show()


