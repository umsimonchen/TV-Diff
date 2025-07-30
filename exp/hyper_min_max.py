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

noise_max = ['5e-3', '1e-2']
noise_min = ['5e-4', '1e-3', '5e-3']
x = list(range(len(noise_max)))
y = list(range(len(noise_min)))
xx = np.array([0,1,0,1,1])
yy = np.array([0,0,1,1,2])

r_fm = [0.31327,
0.31217,
0.31283,
0.31201,
# 0.31217, #0
0.31321,
]
n_fm = [0.29533,
0.29468,
0.295,
0.29459,
# 0.29459, #0
0.29532,
]

r_ab = [0.14792,
0.13653,
0.14745,
0.13825,
# 0.13653, #0
0.14789,
]
n_ab = [0.08503,
0.07688,
0.08516,
0.07808,
# 0.07688, #0
0.08505,
]

r_db = [0.20191,
0.20333,
0.20053,
0.20366,
# 0.20053, #0
0.20123,
]
n_db = [0.18902,
0.19056,
0.18975,
0.19078,
# 0.18902, #0
0.18947,
]

r_ye = [0.06997,
0.07001,
0.06979,
0.07011,
# 0.06979, #0
0.07003,
]
n_ye = [0.0582,
0.05806,
0.0581,
0.05812,
# 0.05806, #0
0.05826,
]

r_go = [0.21841,
0.21919,
0.2176,
0.21952,
# 0.2176, #0
0.21875,
]
n_go = [0.17565,
0.17596,
0.17523,
0.17615,
# 0.17523, #0
0.17592,
]

fig = plt.figure(figsize=(18, 9))
plt.subplots_adjust(top=0.94,
bottom=0.02,
left=0.0,
right=1.0,
hspace=0.05,
wspace=0.0)

gs = gridspec.GridSpec(2, 6)
ax1 = plt.subplot(gs[0, 1:3], projection='3d')
ax2 = plt.subplot(gs[0, 3:5], projection='3d')
ax3 = plt.subplot(gs[1, :2], projection='3d')
ax4 = plt.subplot(gs[1, 2:4], projection='3d')
ax5 = plt.subplot(gs[1, 4:], projection='3d')

ax1.bar3d(xx-0.35, yy-0.35, r_fm, 0.46, 0.7, -np.array(r_fm)+np.array(r_fm).min(), color='violet', alpha=0.6)
ax2.bar3d(xx-0.35, yy-0.35, r_ab, 0.46, 0.7, -np.array(r_ab)+np.array(r_ab).min(), color='violet', alpha=0.6)
ax3.bar3d(xx-0.35, yy-0.35, r_db, 0.46, 0.7, -np.array(r_db)+np.array(r_db).min(), color='violet', alpha=0.6)
ax4.bar3d(xx-0.35, yy-0.35, r_ye, 0.46, 0.7, -np.array(r_ye)+np.array(r_ye).min(), color='violet', alpha=0.6)
ax5.bar3d(xx-0.35, yy-0.35, r_go, 0.46, 0.7, -np.array(r_go)+np.array(r_go).min(), color='violet', alpha=0.6)

ax1.set_title('LastFM', fontsize=40)
ax1.set_xlabel('Noise Max.', fontsize=20)
ax1.set_ylabel('Noise Min.', fontsize=20)
ax1.set_zlabel('Recall@20', fontsize=20)
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.set_xticks(x, noise_max)
ax1.set_xticklabels(noise_max)
ax1.set_yticks(y, noise_min)
ax1.set_yticklabels(noise_min)
ax1.grid(linestyle='--',alpha=0.5)
ax1.set_zlim((np.array(r_fm).min(),np.array(r_fm).max()))
ax1.view_init(elev=35, azim=125)

ax2.set_title('Amazon-Beauty', fontsize=40)
ax2.set_xlabel('Noise Max.', fontsize=20)
ax2.set_ylabel('Noise Min.', fontsize=20)
ax2.set_zlabel('Recall@20', fontsize=20)
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.set_xticks(x, noise_max)
ax2.set_xticklabels(noise_max)
ax2.set_yticks(y, noise_min)
ax2.set_yticklabels(noise_min)
ax2.grid(linestyle='--',alpha=0.5)
ax2.set_zlim((np.array(r_ab).min(),np.array(r_ab).max()))
ax2.view_init(elev=35, azim=125)

ax3.set_title('Douban-Book', fontsize=40)
ax3.set_xlabel('Noise Max.', fontsize=20)
ax3.set_ylabel('Noise Min.', fontsize=20)
ax3.set_zlabel('Recall@20', fontsize=20)
ax3.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.set_xticks(x, noise_max)
ax3.set_xticklabels(noise_max)
ax3.set_yticks(y, noise_min)
ax3.set_yticklabels(noise_min)
ax3.grid(linestyle='--',alpha=0.5)
ax3.set_zlim((np.array(r_db).min(),np.array(r_db).max()))
ax3.view_init(elev=35, azim=125)

ax4.set_title('Yelp2018', fontsize=40)
ax4.set_xlabel('Noise Max.', fontsize=20)
ax4.set_ylabel('Noise Min.', fontsize=20)
ax4.set_zlabel('Recall@20', fontsize=20)
ax4.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax4.set_xticks(x, noise_max)
ax4.set_xticklabels(noise_max)
ax4.set_yticks(y, noise_min)
ax4.set_yticklabels(noise_min)
ax4.grid(linestyle='--',alpha=0.5)
ax4.set_zlim((np.array(r_ye).min(),np.array(r_ye).max()))
ax4.view_init(elev=35, azim=125)

ax5.set_title('Gowalla', fontsize=40)
ax5.set_xlabel('Noise Max.', fontsize=20)
ax5.set_ylabel('Noise Min.', fontsize=20)
ax5.set_zlabel('Recall@20', fontsize=20)
ax5.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax5.set_xticks(x, noise_max)
ax5.set_xticklabels(noise_max)
ax5.set_yticks(y, noise_min)
ax5.set_yticklabels(noise_min)
ax5.grid(linestyle='--',alpha=0.5)
ax5.set_zlim((np.array(r_go).min(),np.array(r_go).max()))
ax5.view_init(elev=35, azim=125)

# handles_h, labels_h = ax1.get_legend_handles_labels()
# handles_d, labels_d = ax1_1.get_legend_handles_labels()
# fig.legend(handles_h+handles_d, labels_h+labels_d, loc='upper center', ncol=4, fontsize=20)

plt.show()


