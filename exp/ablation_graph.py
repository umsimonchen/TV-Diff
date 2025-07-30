# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:52:22 2023

@author: simon
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:31:47 2023

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

labels = ['Recall@20', 'NDCG@20']
none_fm = [0.30659,	0.29533]
sys_fm = [0.31327,	0.29609]
left_fm = [0.30182,	0.28817]
lp_fm = [0.30843, 0.29557]

none_ab = [0.13085,	0.07681]
sys_ab = [0.14745,	0.08516]
left_ab = [0.13444,	0.07742]
lp_ab = [0.13843, 0.07968]

none_db = [0.19079,	0.17381]
sys_db = [0.20366,	0.19078]
left_db = [0.17995,	0.16608]
lp_db = [0.19403, 0.18342]

none_ye = [0.06541,	0.05435]
sys_ye = [0.06997,	0.05819]
left_ye = [0.06659,	0.05496]
lp_ye = [0.06743, 0.05624]

none_go = [0.20307,	0.16148]
sys_go = [0.21952,	0.17615]
left_go = [0.20774,	0.16592]
lp_go = [0.21140, 0.16998]

x = np.arange(len(labels))  # the label locations
fig = plt.figure(figsize=(18, 9))
plt.subplots_adjust(top=0.8,
bottom=0.05,
left=0.065,
right=0.935,
hspace=0.355,
wspace=0.48)

gs = gridspec.GridSpec(2, 6)
gs.update(wspace=1.0)
ax1 = plt.subplot(gs[0, 1:3])
ax2 = plt.subplot(gs[0, 3:5])
ax3 = plt.subplot(gs[1, :2])
ax4 = plt.subplot(gs[1, 2:4])
ax5 = plt.subplot(gs[1, 4:])

ax1_1 = ax1.twinx()
ax2_1 = ax2.twinx()
ax3_1 = ax3.twinx()
ax4_1 = ax4.twinx()
ax5_1 = ax5.twinx()

width = 0.3  # the width of the bars

rects11 = ax1.bar(0 - width, none_fm[0], width/3*2, label='None', color=plt.cm.Set3(6))
rects12 = ax1.bar(0 - width/3, sys_fm[0], width/3*2, label='Sym.', color='gold')
rects13 = ax1.bar(0 + width/3, left_fm[0], width/3*2, label='Left', color=plt.cm.Set3(3))
rects14 = ax1.bar(0 + width, lp_fm[0], width/3*2, label='LP', color=plt.cm.Set3(4))
rects11_ = ax1_1.bar(1 - width, none_fm[1], width/3*2, label='None', color=plt.cm.Set3(6))
rects12_ = ax1_1.bar(1 - width/3, sys_fm[1], width/3*2, label='Sym.', color='gold')
rects13_ = ax1_1.bar(1 + width/3, left_fm[1], width/3*2, label='Left', color=plt.cm.Set3(3))
rects14_ = ax1_1.bar(1 + width, lp_fm[1], width/3*2, label='LP', color=plt.cm.Set3(4))

rects21 = ax2.bar(0 - width, none_ab[0], width/3*2, label='None', color=plt.cm.Set3(6))
rects22 = ax2.bar(0 - width/3, sys_ab[0], width/3*2, label='Sym.', color='gold')
rects23 = ax2.bar(0 + width/3, left_ab[0], width/3*2, label='Left', color=plt.cm.Set3(3))
rects24 = ax2.bar(0 + width, lp_ab[0], width/3*2, label='LP', color=plt.cm.Set3(4))
rects21_ = ax2_1.bar(1 - width, none_ab[1], width/3*2, label='None', color=plt.cm.Set3(6))
rects22_ = ax2_1.bar(1 - width/3, sys_ab[1], width/3*2, label='Sym.', color='gold')
rects23_ = ax2_1.bar(1 + width/3, left_ab[1], width/3*2, label='Left', color=plt.cm.Set3(3))
rects24_ = ax2_1.bar(1 + width, lp_ab[1], width/3*2, label='LP', color=plt.cm.Set3(4))

rects31 = ax3.bar(0 - width, none_db[0], width/3*2, label='None', color=plt.cm.Set3(6))
rects32 = ax3.bar(0 - width/3, sys_db[0], width/3*2, label='Sym.', color='gold')
rects33 = ax3.bar(0 + width/3, left_db[0], width/3*2, label='Left', color=plt.cm.Set3(3))
rects34 = ax3.bar(0 + width, lp_db[0], width/3*2, label='LP', color=plt.cm.Set3(4))
rects31_ = ax3_1.bar(1 - width, none_db[1], width/3*2, label='None', color=plt.cm.Set3(6))
rects32_ = ax3_1.bar(1 - width/3, sys_db[1], width/3*2, label='Sym.', color='gold')
rects33_ = ax3_1.bar(1 + width/3, left_db[1], width/3*2, label='Left', color=plt.cm.Set3(3))
rects34_ = ax3_1.bar(1 + width, lp_db[1], width/3*2, label='LP', color=plt.cm.Set3(4))

rects41 = ax4.bar(0 - width, none_ye[0], width/3*2, label='None', color=plt.cm.Set3(6))
rects42 = ax4.bar(0 - width/3, sys_ye[0], width/3*2, label='Sym.', color='gold')
rects43 = ax4.bar(0 + width/3, left_ye[0], width/3*2, label='Left', color=plt.cm.Set3(3))
rects44 = ax4.bar(0 + width, lp_ye[0], width/3*2, label='LP', color=plt.cm.Set3(4))
rects41_ = ax4_1.bar(1 - width, none_ye[1], width/3*2, label='Vanilla', color=plt.cm.Set3(6))
rects42_ = ax4_1.bar(1 - width/3, sys_ye[1], width/3*2, label='Sym.', color='gold')
rects43_ = ax4_1.bar(1 + width/3, left_ye[1], width/3*2, label='Left', color=plt.cm.Set3(3))
rects44_ = ax4_1.bar(1 + width, lp_ye[1], width/3*2, label='LP', color=plt.cm.Set3(4))

rects51 = ax5.bar(0 - width, none_go[0], width/3*2, label='None', color=plt.cm.Set3(6))
rects52 = ax5.bar(0 - width/3, sys_go[0], width/3*2, label='Sym.', color='gold')
rects53 = ax5.bar(0 + width/3, left_go[0], width/3*2, label='Left', color=plt.cm.Set3(3))
rects54 = ax5.bar(0 + width, lp_go[0], width/3*2, label='LP', color=plt.cm.Set3(4))
rects51_ = ax5_1.bar(1 - width, none_go[1], width/3*2, label='None', color=plt.cm.Set3(6))
rects52_ = ax5_1.bar(1 - width/3, sys_go[1], width/3*2, label='Sym.', color='gold')
rects53_ = ax5_1.bar(1 + width/3, left_go[1], width/3*2, label='Left', color=plt.cm.Set3(3))
rects54_ = ax5_1.bar(1 + width, lp_go[1], width/3*2, label='LP', color=plt.cm.Set3(4))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Recall@20', fontsize=45)
ax3.set_ylabel('Recall@20', fontsize=45)
ax2_1.set_ylabel('NDCG@20', fontsize=45)
ax5_1.set_ylabel('NDCG@20', fontsize=45)
ax1.set_title('LastFM', fontsize=40)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=30)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax1_1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax1.grid(linestyle='--',alpha=0.5)
ax1.set_ylim(0.295,0.315)
ax1_1.set_ylim(0.28,0.3)

ax2.set_title('Amazon-Book', fontsize=40)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=30)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2_1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.grid(linestyle='--',alpha=0.5)
ax2.set_ylim(0.115,0.15)
ax2_1.set_ylim(0.07,0.09)

ax3.set_title('Douban-Book', fontsize=40)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=30)
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax3_1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax3.grid(linestyle='--',alpha=0.5)
ax3.set_ylim(0.16,0.215)
ax3_1.set_ylim(0.15,0.195)

ax4.set_title('Yelp2018', fontsize=40)
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=30)
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax4_1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax4.grid(linestyle='--',alpha=0.5)
ax4.set_ylim(0.055,0.075)
ax4_1.set_ylim(0.045,0.06)

ax5.set_title('Gowalla', fontsize=40)
ax5.set_xticks(x)
ax5.set_xticklabels(labels, fontsize=30)
ax5.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax5_1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax5.grid(linestyle='--',alpha=0.5)
ax5.set_ylim(0.195,0.225)
ax5_1.set_ylim(0.15,0.18)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=40)

plt.subplot_tool()
plt.show()