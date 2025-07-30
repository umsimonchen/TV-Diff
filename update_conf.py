# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:18:44 2024

@author: user
"""

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str)
parser.add_argument("-K", type=int)
args = parser.parse_args()

with open('conf/MF.conf', 'r') as fp:
    lines = fp.readlines()

lines[0] = "training.set=./dataset/%s/train.txt"%(args.dataset)+"\n"
lines[1] = "test.set=./dataset/%s/test.txt"%(args.dataset)+"\n"
lines[4] = "item.ranking=-topN %d"%(args.K)+"\n"
    
# lines[10] = "CDAE=-corruption_ratio %f"%(args.corruption)+"\n"

with open('conf/MF.conf', 'w') as fp:
    fp.writelines(lines)