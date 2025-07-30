# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:13:28 2024

@author: user
"""

import pandas as pd

data = pd.read_csv('ratings.txt', sep=' ', header=None)
data = data[[0,1,2]]

train = data.sample(frac=0.8).sort_values(by=[0,1])
remaining = data.drop(train.index)
# validation = remaining.sample(frac=1/3).sort_values(by=[0,1])
test = remaining

train.to_csv('train.txt', sep=' ', index=None, header=None)
# validation.to_csv('validation.txt', sep=' ', index=None, header=None)
test.to_csv('test.txt', sep=' ', index=None, header=None)