# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 01:06:51 2017

@author: HERVE
"""


from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('DENGUE_OK.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 4, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()