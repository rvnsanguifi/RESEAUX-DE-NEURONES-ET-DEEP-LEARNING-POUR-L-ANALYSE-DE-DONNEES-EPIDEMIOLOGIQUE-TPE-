# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 00:35:09 2017

@author: HERVE
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 00:17:02 2017

@author: HERVE
"""

from pandas import read_csv
from datetime import datetime

# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d')
dataset = read_csv('dengue_okkkk.csv',  parse_dates = [['year', 'month', 'day']], index_col=0, date_parser=parse, sep=";")
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['dengue', 'temp_moy', 'temp_max', 'temp_min', 'hum_rela', 'hum_abs', 'precip', 'heur_sol']
dataset.index.name = 'date'
# mark all NA values with 0
#dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
#dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(10))
# save to file
dataset.to_csv('DENGUE_OK.csv')