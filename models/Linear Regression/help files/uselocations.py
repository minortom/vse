import numpy as np
import pandas as pd
import s2sphere
import math

def isNaN(num):
	return num != num
origData = pd.read_csv('../../data/locationdata.csv', na_values=['#NAME?'])

for index, row in origData.iterrows():
	lat = row['lat']
	lon = row['lon']
	if not isNaN(lat):
		newvalue = (lat **  2) + (lon ** 2)
		newvalue = math.sqrt(newvalue)
		print newvalue
		origData.set_value(index,'CombinedLocation', newvalue)


		
origData.to_csv("testing.csv", sep=',', encoding='utf-8')

