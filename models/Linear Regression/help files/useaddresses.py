
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
geolocator = Nominatim()

origData = pd.read_csv('../../data/test.csv', na_values=['#NAME?'])
allData = pd.read_csv('../../data/28-12-2016.csv', na_values=['#NAME?'])

for index, row in origData.iterrows():
	MSSubClass = row['MSSubClass']
	LotFrontage = row['LotFrontage']
	YearBuilt = row['YearBuilt']
	Neighborhood = row['Neighborhood']
	BsmtFinType1 = row['BsmtFinType1']
	GrLivArea = row['GrLivArea']
	GarageArea = row['GarageArea']
	GarageCars = row['GarageCars']
	MoSold = row['MoSold']
	ExterCond = row['ExterCond']
	ID = allData[(allData['MSSubClass'] == MSSubClass) & (allData['LotFrontage'] == LotFrontage) & (YearBuilt == allData['YearBuilt']) & (BsmtFinType1 == allData['BsmtFinType1']) & (Neighborhood == allData['Neighborhood']) & (GrLivArea == allData['GrLivArea']) & (GarageArea == allData['GarageArea']) & (GarageCars == allData['GarageCars']) & (MoSold == allData['MoSold']) & (ExterCond == allData['ExterCond'])]
	if len(ID.index) == 1:
		print "pass"
		try:
			location = geolocator.geocode(ID['StreetName'].item() + " Ames")
			print location.latitude
		except:
			pass
		try:
			location
			print "pass loc2"
			try:
				origData.set_value(index,'Lat',location.latitude)
			except:
				pass
			try:
				origData.set_value(index,'Lon',location.longitude)
			except:
				pass
		except:
			pass
		origData.set_value(index,'StreetName',str(ID['StreetName'].item()) + " Ames")
	
origData.to_csv("testremake.csv", sep=',', encoding='utf-8')