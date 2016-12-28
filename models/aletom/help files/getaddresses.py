import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from address import AddressParser, Address

ap = AddressParser()
geolocator = Nominatim()

origData = pd.read_csv('../../data/originaldata.csv', na_values=['#NAME?'])
allData = pd.read_csv('../../data/newdatawithaddresses.csv', na_values=['#NAME?'])

for index, row in origData.iterrows():
	pid = row['PID']
	ID = allData[allData['MapRefNo'] == pid]
	if len(ID.index) > 1:
		origData.set_value(index,'Address',ID['Prop_Addr'].iloc[0] + " Ames")
	else:
		if ID.empty:
			print "empty"
		else:
			origData.set_value(index,'Address',ID['Prop_Addr'].item() + " Ames")
			try:
			  location = geolocator.geocode(ID['Prop_Addr'].item() + " Ames")
			except:
				pass
	try:
		if location:
			try:
				origData.set_value(index,'Lat',location.latitude)
			except:
				pass
			try:
				origData.set_value(index,'Lng',location.longitude)
			except:
				pass
			addr = ap.parse_address(location.address)
			try:
				origData.set_value(index,'Address', location.address)
			except:
				pass
			try:
				origData.set_value(index,'PostalCode', addr.zip)
			except:
				pass
			try:
				origData.set_value(index,'HouseNumber', addr.house_number)
			except:
				pass
	except NameError:
		print "well, it WASN'T defined after all!"
	else:
		print "sure, it was defined."

	
		
origData.to_csv("testing.csv", sep=',', encoding='utf-8')



