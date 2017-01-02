from streetaddress import StreetAddressFormatter, StreetAddressParser
from geopy.geocoders import Nominatim
addr_parser = StreetAddressParser()
geolocator = Nominatim()
import numpy as np
import pandas as pd



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
			r = ID['Prop_Addr'].item() + " Ames"
			origData.set_value(index,'Address',ID['Prop_Addr'].item() + " Ames")
			try:
				location = geolocator.geocode(ID['Prop_Addr'].item() + " Ames")
				address = addr_parser.parse(r)
				d = address['street_full']
				origData.set_value(index,'StreetName', str(d))
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
	

		
origData.to_csv("testing3.csv", sep=',', encoding='utf-8')