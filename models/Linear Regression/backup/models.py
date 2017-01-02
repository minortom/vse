# Load libraries
## General libraries
from itertools import ifilter
import numpy as np
import inspect
import csv
from astropy.table import Table, Column
import random

## For machine learning
import graphlab
import matplotlib.pyplot as plt

# Load all data sources
trainData = graphlab.SFrame('../data/trainremake.csv')
testData = graphlab.SFrame('../data/testremake.csv')

# trainData.print_rows(num_rows=10, num_columns=81)

class loadModels():
    def linearModel(self, whatIndicator, data):
        #Name of your model
        """Linear model"""
       	features = ['OverallQual', 'GrLivArea', 'GarageCars','GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath','TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
        model = graphlab.linear_regression.create(trainData, target='SalePrice', features=features,validation_set=None,verbose=False, feature_rescaling=True)
        
        if whatIndicator == 'max_error':
        	return model.evaluate(testData)['max_error']
        elif whatIndicator == 'rmse':
        	return model.evaluate(testData)['rmse']
        elif whatIndicator == 'prediction':
        	return model.predict(data)


# Function to evaluate specific models
def evaluateModels(models):
    f = models()
    attrs = (getattr(f, name) for name in dir(f))
    methods = ifilter(inspect.ismethod, attrs)
    tableData = [0] * 100
    i = 0
    for method in methods:
        try:
            t = Table([[method.__doc__], [method('max_error','')], [method('rmse', '')]], names=('Model', 'Max Error', 'RMSE'))
            print t
            i = i + 1
        except TypeError:
            # Can't handle methods with required arguments.
            print "error"
            pass


# Function to create a CSV for testing at Kaggle competition
def generateCSVForKaggle(model, data, tag):
    with open('output.csv', 'w') as outcsv: 
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['Id', 'SalePrice'])
        dataPrediction = model('prediction', data)
        for i in xrange(0, len(data)):
            dataId = data[i]['Id']
            if dataPrediction[i] < 0:
            	writer.writerow([dataId, 0])
            else:
            	writer.writerow([dataId, dataPrediction[i]])
        
    
# Call evaluation function on all models

evaluateModels(loadModels)
generateCSVForKaggle(loadModels().linearModel, testData, 'SalePrice')