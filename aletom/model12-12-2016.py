# Import data and take a look
import numpy as np
import pandas as pd

train = pd.read_csv('../data/train.csv', na_values=['#NAME?'])
test = pd.read_csv('../data/test.csv', na_values=['#NAME?'])
yy = test
y = test.SalePrice
n = train.SalePrice
train.drop('SalePrice', 1)

def categoriesToUse(data):
# Decide which categorical variables you want to use in model
	name_array = ""
	for col_name in data.columns:
	    if data[col_name].dtypes == 'object':
	        unique_cat = len(data[col_name].unique())
	        name_array = name_array + ", '" + col_name + "'"
	return name_array

# Create a list of features to dummy
todummy_list = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# Function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

train = dummy_df(train, todummy_list)
test = dummy_df(test, todummy_list)

# Impute missing values using Imputer in sklearn.preprocessing
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(train)
train = pd.DataFrame(data=imp.transform(train) , columns=train.columns)

imps = Imputer(missing_values='NaN', strategy='median', axis=0)
imps.fit(test)
test = pd.DataFrame(data=imps.transform(test) , columns=test.columns)

# Use PolynomialFeatures in sklearn.preprocessing to create two-way interactions for all features
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]
    
    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    
    # Remove interaction terms with all 0 values            
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)
    
    return df

    
train = add_interactions(train)
test = add_interactions(test)




    # Use train_test_split in sklearn.cross_validation to split data into train and test sets
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, n, train_size=1, random_state=1)
# Such a large set of features can cause overfitting and also slow computing
# Use feature selection to select the most important features
import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [train.columns[i] for i in indices_selected]

colnames_selected.remove('SaleType_CWD_SaleCondition_Abnorml')
colnames_selected.remove('SaleType_CWD_SaleCondition_Family')

X_train_selected = train[colnames_selected]
X_test_selected = test[colnames_selected]


# Function to build model and find model performance
from sklearn.linear_model import LinearRegression
print(X_train_selected.shape)
print(X_test_selected.shape)



clf = LinearRegression(fit_intercept=True)
clf.fit(X_train, y_train)
# predictionData =  X_test_selected
# print(clf.predict(predictionData))
# Split unprocessed data into train and test set
# Build model and assess performance
X_train_unprocessed, X_test_unprocessed, y_train, y_test = train_test_split(
    X_train_selected, y, train_size=1, random_state=1)

predictionData = X_test_selected
import csv
with open('output.csv', 'w') as outcsv: 
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['Id', 'SalePrice'])
    dataPrediction = clf.predict(predictionData)
    print dataPrediction
    for i in xrange(0, len(predictionData)):
        dataId = test['Id'][i]
        if dataPrediction[i] < 0:
            writer.writerow([dataId, 0])
        else:
            writer.writerow([dataId, dataPrediction[i]])

