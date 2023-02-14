import pandas as pd
import numpy as np

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

def apply_nanashi_data_processing(train, test):
    
    # Deleting the ID
    train.drop(['Id'], axis=1, inplace=True)
    test.drop(['Id'], axis=1, inplace=True)
    
    # Filtering outliers in GrLivArea feature
    train = train[train.GrLivArea < 4500].copy()
    train.reset_index(drop=True, inplace=True)
    
    # Apply logarithmic function to the target
    train["SalePrice"] = np.log1p(train["SalePrice"])
    y = train['SalePrice'].reset_index(drop=True)
    
    # Dropping the target from the training data
    train_features = train.drop(['SalePrice'], axis=1)
    test_features = test
    
    # Concatenating the train data and submission data (test) in the same dataframe in order to do the same processing to them
    features = pd.concat([train_features, test_features]).reset_index(drop=True)
    
    # Correct some types
    features['MSSubClass'] = features['MSSubClass'].apply(str)
    features['YrSold'] = features['YrSold'].astype(str)
    features['MoSold'] = features['MoSold'].astype(str)
    
    # Fill null values with specific values
    features['Functional'] = features['Functional'].fillna('Typ') 
    features['Electrical'] = features['Electrical'].fillna("SBrkr") 
    features['KitchenQual'] = features['KitchenQual'].fillna("TA") 
    features["PoolQC"] = features["PoolQC"].fillna("None")
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
    
    # Fill null values with the mode
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    # Fill null values with zeros
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    
    # Fill every nulls of object columns with None 
    objects = []
    for column in features.columns:
        if features[column].dtype == object:
            objects.append(column)
    features.update(features[objects].fillna('None'))
    
    # Fill null values with the median
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
    # Fill every nulls of numeric columns with zeros
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numerics.append(i)
    features.update(features[numerics].fillna(0))
    
    # Transformation of numeric features for having a better distribution
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics2 = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numerics2.append(i)
    skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
    
    # Feature engineering
    features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

    features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
    features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

    features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                     features['1stFlrSF'] + features['2ndFlrSF'])

    features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                                   features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

    features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                                  features['EnclosedPorch'] + features['ScreenPorch'] +
                                  features['WoodDeckSF'])
    
    features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
    # Dividing the main dataframe in X (for training) and X_submission (for data to predict and submit in Kaggle)
    final_features = pd.get_dummies(features).reset_index(drop=True)
    final_features.shape
    X = final_features.iloc[:len(y), :]
    X_submission = final_features.iloc[len(y):, :]

    return X, y, X_submission