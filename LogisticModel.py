# Importing the model class 
from sklearn.linear_model import LogisticRegression

# Data wrangling 
import pandas as pd 

# Saving the model 
import pickle

# Reading data 
d = pd.read_csv('framingham_heart_disease.csv')

# Defining the feature list 
features = [
    'heartRate',
    'glucose',
    'BMI',
    'totChol',
    'cigsPerDay'
]

# Creating the X and Y matrices 
X = d[features].copy()
Y = d['TenYearCHD']

# Filling possible missing values 
X.fillna(X.mean(), inplace=True)

# Initiating the model 
model = LogisticRegression()

# Fitting the model 
model.fit(X, Y)

# Saving the feature names to model object
model.feature_names = X.columns.tolist()

# Saving the model object to a pickle file 
pickle.dump(model, open('models/model_v1.sav', 'wb'))