# Importing the model class 
from sklearn.linear_model import LogisticRegression

# Data wrangling 
import pandas as pd 

# Saving the model 
import pickle

# Configuration reading 
import yaml 

# Directory managment
import os 

# Regular expressions
import re

# Reading the configurations
conf = yaml.load(open("conf.yml"), Loader=yaml.FullLoader)

# Reading data 
d = pd.read_csv('framingham_heart_disease.csv')

# Defining the feature list 
numeric_features = conf.get("numeric_features")
categorical_features = conf.get('categorical_features')

# Filling the missing values with the term "missing"
d[categorical_features] = d[categorical_features].fillna('missing')

# Converting to type categorical
d[categorical_features] = d[categorical_features].astype(str)

# Subbing the .0 in the categorical values 
d[categorical_features] = d[categorical_features].apply(lambda x: [re.sub(".0", "", value) for value in x])

# Leaving only the columns from the list 
d = d[numeric_features + categorical_features + ['TenYearCHD']]

# Creating the dummy frame 
d = pd.get_dummies(d, drop_first=True)

# Filling the numeric missing values with means
d[numeric_features] = d[numeric_features].fillna(d[numeric_features].mean())

# Creating the folder clean data 
if not os.path.exists('clean-data'):
    os.mkdir('clean-data')

# Savign the cleaned data to that folder 
d.to_pickle("clean-data/clean-data.pkl")