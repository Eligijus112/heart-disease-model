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

# System codes
import sys 

# Reading the configurations
conf = yaml.load(open("conf.yml"), Loader=yaml.FullLoader)

# Reading data 
d = pd.read_csv('framingham_heart_disease.csv')

# Defining the feature list 
numeric_features = conf.get("numeric_features", [])
categorical_features = conf.get('categorical_features', [])

# The final feature list 
final_features = numeric_features + categorical_features

if len(final_features) == 0:
    print('No numeric or categorical features; Please add them to the configuration file.')
    sys.exit(1)

if len(categorical_features) > 0:

    # Filling the missing values with the term "missing"
    d[categorical_features] = d[categorical_features].fillna('missing')

    # Converting to type categorical
    d[categorical_features] = d[categorical_features].astype(str)

    # Subbing the .0 in the categorical values 
    d[categorical_features] = d[categorical_features].apply(lambda x: [re.sub(".0", "", value) for value in x])

    # Converting to str type 
    d[categorical_features] = d[categorical_features].astype(str)

if len(numeric_features) > 0:
    
    # Converting to appropriate type
    d[numeric_features] = d[numeric_features].astype(float)

    # Filling the  missing values with mean
    d[numeric_features] = d[numeric_features].fillna(d[numeric_features].mean())

# Leaving only the columns from the lists + the Y variable 
d = d[final_features + ['TenYearCHD']]

# Creating the dummy frame; This will not change anything if there are no dummy variables 
d = pd.get_dummies(d, drop_first=True)

# Creating the folder clean data 
if not os.path.exists('clean-data'):
    os.mkdir('clean-data')

# Savign the cleaned data to that folder 
d.to_pickle("clean-data/clean-data.pkl")

# Sending the system that everything is OK 
sys.exit(0)