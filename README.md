# heart-disease-model
A project to suplement an ML API with a model. The API can be found via: https://github.com/Eligijus112/flask-application 

# Data source 

More about the heart disease data can be found via: https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset 

# Creating the virtual env

```
virtualenv heart
```

The bellow command activates the env as well

```
source heart/bin/activate
```

```
pip install -r requirements.txt
```

# Dependant variable 

TenYearCHD - whether or not a person developed Coronary Heart Disease (CHD).

# Feature list for the independant variables 

## Demographic Risks

gender

age

education (1: high school, 2: diploma, 3: college, 4: higher than degree)

## Behavioral Risks

CurrentSmoker - Current smoker or not

CigsPerDay - Average number of cigarettes smoked per day

## Medical experiments

BPMeds - Patient is under blood pressure medication

PrevalentStroke - Previously had a stroke or not

PrevalentHyp - Prevalent Hypertension or not

Diabetes - Patient has diabetes or not

TotChol - Total Cholesterol

Glucose - Glucose level

## Physical examination

DiaBP - Diastolic blood pressure

BMI - Body mass index

Heart - Rate Heart Rate

SysBP - Systolic blood pressure

# Amount of data 

The data has 4238 observations.

# Running the pipeline 

## Configuration file 

The configurations should be in **conf.yml** file. 

## Cleaning the data 

The data is cleaned using the command: 

```
python dataPreproc.py 
```

The cleaned data is stored in the directory clean-data/ .

## Creating the model 

To create the model run:

```
python LogisticModel.py 
```