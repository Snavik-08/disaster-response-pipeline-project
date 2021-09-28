# Disaster Response Pipeline Project


### Project Motivation
This is a Project that is required as a task for the Udacity Data Science Nanodegree Program. It is an app that can be used in emergency situation where individuals post emergency messages in the app that are then classified based on their wording. The results of the emergency message classification is then displayed within a web app based on flask. 


### File Descriptions
app

| - template 

| |- master.html  

| |- go.html  

|- run.py  


data

|- disaster_categories.csv 

|- disaster_messages.csv

|- process_data.py 


models

|- train_classifier.py  

README.md


### Components
There are three components for this Project

#### 1. ETL Pipeline
A Python script, process_data.py, writes a data cleaning pipeline that:

- Loading of the messages and its catagories DBs. This is merged and used as train DF. 
- Cleaning and formatting for later training
- Saves cleaned DF in SQLite DB


#### 2. ML Pipeline
A Python script, train_classifier.py, writes a machine learning pipeline that:

- Loading Data from a SQLite DB
- Splitting in Train and Test Data
- Creates a Pipeline where data is processed and used for training
- Training is done via Grid Search Optimization that is also included in the pipeline
- Evaluates Model performance on a test set
- Exports train model as pickle file
- A jupyter notebook ML Pipeline Preparation was used to do EDA to prepare the train_classifier.py python script.


#### 3. Flask Web App
The UI of this project is a flask web app where messages can be posted that then are calssified based on there emergency type.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
