from tokenize import PlainToken
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

st.set_page_config(page_title="Diagnosis prediction", page_icon="📈")

st.sidebar.header("Machine learning model for prediction of Alzheimer's disease diagnosis. \n Based on the provided patient data and metrics, the machine learning model provides a prediction on Alzheimer's disease diagnosis.")

st.title("Machine Learning Model for Prediction of Alzheimer's Disease Diagnosis")



#Insert dataframe with the data used for training the model 
df = pd.read_csv('Virtual_Assistant_AD_diagnosis/Streamlit_app/data/alzheimers_data_clean.csv')
df.drop(columns=['SES', 'MMSE'], inplace=True)

# adjust the column names to more user friendly titles
df.rename(columns = {
    'M/F':'Gender',
    'EDUC':'Years of Education',
    'SES_filled': 'Socioeconomic Status', 
    'MMSE_filled': 'Mini Mental State Examination', 
    'CDR': 'Clinical Dementia Rating', 
    'eTIV': 'Estimated total intracranial volume', 
    'nWBV': 'Normalize Whole Brain Volume', 
    'ASF': 'Atlas Scaling Factor'}, inplace = True)

st.header("Input of patient data:")

st.write("Here you can provide the data and metrics of the patient to make a diagnosis prediction. 🩺")
st.write("By pushing on the provided buttons you can get access to the statistical comparison amongs patient subgroups to guide your diagnosis.")

#Provide plots for comparison

fig1 = px.histogram(df, x='Gender', color="Group", barmode='group')

fig2 = px.histogram(df, x='Age', color='Group')

fig3 = px.histogram(df, x='Years of Education', color='Group')

fig4 = px.histogram(df, x='Socioeconomic Status', color="Group", barmode='group')

fig5 = px.histogram(df, x='Mini Mental State Examination', color="Group")

fig6 = px.histogram(df, x='Clinical Dementia Rating', color="Group", barmode='group')

#if st.button("Show me the distribution of patient Clinical Dementia Rating (CDR)."):
 #   st.header("The distribution of the patients according to Clinical Dementia Rating (CDR):")
  #  st.write("In the study group there are more non-demented patients")
   # st.plotly_chart(fig6)

fig7 = px.histogram(df, x='Estimated total intracranial volume', color="Group")

fig8 = px.histogram(df, x='Normalize Whole Brain Volume', color="Group")

fig9 = px.histogram(df, x='Atlas Scaling Factor', color="Group")


#User input of patient parameteres: 


patient_identification = st.text_input("Patient ID:", )

new_patient_gender = st.radio('Gender:', df['Gender'].unique())

if st.button("Show me comparison between male and female patients."):
    st.subheader("The geneder bar plot:")
    st.write("The male patients have higher amount of alzheimers gianosis")
    #show plot when click on the buuton 
    st.plotly_chart(fig1)

new_patient_age = st.number_input(
     'Age:', )

if st.button("Show me the age distribution of the patients."):
    st.subheader("The age histogram:")
    st.write("Between ages ~65 and ~80 there is strong increase in patients with alzheimers diagnosis")
    st.plotly_chart(fig2)

new_patient_educ = st.number_input(
     'Years of eductaion:', )

if st.button("Show me the years in education distribution of the patients."):
    st.subheader("The education histogram:")
    st.write("Patients with more years spent in education have lower probability for alzheimers diagnosis")
    st.plotly_chart(fig3)

new_patient_etiv = st.number_input(
     ' Estimated total intracranial volume (eTIV):', min_value=1100, max_value=2000, )

if st.button("Show me the estimated total intracranial volume (eTIV) in the patient studygroup."):
    st.subheader("The distribution of the patients estimated total intracranial volume (eTIV):")
    st.plotly_chart(fig7)

new_patient_asf= st.number_input(
     ' Atlas Scaling Factor (ASF):', min_value=0.9, max_value=1.5, )

if st.button("Show me the Atlas Scaling Factor (ASF) in the patient studygroup."):
    st.subheader("The distribution of the Atlas Scaling Factor (ASF):")
    st.plotly_chart(fig9)

new_patient_nWBV = st.number_input(
     ' Normalize Whole Brain Volume (nWBV):', min_value=0.65, max_value=0.825, )

if st.button("Show me the Normalize Whole Brain Volume (nWBV) in the patient studygroup."):
    st.subheader("The distribution of the Normalize Whole Brain Volume (nWBV):")
    st.write("With smaller nWBV, it is more likely to have alzheimers diagnosis.")
    st.plotly_chart(fig8)

new_patient_SES = st.selectbox(
     ' Socioeconomic status (SES, 1-5):', (1, 2, 3, 4,5))

if st.button("Show me the socioeconomic status (SES, 1-5) distribution of the patients."):
    st.subheader("The socioeconomic status bardiagram:")
    st.write("Patients belonging to higher SES category have lower probability for alzheimers diagnosis")
    st.plotly_chart(fig4)

new_patient_MMSE = st.number_input(
     'Mini Mental State Examination (MMSE) score:', min_value=1, max_value=30)

if st.button("Show me the distribution of the Mini Mental State Examination score amongst the patients."):
    st.subheader("The Mini Mental State Examination (MMSE) histogram:")
    st.write("Patients scoring less than 25 are likely to have dementia")
    st.plotly_chart(fig5)


st.write('')
st.write('')
st.write('')
st.write('')
st.write('')


#Genereate data frame from user inputs 

df_input = pd.DataFrame({
    'Patient ID': patient_identification, 
    'Gender':new_patient_gender, 
    'Age':new_patient_age, 
    'Years of Education':new_patient_educ , 
    'Estimated total intracranial volume':new_patient_etiv, 
    'Normalize Whole Brain Volume':new_patient_nWBV, 
    'Atlas Scaling Factor': new_patient_asf, 
    'Socioeconomic Status': new_patient_SES, 
    'Mini Mental State Examination':new_patient_MMSE
    }, index=[0])

#Display the dataframe 

st.header("Make a Prediction Based on Patient data")

if st.button('Generate input dataframe'):
    st.dataframe(df_input)
else:
    print('The needed data has not been provided.')

#Introduce the model 

from sklearn.linear_model import LogisticRegression

filename = 'Virtual_Assistant_AD_diagnosis/Streamlit_app/data/mmse_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Prep data 

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

cat_pipe = make_pipeline(
    OneHotEncoder(sparse=False, handle_unknown='ignore')
)

num_pipe = make_pipeline(
    MinMaxScaler()
)

feature_transform = ColumnTransformer([
    ('cat_processing', cat_pipe, ['Gender','Socioeconomic Status']), # 'CDR'
    ('num_preprocessing', num_pipe, ['Age', 'Years of Education', 'Estimated total intracranial volume', 'Normalize Whole Brain Volume', 'Atlas Scaling Factor', 'Mini Mental State Examination']),
])

X_background = df[['Gender', 'Age', 'Years of Education', 'Estimated total intracranial volume', 'Normalize Whole Brain Volume', 'Atlas Scaling Factor', 'Socioeconomic Status','Mini Mental State Examination']]
X_b_trans = feature_transform.fit_transform(X_background)

#Make a prediction based on user input 

if st.button('Generate prediction'):
    X = df_input[['Gender', 'Age', 'Years of Education', 'Estimated total intracranial volume', 'Normalize Whole Brain Volume', 'Atlas Scaling Factor', 'Socioeconomic Status','Mini Mental State Examination']]
    X_trans = feature_transform.transform(X)
    y_pred = loaded_model.predict(X_trans)
    probability= loaded_model.predict_proba(X_trans)
    prob= round(np.max(probability),2)*100
    result = f"The model suggests that patient diagnosis is: **{y_pred}** with lieklyhood of **{prob}%**."
    st.subheader('The machine learning model prediction:')
    st.write(result)

    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')


## ad a select box

if st.checkbox('Would you like to see a comparison to other patients?'):
    subgroup = st.selectbox("Which patient group do you want to see?", df['Group'].unique())
    df_subgroup = df.loc[df['Group'] == subgroup, df.columns].sample(10)
    st.dataframe(df_subgroup)

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

#Info about logistic regression 

ML_LR_pred = pd.read_csv('Virtual_Assistant_AD_diagnosis/Streamlit_app/data/Testing_ML_LR_model.csv')

bodyML = """ 
#Separate the data in test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

#Apply transformation - feature engineering
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

#Prepare pipline for categorical features - OneHotEncoding
cat_pipe = make_pipeline(
    OneHotEncoder(sparse=False, handle_unknown='ignore')
)

#Prepare pipline for numerical features - scale values between 0 and 1
num_pipe = make_pipeline(
    MinMaxScaler()
)

#Pipeline for feature engineering 
feature_transform = ColumnTransformer([
    ('cat_processing', cat_pipe, ['Gender,'SES']),
    ('num_preprocessing', num_pipe, ['Age', 'EDUC', 'eTIV', 'nWBV', 'ASF', 'MMSE']),
])

#Apply transformation
X_train_trans = feature_transform.fit_transform(X_train)
X_test_trans = feature_transform.transform(X_test)

#Train and test the model
from sklearn.linear_model import LogisticRegression
m = LogisticRegression()
m.fit(X_train, y_train)
y_predicted = m.predict(X_test)
probabilities = m.predict_proba(X_test) 
"""

st.header("The Machine Learning Model")

if st.checkbox('Would you like to know more about the model? 🧐'):
    st.write("The model is built using **sklearn linear model - Logistic Regression classifier** ")
    st.write("The baseline precision of the logistic regression model based on validation dataset is **78%** 📈 ")
    st.write('Input parameters like Gender and Socioeconomic Status are treated as categorical variables and processed with **(OneHotEncoding)**.')
    st.write('Input numerical values (Age, Years of Education, Estimated total intracranial volume, Normalize Whole Brain Volume, Atlas Scaling Factor, Mini Mental State Examination) are scaled between values 0 and 1 **(MinMaxScaler)**.')
    with st.expander('Display the code'):
        st.code(bodyML, language="python")
        st.write('')
    with st.expander('Testing the Logistic Regression model'):
       st.dataframe(ML_LR_pred)
    with st.expander('Basics of Logistic Regression model'):
        st.markdown("""Logistic regression is a classification algorithm. It is used to predict a binary outcome based on a set of independent variables.
    In this model activation function (sigmoid function) returns a probability (between 0 and 1) of the event taking place (e.g. True/False). The prediction of the outcome is assigned depending if the value of probability is greater or smaller then the pre-defined threshold.""")
        st.image('Virtual_Assistant_AD_diagnosis/Streamlit_app/data/LR_sketch.png')
    with st.expander('Resources'):
        st.write("Data source: https://www.kaggle.com/datasets/brsdincer/alzheimer-features")


st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

st.write('Created by Arita Silapetere')