from tokenize import PlainToken
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

st.set_page_config(
    page_title="Virtual assistant demo",
    page_icon="🧠",
)

st.sidebar.header("Welcome to the virtual assistant for diagnosis of Alzheimer's disease. \n Please select one of the pages listed above.")

#add title and description 
st.title("Virtual Assistant for Diagnosis of Alzheimer's disease")
st.write("Demo app for machine learning - logistic regression and convolutional neural network (CNN) based predictions.")

st.image("Logo2.png")
st.write('Created by Arita Silapetere')

st.write(' ')
st.write(' ')
st.markdown("""
Artificial intellgiance and Machine Learning have multiple applications in the healthcare indurty.
The AI and ML can be employed for the analysis of the complex datasets and metrics of the patients, and to make diagnosis predictions and treatment recommendations. 
These methods provide options for personlized, more efficient healthcare and are important part of the future of the patient care.
\n
\n

With this project I present a case study of a virtual assistant for healthcare workers to aid the diagnosis of Alzheimer's disease.
\n
Alzheimer’s disease (AD) is a brain disorder, which results in loss of cognitive function (remembering, reasoning), affecting the persons daily life and ability to carry out basic tasks. Changes in the brain have been assocoiated with development of AD development - loss of neuron conections, formation of amyloid plaques and neurofibrillary tangles in brain.
Early diagnosis of the AD allows to start treatment early and preserve the patients cognitive abilities for longer, help families plan the future (safty, leagal and living arrangemnts) and provide opportonity to choose to participate in a clinical trial for AD cure. 
The key steps in the AD diagnosis iclude: 
* Standard medical tests (e. g. blood tests) 
* Interview with the patient and their familiy to asses the overal health and cognitive abilities.
* Carry out memory and problem solving tests. 
* Investigate brain with imaging techniques - computed tomography (CT), magnetic resonance imaging (MRI), or positron emission tomography (PET). 
\n
\n

The virtual assistant is created to aid the diagnosis process of the hethcare workers by providing AI based predictions and provide comparison to other patients. 
It consists of two parts: 
1) ML model prediction of the diagnosis, based on the patient entry data. Comparison with the statictis of the other patient subgroups. 
2) Convolutional Neural Network (CNN) magnetic resonance image (MRI) classification.
\n
""")

