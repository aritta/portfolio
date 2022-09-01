from tokenize import PlainToken
import streamlit as st
import pandas as pd

st.set_page_config(page_title="CNN image classification", page_icon="📊")

st.sidebar.header("Magnetic resonance image classification using artificial intelligance. \n Based on the provided 2D crossection of MRI scan of a patient, a deep learning model provides prediction of Alzheimer's disease diagnosis.")

st.title("Magnetic Resonance Image Classification Using Artificial Intelligance.")

st.write("This appa lows you to provide a 2D slice of a magnetic resonance image (MRI) for image classification. Based on convolutional neural network (CNN) the image will be classified in one of the four subgroups -  Non Demented, Very Mild Demented, Mild Demented or Moderate Demented")

st.write("Currently online version only has a **preview version of image classification.**")

st.header("Prveiew Image classifier.")

cnn_classifier = """
#Prepare the model 

#Import packages
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array 
from tempfile import NamedTemporaryFile

#image upload
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Please provide an MRI 2D image:")
if st.button("Show me the patient mri scan."):
    st.write("The crossesction 2D image:")
    st.image(uploaded_file)
temp_file = NamedTemporaryFile(delete=False)

#Import model
cnn_model = load_model("RSP_weights_C.h5")


#Prep the data - create an numpy array containing image pixel intensity distribution. And make a prediction, assign a label. 
X_test=[]

if uploaded_file is not None:
    temp_file.write(uploaded_file.getvalue())
    image = load_img(temp_file.name, color_mode="grayscale")
    image_array = img_to_array(image)
    shape_array=image_array.shape
    shape=f"The array shape is{shape_array}"
    st.write(shape)
    X_test.append(image_array)
    X_test_array = np.array(X_test)
    #Make a predition
    y_pred = cnn_model.predict(X_test_array)
    #Assign labels 
    for i in y_pred:
        perc = np.max(i)
        likely = round((perc*100), 1)#Likelyhood 
        if np.argmax(i) ==0:
            st.write("The patiend likely belongs to diagnosis group - **Very Mild Demented**") 
        elif np.argmax(i) ==1:
            st.write("The patiend likely belongs to diagnosis group - **Non Demented**") 
        elif np.argmax(i) ==2:
            st.write("The patiend likely belongs to diagnosis group - **Moderate Demented**")
        elif np.argmax(i) ==3:
            st.write("The patiend likely belongs to diagnosis group - **Mild Demented**")  
        else:
            st.write("The prediction could not be properly assigned")
        st.write(f"With likelyhood of **{likely}**")

#Buttons for images:

import random, os

base_path = "../alzheimer_mri/train"
classes = ["VeryMildDemented", "NonDemented", "ModerateDemented", "MildDemented"]
option = st.selectbox('Would you like to see sample of a subcategory?',classes)


#to dispay for random images from trainig dataset with the same diagnosis. 
if st.button("Show me sample mri scan."):
    dir = f"{base_path}/{option}"
    files = os.listdir(dir)
    index1 = random.choice(files)
    index2 = random.choice(files)
    index3 = random.choice(files)
    index4 = random.choice(files)
    random_filename1=f"{base_path}/{option}/{index1}"
    random_filename2=f"{base_path}/{option}/{index2}"
    random_filename3=f"{base_path}/{option}/{index3}"
    random_filename4=f"{base_path}/{option}/{index4}"
    images = [random_filename1, random_filename2, random_filename3, random_filename4]
    st.image(images, width = 150)
"""

with st.expander('Step 1:'):
    st.write("Choose an MRI image of the test set for upload")
    st.image("Virtual_Assistant_AD_diagnosis/Streamlit_app/data/Step1.png")
with st.expander('Step 2:'):
    st.write("The app will provide an image of the MRI scan, predicted class (Non Demented, Very Mild Demented, Mild Demented or Moderate Demented) and the probability.")
    st.image("Virtual_Assistant_AD_diagnosis/Streamlit_app/data/Step2.png")
with st.expander('Step 3:'):
    st.write("The healthcare provider can access random selection of four images form the training set with the same class/diagnosis for comparison.")
    st.image("Virtual_Assistant_AD_diagnosis/Streamlit_app/data/Step3.png")
with st.expander('Show the code:'):
    st.code(cnn_classifier, language="python")



st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

#Information about the model 

body = """model_C = Sequential([
    
    # first convolutional and max pooling layer
    Conv2D(filters=5,
          kernel_size=(6,6),
          strides=(1,1),
          padding="same",
          activation="relu",
          input_shape=(208, 176, 1)),
    MaxPooling2D(pool_size=(2,2),
                strides=(2,2),
                padding="same"),
    #BatchNormalization(), # optional

    # second convolutional and max pooling layer
    Conv2D(filters=15,
          kernel_size=(2,2),
          strides=(1,1),
          padding="same",
          activation="relu",
          input_shape=(51, 43, 5)),
    MaxPooling2D(pool_size=(2,2),
                strides=(2,2),
                padding="same"),
    
    #Flatten
    Flatten(),
    
    # Fully connected
    # layer 1
    Dense(500, activation="elu"),
    BatchNormalization(),
    Dropout(0.2),

    # layer 2
    Dense(700, activation="elu"),
    BatchNormalization(),
    Dropout(0.2),
    
    # layer 3
    Dense(100, activation="elu"),
    BatchNormalization(),
    Dropout(0.2),

    # Output layer
    Dense(4, activation="softmax")
])"""

model_history = pd.read_csv('Virtual_Assistant_AD_diagnosis/Streamlit_app/data/model_C_history.csv', delimiter=';', index_col=None, decimal=',')
df_testing_model_C = pd.read_csv('Virtual_Assistant_AD_diagnosis/Streamlit_app/data/Testing_CNN_model.csv')
#df_testing_model_C.drop(columns=['Unnamed: 0'], inplace = True)

import plotly.graph_objects as go


st.header('Deep Learning model')

#Model info 
if st.checkbox('Would you like to know more about the CNN model? 🧐'):
   st.write('How does CNN work: 🧠')
   st.image('Virtual_Assistant_AD_diagnosis/Streamlit_app/data/CNN_layers.png')
   st.write('')
   st.write("""
   CNN (convolutional neural networks) is a type of a deep learning models used for processing data that has a grid pattern(image). 
   The CNN model automatically and adaptively learns spatial hierarchies of features, from low- to high-level patterns. \n
   A CNN architecture is formed by a sequence of distinct layers (shown above). 
   The input layer contains an image. Convolutional layers are used to reduce the complexity of image and and the number of input nodes. Convolutional filter (kernel) take advantage of the correlation of the neighboring pixels. 
   The dot produt of the input array and the filter provide a feature map with lower dimensions. This is followed by max value filter (maxpooling), to further reduce the number of the input nodes. 
   After (multiple) convolutional layers, the array goes through flattening layer and this serves as an input to the fully connected layers. 
   The weighted features then are passed through an activation function (softmax) which provides probabilities for each possible output label. 
   """)
   with st.expander('Display code:'):
       st.write("The **CNN** model architecture")
       st.code(body, language="python")
       st.write('')
       st.write("Model summary")
       cnn_model.summary(print_fn=lambda x: st.text(x))
       st.write('The model used **250 batch size** and **15 epochs**.')
   with st.expander('How did the model perform during training?'):
       fig11 = go.Figure()
       fig11.add_trace(go.Line(x=model_history.index, y=model_history["categorical_accuracy"], name='Training data categorigal accuracy', line=dict(color='purple', width=4)))
       fig11.add_trace(go.Line(x=model_history.index, y=model_history["val_categorical_accuracy"], name='Validation data categorigal accuracy', line=dict(color='orange', width=4)))
       fig11.update_layout(xaxis=dict(title_text="Epoch"))
       fig12 = go.Figure()
       fig12.add_trace(go.Scatter(x=model_history.index, y=model_history["loss"], name='Training data loss', line=dict(color='purple', width=4)))
       fig12.add_trace(go.Scatter(x=model_history.index, y=model_history["val_loss"], name='Validation data data loss', line=dict(color='orange', width=4)))
       fig12.update_layout(xaxis=dict(title_text="Epoch"))
       st.plotly_chart(fig11)
       st.plotly_chart(fig12)
   with st.expander('Testing the model.'):
       st.write('However once the model was applied on testing set, it showed much lower accuracy.')
       st.write('The model could predict dementia with **68%** accuracy.')
       st.write('However the correct assignment of each of the four subcategories was only **57%**')
       st.write('The results suggest strong **overfitting** during the training steps of the model.')
       st.dataframe(df_testing_model_C)
   with st.expander('How to improve the model?'):
       st.write('Hyperparameter tuning in Deep Neural Network')
       """
       * Number of Hidden Layers
            * Choose the number of hidden layers between the input layer and the output layer.
       * Dropout
            * Dropout is regularization technique to avoid overfitting, it 'switches off' random neurons (~20%) during the fitting process.
       * Batch Normalization 
            * Batch normalization adds two hyperparamters per layer that normalize the inputs to the activation function 
       * Choose an activation function 
            * Activation functions are used to introduce nonlinearity to models, which allows deep learning models to learn nonlinear prediction boundaries.
            * Rule of thumb: ELU  > Leaky ReLU > ReLU > tanh > sigmoid (depicted below)
            * For output layer Sigmoid is used for binary predictions and Softmax is used for multi-class predictions.
       * Hyperparameters related to Training Algorithm
            * Number of epochs - number of times the whole training data is shown to the network while training.
            * Batch size - number of sub samples given to the network after which parameter update happens.
       * Avoid overfitting with image pre-processing - flip (horizontal/vertical), rotation, rescale, zoom, shift
       * Use pretrained models and transfer learning! 
       * Use much more layers and neurons!
            * E.g., reported studies with Inception-v4 & 497 layers - https://www.nature.com/articles/s41598-020-79243-9
       """
       st.image('Virtual_Assistant_AD_diagnosis/Streamlit_app/data/activation_functions.png')
       st.caption('Image credit - Spiced Acdemy')
   with st.expander('Resources'):
       st.write('Tensorflow image classification: https://www.tensorflow.org/tutorials/images/cnn')
       st.write("Source for the MRI scans: https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images")

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

st.write('Created by Arita Silapetere')