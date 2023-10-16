import streamlit as st
import pandas as pd
import numpy as np

from transform_module import transform
from tflite_runtime.interpreter import Interpreter

import os

absolute_path = os.path.dirname(__file__)

scaler_path=os.path.join(absolute_path,"preprocessing_files/robust_scaler.pkl" )
ohe_path=os.path.join(absolute_path,"preprocessing_files/onehot_encoder.pkl" )
pca_path=os.path.join(absolute_path,"preprocessing_files/pca.pkl")
model_path = os.path.join(absolute_path,'model_files/quantization_aware_quantized_model.tflite' )

files_exist = all(
    os.path.exists(path)
    for path in [scaler_path, ohe_path, pca_path, model_path]
)
# Display a message if any of the files are missing
if not files_exist:
    st.write("Some required files are missing. Please ensure all files are loaded.",font_size=30)

# Define the categorical values for each column
occupation_values = [
    'Software Engineer', 'Sales Representative', 'Doctor', 'Lawyer',
    'Graphic Designer', 'Business Consultant', 'Marketing Specialist', 'CEO',
    'Project Manager', 'Engineer', 'Accountant', 'Architect', 'Nurse', 'Analyst',
    'Teacher', 'IT Manager', 'Writer', 'Business Analyst', 'Software Developer'
]

education_values = [
    "Master's Degree", "Bachelor's Degree", "Doctorate Degree", "Associate's Degree"
]

marital_status_values = ['Married', 'Single', 'Divorced', 'Widowed']

favorite_color_values = ['Blue', 'Green', 'Purple', 'Red', 'Yellow', 'Black', 'Pink', 'Orange', 'Grey']

# Define the column names and data types
columns = [' Age', ' Height (cm)', ' Weight (kg)', ' Occupation',
           ' Education Level', ' Marital Status', ' Income (USD)', ' Favorite Color']
data_types = {' Age': int, ' Height (cm)': int, ' Weight (kg)': int,
              ' Occupation': 'object', ' Education Level': 'object',
              ' Marital Status': 'object', ' Income (USD)': int, ' Favorite Color': 'object'}

# Create an empty DataFrame with the specified columns
df = pd.DataFrame(columns=columns).astype(data_types)

# Streamlit app
st.title('Simple Gender Classification')

for column in [' Age', ' Height (cm)', ' Weight (kg)', ' Income (USD)']:
    df.at[0,column] = st.number_input(f'Enter {column}', key=column)

 
occupation = st.selectbox('Select Occupation', options=[''] + occupation_values, key='Occupation')
if occupation:
    df.at[0, ' Occupation'] = occupation

 
education = st.selectbox('Select Education Level', options=[''] + education_values, key='Education Level')
if education:
    df.at[0, ' Education Level'] = education

marital_status = st.selectbox('Select Marital Status', options=[''] + marital_status_values, key='Marital Status')
if marital_status:
    df.at[0, ' Marital Status'] = marital_status

favorite_color = st.selectbox('Select Favorite Color', options=[''] + favorite_color_values, key='Favorite Color')
if favorite_color:
    df.at[0, ' Favorite Color'] = favorite_color


if st.button('Run Inference') and df.iloc[0].notna().all():

 features=transform(dataframe=df,scaler_filepath=scaler_path,
                   encoder_filepath=ohe_path,pca_filepath=pca_path)

 interpreter = Interpreter(model_path)
 interpreter.allocate_tensors()


 input_details = interpreter.get_input_details()
 input_shape = input_details[0]['shape']


 input_data = np.array([features], dtype=np.float32)
 input_data = input_data.reshape(input_shape)

 interpreter.set_tensor(input_details[0]['index'], input_data)

 interpreter.invoke()

 output_details = interpreter.get_output_details()
 output_data = interpreter.get_tensor(output_details[0]['index'])

 interpreter = None

 result = "male" if output_data > 0.5 else "female"
 st.write('Inference Result:', result,font_size=40)




