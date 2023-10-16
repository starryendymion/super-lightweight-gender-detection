import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.decomposition import PCA
import joblib

def transform(dataframe,encoder_filepath,scaler_filepath,pca_filepath):
    
    encoder = joblib.load(encoder_filepath)
    scaler = joblib.load(scaler_filepath)
    pca_object = joblib.load(pca_filepath)

    categorical_data=dataframe[[' Occupation',
       ' Education Level', ' Marital Status',
       ' Favorite Color']]
    
    encoded_data=encoder.transform(categorical_data)

    numeric_data=np.array(dataframe[[' Age', ' Height (cm)', ' Weight (kg)',' Income (USD)']])

    features=np.concatenate((numeric_data,encoded_data),axis=1)

    scaled_features=scaler.transform(features)

    reduced_features=pca_object.transform(scaled_features)

    return reduced_features


