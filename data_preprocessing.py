"""
This file has the functions to load and normalize the data.
"""

# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Function to data loading
def load_data(clinicalData_path, imgData_path, icuStay_path):
    clinicalData_df = pd.read_csv(clinicalData_path) 
    imgData_df = pd.read_csv(imgData_path)
    icuStay_df = pd.read_csv(icuStay_path)
    return clinicalData_df, imgData_df, icu_length_of_stay

# Function to data normalization
def normalize_data(clinicalData_df, imgData_df, icuStay_df):

    scaler_clinical = MinMaxScaler()
    scaler_img = MinMaxScaler()
    scaler_target = MinMaxScaler()

    clinicalData_df = pd.DataFrame(scaler_clinical.fit_transform(clinicalData_df), columns=clinicalData_df.columns)
    imgData_df = pd.DataFrame(scaler_img.fit_transform(imgData_df), columns=imgData_df.columns)
    icuStay_df['img_length_of_stay'] = scaler_target.fit_transform(icuStay_df[['img_length_of_stay']])


    return clinicalData_df, imgData_df, icuStay_df




