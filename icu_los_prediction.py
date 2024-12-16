"""
This python file is to predict the ICU length of stay using clinical information 
(demographics, lab events, chart events) and xray images embeddings.
"""

# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import load_data, normalize_data
import create_dataset as cd
import optuna_tuning  as ot
from train_evaluate import train_final_model

# Use GPU
try:
  if(torch.cuda.is_available()):
      print("GPU successfully detected - ")
      print(torch.cuda.get_device_name(0))
      device = torch.device("cuda:0")
except Exception as e:
  print("GPU not detected. Change the settings as mentioned earlier and run session again")
  device = torch.device("cpu")

#Setting random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

EPOCHS = 10
N_TRIALS = 5

# Load data
clinicalData_path = "Deep_learning/Project/Demo_CE_LE.csv" # change this line according to ur path
imgData_path = "Deep_learning/Project/vd.csv" # change this line according to ur path
icuStay_path = "Deep_learning/Project/y.csv" # change this line according to ur path

clinicalData_df, imgData_df, icuStay_df = load_data(clinicalData_path, imgData_df, icuStay_df)

# Data normalization
clinicalData_df, imgData_df, icuStay_df = normalize_data(clinicalData_df, imgData_df, icuStay_df)

# Create dataset
dataset = cd.ICUDataset(clinicalData_df, imgData_df, icuStay_df['img_length_of_stay'])

# Split dataset into training, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Parameter tuning with optuna
optuna_ob = ob.Optuna_tuning(clinicalData_df, imgData_df, icu_stay, train_loader, val_loader, device, EPOCHS)

# Model with attention layers
study_attention = optuna.create_study(direction='minimize')
study_attention.optimize(lambda trial: optuna_ob.objective(trial, model_type='attention'), n_trials=N_TRIALS)
print(study_attention.best_params)

train_final_model(study_attention,clinicalData_df,imgData_df,train_loader,val_loader,test_loader,"attention",EPOCHS,device):


# Model without attention layers

study_noattention = optuna.create_study(direction='minimize')
study_noattention.optimize(lambda trial: optuna_ob.objective(trial, model_type='no_attention'), n_trials=N_TRIALS)
print(study_noattention.best_params)

train_final_model(study_noattention,clinicalData_df,imgData_df,train_loader,val_loader,test_loader,"no_attention",EPOCHS,device):


# Model with only self attention layer

study_selfattention = optuna.create_study(direction='minimize')
study_selfattention.optimize(lambda trial: optuna_ob.objective(trial, model_type='only_self_attention'), n_trials=N_TRIALS)
print(study_selfattention.best_params)

train_final_model(study_selfattention,clinicalData_df,imgData_df,train_loader,val_loader,test_loader,"only_self_attention",EPOCHS,device):


# Model with only cross attention layer

study_crossattention = optuna.create_study(direction='minimize')
study_crossattention.optimize(lambda trial: optuna_ob.objective(trial, model_type='only_cross_attention'), n_trials=N_TRIALS)
print(study_crossattention.best_params)

train_final_model(study_crossattention,clinicalData_df,imgData_df,train_loader,val_loader,test_loader,"only_cross_attention",EPOCHS,device):


# Model with only clinical features no xray images

study_noimages = optuna.create_study(direction='minimize')
study_noimages.optimize(lambda trial: optuna_ob.objective(trial, model_type='no_attention_no_images'), n_trials=N_TRIALS)
print(study_noimages.best_params)

train_final_model(study_noimages,clinicalData_df,None,train_loader,val_loader,test_loader,"only_cross_attention",EPOCHS,device):
