"""
# Hyperparameter optimization with Optuna
"""


# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import optuna
import numpy as np
from train_evaluate import train_model
from selfAttention_model import ICUModel_SelfAttention
from attention_model import ICUModel_Attention
from noAttention_model import ICUModel_NoAttention
from crossAttention_model import ICUModel_CrossAttention
from onlyClinicalfeatures_model import ICUModel_NoImages

class Optuna_tuning:
  def __init__(self, clinicalData_df, imgData_df, icu_stay, train_loader, val_loader, device, epochs):
    self.clinicalData_df = clinicalData_df
    self.imgData_df = imgData_df
    self.icu_stay = icu_stay
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.device = device
    self.epochs = epochs

  # weights initialization
  def init_weights(m):
      if isinstance(m, nn.Linear):
          nn.init.xavier_uniform_(m.weight)
          nn.init.zeros_(m.bias)


  # optuna objective function
  def objective(trial,model_type):
    clinical_input_dim = clinicalData_df.shape[1]
    img_input_dim = imgData_df.shape[1]
    hidden_dims = [
        trial.suggest_categorical('hidden_dim1', [64, 128, 256]),
        trial.suggest_categorical('hidden_dim2', [32, 64, 128]),
        trial.suggest_categorical('hidden_dim3', [16, 32, 64]),
        trial.suggest_categorical('hidden_dim4', [8, 16, 32]),
        trial.suggest_categorical('hidden_dim5', [4, 8, 16])
    ]

    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)


    if model_type == 'attention':
      model = ICUModel_Attention(clinical_input_dim, img_input_dim, hidden_dims, dropout_rate)
    elif model_type == 'no_attention':
      model = ICUModel_NoAttention(clinical_input_dim, img_input_dim, hidden_dims, dropout_rate)
    elif model_type == 'only_self_attention':
      model = ICUModel_SelfAttention(clinical_input_dim, img_input_dim, hidden_dims, dropout_rate)
    elif model_type == 'only_cross_attention':
      model = ICUModel_CrossAttention(clinical_input_dim, img_input_dim, hidden_dims, dropout_rate)
    elif model_type == 'no_attention_no_images':
      model = ICUModel_NoImages(clinical_input_dim, hidden_dims, dropout_rate)
    else:
      raise ValueError("Invalid model type. Choose from 'attention', 'no_attention', 'only_self_attention', 'only_cross_attention' or 'no_attention_no_images'.")

    # Apply weight initialization
    model.apply(init_weights)

    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_model(model, criterion, optimizer, train_loader, val_loader, epochs=self.epochs, device=self.device)

    # Validation loss as objective
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for clinical_data, img_data, icu_stay in val_loader:
            clinical_data, img_data, icu_stay = clinical_data.to(device), img_data.to(device), icu_stay.to(device)
            
            # Conditional forward pass based on model type
            if model_type == 'no_attention_no_images':
              predictions = model(clinical_data)  # Pass only clinical data for ICUModel_NoImages
            else:
              predictions = model(clinical_data, img_data)  # Pass both for other models

            loss = criterion(predictions, icu_stay)
            val_loss += loss.item()
    return val_loss / len(val_loader)

    