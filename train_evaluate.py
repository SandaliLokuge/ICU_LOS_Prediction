"""
# Define training and evaluation functions
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
from selfAttention_model import ICUModel_SelfAttention
from attention_model import ICUModel_Attention
from noAttention_model import ICUModel_NoAttention
from crossAttention_model import ICUModel_CrossAttention
from onlyClinicalfeatures_model import ICUModel_NoImages

# function to train the model
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    model.to(device)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for clinical_data, img_data, icu_stay in train_loader:
            clinical_data, img_data, icu_stay = clinical_data.to(device), img_data.to(device), icu_stay.to(device)
            optimizer.zero_grad()
            
            # Conditional forward pass based on model type
            if isinstance(model, ICUModel_NoImages):  # Check if model is ICUModel_NoImages
                predictions = model(clinical_data)  # Pass only clinical data
            else:
                predictions = model(clinical_data, img_data)  # Pass both clinical and image data

            loss = criterion(predictions, icu_stay)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clinical_data, img_data, icu_stay in val_loader:
                clinical_data, img_data, icu_stay = clinical_data.to(device), img_data.to(device), icu_stay.to(device)
                
                # Conditional forward pass based on model type
                if isinstance(model, ICUModel_NoImages):  # Check if model is ICUModel_NoImages
                    predictions = model(clinical_data)  # Pass only clinical data
                else:
                    predictions = model(clinical_data, img_data)  # Pass both clinical and image data

                loss = criterion(predictions, icu_stay)
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")

    return train_losses, val_losses

# function to evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)
    test_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for clinical_data, img_data, icu_stay in test_loader:
            clinical_data, img_data, icu_stay = clinical_data.to(device), img_data.to(device), icu_stay.to(device)
            # Conditional forward pass based on model type
            if isinstance(model, ICUModel_NoImages):  # Check if model is ICUModel_NoImages
                predictions = model(clinical_data)  # Pass only clinical data
            else:
                predictions = model(clinical_data, img_data)  # Pass both clinical and image data
            loss = criterion(predictions, icu_stay)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")


# function to train the model with best hyperparameters
def train_final_model(study,clinicalData_df,imgData_df,train_loader,val_loader,test_loader,model_type,epochs,device):
    best_params = study.best_params
    clinical_input_dim = clinicalData_df.shape[1]
    img_input_dim = imgData_df.shape[1]
    hidden_dims = [best_params['hidden_dim1'], best_params['hidden_dim2'], best_params['hidden_dim3'], best_params['hidden_dim4'],best_params['hidden_dim5']]

    dropout_rate = best_params['dropout_rate']
    optimizer_name = best_params['optimizer']
    lr = best_params['lr']

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


    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=epochs, device=device)
    evaluate_model(model, test_loader, device=device)

    # Plot the training and validation loss for the best model
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss for Best Model')
    plot_name=model_type +'_model_loss_plot.png'
    plt.savefig(plot_name)
    plt.show()

    # Save the best model
    model_name = model_type +'_best_model.pth'
    torch.save(model.state_dict(), model_name)

