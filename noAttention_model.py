"""
Model with no attention layers
"""

# import libraries
import torch
import torch.nn as nn
import torch.optim as optim

class ICUModel_NoAttention(nn.Module):
    def __init__(self, clinical_input_dim, img_input_dim, hidden_dims, dropout_rate):
        super(ICUModel_NoAttention, self).__init__()

        # Clinical branch
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Image branch
        self.img_net = nn.Sequential(
            nn.Linear(img_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Combined branch
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dims[3] * 2, hidden_dims[4]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[4], 1)
        )

    def forward(self, clinical_data, img_data):
        clinical_features = self.clinical_net(clinical_data)
        img_features = self.img_net(img_data)
        combined_features = torch.cat((clinical_features, img_features), dim=1)
        output = self.combined_net(combined_features)
        return output
