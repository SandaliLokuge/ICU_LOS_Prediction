"""
Model with self attention layers only
"""

# import libraries
import torch
import torch.nn as nn
import torch.optim as optim


class ICUModel_SelfAttention(nn.Module):
    def __init__(self, clinical_input_dim, img_input_dim, hidden_dims, dropout_rate):
        super(ICUModel_SelfAttention, self).__init__()

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

        # Self-attention layers
        self.self_attention_clinical = nn.MultiheadAttention(embed_dim=hidden_dims[3], num_heads=4, batch_first=True)
        self.self_attention_img = nn.MultiheadAttention(embed_dim=hidden_dims[3], num_heads=4, batch_first=True)

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

        # Self-attention for clinical data
        clinical_features = clinical_features.unsqueeze(1)
        clinical_features, _ = self.self_attention_clinical(clinical_features, clinical_features, clinical_features)
        clinical_features = clinical_features.squeeze(1)

        # Self-attention for image data
        img_features = img_features.unsqueeze(1)
        img_features, _ = self.self_attention_img(img_features, img_features, img_features)
        img_features = img_features.squeeze(1)

        # Final combined branch
        output = self.combined_net(torch.cat((clinical_features, img_features), dim=1))
        return output
