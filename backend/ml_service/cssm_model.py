import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(ScaleEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return out, (hn, cn)

class CrossScaleAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super(CrossScaleAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, queries, keys, values):
        N = queries.shape[0]
        Q = self.query(queries).view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(keys).view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(values).view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_weights, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(N, -1, self.hidden_size)
        out = self.fc_out(out)

        return out, attention_weights

class CSSMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_scales=4, num_heads=4):
        super(CSSMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_scales = num_scales

        self.hourly_encoder = ScaleEncoder(input_size, hidden_size)
        self.daily_encoder = ScaleEncoder(input_size, hidden_size)
        self.weekly_encoder = ScaleEncoder(input_size, hidden_size)
        self.monthly_encoder = ScaleEncoder(input_size, hidden_size)

        self.cross_scale_attention = CrossScaleAttention(hidden_size, num_heads)

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * num_scales, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        self.hourly_predictor = nn.Linear(hidden_size, 24)
        self.daily_predictor = nn.Linear(hidden_size, 7)
        self.weekly_predictor = nn.Linear(hidden_size, 4)
        self.monthly_predictor = nn.Linear(hidden_size, 12)

    def forward(self, hourly_data, daily_data, weekly_data, monthly_data):
        hourly_out, _ = self.hourly_encoder(hourly_data)
        daily_out, _ = self.daily_encoder(daily_data)
        weekly_out, _ = self.weekly_encoder(weekly_data)
        monthly_out, _ = self.monthly_encoder(monthly_data)

        hourly_last = hourly_out[:, -1, :]
        daily_last = daily_out[:, -1, :]
        weekly_last = weekly_out[:, -1, :]
        monthly_last = monthly_out[:, -1, :]

        scales = torch.stack([hourly_last, daily_last, weekly_last, monthly_last], dim=1)
        attended_scales, attention_weights = self.cross_scale_attention(scales, scales, scales)

        fused = torch.cat([
            attended_scales[:, 0, :],
            attended_scales[:, 1, :],
            attended_scales[:, 2, :],
            attended_scales[:, 3, :]
        ], dim=1)

        fused_features = self.fusion_layer(fused)

        hourly_pred = self.hourly_predictor(fused_features)
        daily_pred = self.daily_predictor(fused_features)
        weekly_pred = self.weekly_predictor(fused_features)
        monthly_pred = self.monthly_predictor(fused_features)

        return {
            'hourly': hourly_pred,
            'daily': daily_pred,
            'weekly': weekly_pred,
            'monthly': monthly_pred,
            'attention_weights': attention_weights
        }

def create_model(input_size=10, hidden_size=128, num_scales=4, num_heads=4):
    model = CSSMModel(input_size, hidden_size, num_scales, num_heads)
    return model

def save_model(model, path='cssm_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(path='cssm_model.pth', input_size=10, hidden_size=128):
    model = create_model(input_size, hidden_size)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
