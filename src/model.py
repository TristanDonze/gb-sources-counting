import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ResidualConvBlock(nn.Module):
    def __init__(self, 
                 channels : int, 
                 kernel_size : int = 5, 
                 dilation : int = 1, 
                 dropout : int = 0.05):
        super().__init__()

        padding = (kernel_size // 2) * dilation

        self.block = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, dilation=dilation)),
            ('norm_1', nn.BatchNorm1d(channels)),
            ('gelu_1', nn.GELU()),
            ('dropout', nn.Dropout(dropout)),
            ('conv_2', nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, dilation=dilation)),
            ('norm_2', nn.BatchNorm1d(channels)),
        ]))
                
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.activation(x + self.block(x))

class PosEnc(nn.Module):
    def __init__(self, 
                 dim_model : int = 256, 
                 max_len : int = 10_000):
        super(PosEnc, self).__init__()

        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-np.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)] # (batch_size, N', dim_model)

class PoolingConcat(nn.Module):
    def __init__(self, 
                 dim_model : int = 256, 
                 attn_dim : int = 256):
        super().__init__()

        self.attn = nn.Linear(dim_model, attn_dim)
        self.context_vec = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x):
        mean_pool = x.mean(dim=1) # (batch_size, dim_model)
        max_pool, _ = x.max(dim=1) # (batch_size, dim_model)

        a = torch.tanh(self.attn(x)) # (batch_size, N', attn_dim)
        scores = self.context_vec(a).squeeze(-1) # (batch_size, N')
        attn_weights = F.softmax(scores, dim=-1)
        attn_pool = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

        return torch.cat([mean_pool, max_pool, attn_pool], dim=-1)

class CardinalityEstimator(nn.Module):
    def __init__(self, 
                 learning_strategy : str = "mse", 
                 max_K : int = 10, 
                 input_channels : int = 4,
                 dim_model : int = 256):
        super(CardinalityEstimator, self).__init__()

        self.max_K = max_K
        self.dim_model = dim_model

        self.conv_encoder = nn.Sequential(OrderedDict([ 
            ('stem_conv_1', nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, stride=1, padding=2)),
            ('stem_norm_1', nn.BatchNorm1d(64)),
            ('stem_gelu_1', nn.GELU()),
            ('stem_conv_2', nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)),
            ('stem_norm_2', nn.BatchNorm1d(128)),
            ('stem_gelu_2', nn.GELU()),
            ('res_block_1', ResidualConvBlock(channels=128, kernel_size=5, dilation=1, dropout=0.05)),
            ('res_block_2', ResidualConvBlock(channels=128, kernel_size=5, dilation=1, dropout=0.05)),
            ('res_block_3', ResidualConvBlock(channels=128, kernel_size=5, dilation=1, dropout=0.05)),
            ('downsample', nn.Conv1d(in_channels=128, out_channels=dim_model, kernel_size=5, stride=2, padding=2)),
            ('downsample_norm', nn.BatchNorm1d(dim_model)),
            ('downsample_gelu', nn.GELU()),
            ('res_block_4', ResidualConvBlock(channels=dim_model, kernel_size=3, dilation=1, dropout=0.05)),
            ('res_block_5', ResidualConvBlock(channels=dim_model, kernel_size=3, dilation=2, dropout=0.05)),
        ]))

        self.pos_encoder = PosEnc(dim_model=dim_model, max_len=10_000)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model = dim_model,
            nhead = 4,
            dim_feedforward = dim_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )

        self.pooling = PoolingConcat(dim_model=dim_model, attn_dim=dim_model)

        self.classifier = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features = dim_model * 3, out_features = dim_model * 2)),
            ('gelu_1', nn.GELU())
        ]))

        if learning_strategy == "mse":
            self.classifier.add_module('output', nn.Linear(in_features=2*dim_model, out_features=1))
        elif learning_strategy == "cross_entropy":
            self.classifier.add_module('output', nn.Linear(in_features=2*dim_model, out_features=self.max_K))
        elif learning_strategy == "ordinal":
            self.classifier.add_module('output', nn.Linear(in_features=2*dim_model, out_features=self.max_K - 1))
        else:
            raise ValueError(f"Unknown learning strategy: {learning_strategy}")



    def forward(self, x):
        out = self.conv_encoder(x) # (batch_size, dim_model, N')
        out = out.permute(0, 2, 1) # (batch_size, N', dim_model)
        out = self.pos_encoder(out) # (batch_size, N', dim_model)

        out = self.transformer_encoder(out) # (batch_size, N', dim_model)
        out = self.pooling(out) # (batch_size, dim_model * 3)

        out = self.classifier(out) # (batch_size, out_features)
        
        return out


if __name__ == "__main__":
    x = torch.rand(32, 4, 128)
    print(x.shape)
    model = CardinalityEstimator(learning_strategy="ordinal", max_K=10, input_channels=4, dim_model=256)
    out = model(x)
    print(out.shape)