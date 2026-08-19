import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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

class DualHeadOutput(nn.Module):
    def __init__(self, in_features: int, max_K: int):
        super().__init__()
        self.mse_head = nn.Linear(in_features=in_features, out_features=1)
        self.ce_head = nn.Linear(in_features=in_features, out_features=max_K)
    
    def forward(self, x):
        return self.mse_head(x), self.ce_head(x)


class TripleHeadOutput(nn.Module):
    def __init__(self, in_features: int, max_K: int):
        super().__init__()
        self.mse_head = nn.Linear(in_features=in_features, out_features=1)
        self.ce_head = nn.Linear(in_features=in_features, out_features=max_K)
        self.ordinal_head = nn.Linear(in_features=in_features, out_features=max_K - 1)

    def forward(self, x):
        return self.mse_head(x), self.ce_head(x), self.ordinal_head(x)

class CardinalityEstimator(nn.Module):
    def __init__(self, 
                 learning_strategy: str = "mse", 
                 max_K: int = 10, 
                 input_channels: int = 4,
                 dim_model: int = 196,
                 channel_multiplier: int = 1,
                 conv_1_kernel_size: int = 7,
                 conv_2_kernel_size: int = 5,
                 conv_3_kernel_size: int = 5,
                 conv_4_kernel_size: int = 3,
                 conv_2_stride: int = 2,
                 transformer_nhead: int = 4,
                 transformer_ff_multiplier: float = 2,
                 transformer_num_layers: int = 2):
        super(CardinalityEstimator, self).__init__()

        self.max_K = max_K
        self.dim_model = dim_model

        kernel_sizes = [
            conv_1_kernel_size,
            conv_2_kernel_size,
            conv_3_kernel_size,
            conv_4_kernel_size,
        ]
        if any(kernel_size % 2 == 0 for kernel_size in kernel_sizes):
            raise ValueError(f"Conv kernel sizes must be odd, got {kernel_sizes}")
        if self.dim_model % transformer_nhead != 0:
            raise ValueError(
                f"dim_model ({self.dim_model}) must be divisible by "
                f"transformer_nhead ({transformer_nhead})"
            )

        conv_1_channels = 32 * channel_multiplier
        conv_2_channels = 64 * channel_multiplier
        conv_3_channels = 128 * channel_multiplier
        transformer_dim_feedforward = int(self.dim_model * transformer_ff_multiplier)

        self.conv_encoder = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv1d(
                input_channels,
                conv_1_channels,
                kernel_size=conv_1_kernel_size,
                stride=1,
                padding=conv_1_kernel_size // 2,
            )),
            ('norm_1', nn.BatchNorm1d(conv_1_channels)),
            ('gelu_1', nn.GELU()),

            ('conv_2', nn.Conv1d(
                conv_1_channels,
                conv_2_channels,
                kernel_size=conv_2_kernel_size,
                stride=conv_2_stride,
                padding=conv_2_kernel_size // 2,
            )),
            ('norm_2', nn.BatchNorm1d(conv_2_channels)),
            ('gelu_2', nn.GELU()),

            ('conv_3', nn.Conv1d(
                conv_2_channels,
                conv_3_channels,
                kernel_size=conv_3_kernel_size,
                stride=1,
                padding=conv_3_kernel_size // 2,
            )),
            ('norm_3', nn.BatchNorm1d(conv_3_channels)),
            ('gelu_3', nn.GELU()),

            ('conv_4', nn.Conv1d(
                conv_3_channels,
                self.dim_model,
                kernel_size=conv_4_kernel_size,
                stride=1,
                padding=conv_4_kernel_size // 2,
            )),
            ('norm_4', nn.BatchNorm1d(self.dim_model)),
            ('gelu_4', nn.GELU()),
        ]))

        self.pos_encoder = PosEnc(dim_model=self.dim_model, max_len=10_000)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=transformer_num_layers,
            enable_nested_tensor=False,
        )

        self.pooling = PoolingConcat(dim_model=self.dim_model, attn_dim=self.dim_model)

        self.classifier = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=self.dim_model * 3, out_features=self.dim_model * 2)),
            ('gelu_1', nn.GELU()),
            ('dropout', nn.Dropout(0.1)),
        ]))

        if learning_strategy == "mse":
            self.classifier.add_module('output', nn.Linear(in_features=2*self.dim_model, out_features=1))
        elif learning_strategy == "cross_entropy":
            self.classifier.add_module('output', nn.Linear(in_features=2*self.dim_model, out_features=self.max_K))
        elif learning_strategy == "ordinal":
            self.classifier.add_module('output', nn.Linear(in_features=2*self.dim_model, out_features=self.max_K - 1))
        elif learning_strategy == "mse+ce":
            self.classifier.add_module('output', DualHeadOutput(in_features=2*self.dim_model, max_K=self.max_K))
        elif learning_strategy == "mse+ce+or":
            self.classifier.add_module('output', TripleHeadOutput(in_features=2*self.dim_model, max_K=self.max_K))
        else:
            raise ValueError(f"Unknown learning strategy: {learning_strategy}")

    def forward(self, x):
        out = self.conv_encoder(x)          # (batch_size, 128, 64)
        out = out.permute(0, 2, 1)          # (batch_size, 64, 128)
        out = self.pos_encoder(out)         # (batch_size, 64, 128)

        out = self.transformer_encoder(out) # (batch_size, 64, 128)
        out = self.pooling(out)             # (batch_size, 384)

        out = self.classifier(out)          # (batch_size, out_features)
        return out
        
if __name__ == "__main__":
    x = torch.randn(16, 4, 128)
    print(f"Input shape: {x.shape}")
    learning_strategy = "mse+ce+or"
    CE = CardinalityEstimator(learning_strategy=learning_strategy, max_K=10, input_channels=4)
    nb_params = sum(p.numel() for p in CE.parameters())
    print(f"Number of parameters: {nb_params}")
    if learning_strategy == "mse+ce":
        out_mse, out_ce = CE(x)
        print(f"MSE output shape: {out_mse.shape}")
        print(f"CE output shape: {out_ce.shape}")
    else:
        out = CE(x)
        print(f"Output shape: {out.shape}")
