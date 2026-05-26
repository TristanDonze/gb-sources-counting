import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=10_000):
        super(PosEnc, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)] # (batch_size, N', d_model)
    
class PoolingConcat(nn.Module):
    def __init__(self, d_model, attn_dim=128):
        super().__init__()
        self.attn = nn.Linear(d_model, attn_dim)
        self.context_vec = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x):
        mean_pool = x.mean(dim=1)  # (batch_size, d_model)
        max_pool, _ = x.max(dim=1)  # (batch_size, d_model)

        a = torch.tanh(self.attn(x))  # (B, N', attn_dim)
        scores = self.context_vec(a).squeeze(-1)  # (B, N')
        attn_weights = F.softmax(scores, dim=-1)
        attn_pool = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

        return torch.cat([mean_pool, max_pool, attn_pool], dim=-1)

class CardinalityEstimator(nn.Module):
    def __init__(self, learning_strategy="mse", max_K=10, input_channels=4, dim_model=256):
        super(CardinalityEstimator, self).__init__()

        self.max_K = max_K
        self.dim_model = dim_model

        self.conv_encoder = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=2, padding=3)),
            ('gelu_1', nn.GELU()),
            ('conv_2', nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=2, padding=3)),
            ('gelu_2', nn.GELU()),
            ('conv_3', nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=3)),
            ('gelu_3', nn.GELU()),
            ('conv_4', nn.Conv1d(in_channels=128, out_channels=dim_model, kernel_size=3, stride=1, padding=1)),
        ]))

        self.pos_encoder = PosEnc(d_model=dim_model, max_len=10_000)

        self.MHAttention = nn.MultiheadAttention(
            embed_dim=dim_model, num_heads=4, batch_first=True
        )
        # self.attn_norm = nn.LayerNorm(dim_model)
        # self.ffn = nn.Sequential(OrderedDict([
        #     ('fc_1', nn.Linear(in_features=dim_model, out_features=2*dim_model)),
        #     ('gelu_1', nn.GELU()),
        #     ('fc_2', nn.Linear(in_features=2*dim_model, out_features=dim_model)),
        # ]))
        # self.ffn_norm = nn.LayerNorm(dim_model)

        self.pooling = PoolingConcat(d_model=dim_model, attn_dim=dim_model)

        self.classifier = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=dim_model*3, out_features=2*dim_model)),
            ('gelu_1', nn.GELU()),
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
        out = self.conv_encoder(x) # (batch_size, 128, N')
        out = out.permute(0, 2, 1) # (batch_size, N', 128)
        out = self.pos_encoder(out) # (batch_size, N', 128)

        # attn_output, _ = self.MHAttention(out, out, out, need_weights=False) # (batch_size, N', 128)
        # out = self.attn_norm(out + attn_output)
        # out = self.ffn_norm(out + self.ffn(out))
        # out = self.pooling(out) # (batch_size, 128*3)

        attn_output, attn_output_weights = self.MHAttention(out, out, out) # (batch_size, N', 128)
        out = self.pooling(attn_output)
        out = self.classifier(out) # (batch_size, out_features)
        return out
    
if __name__ == "__main__":
    x = torch.randn(1, 4, 128)
    print(x.shape)
    CE = CardinalityEstimator(learning_strategy="ordinal", max_K=10, input_channels=4)
    out = CE(x)
