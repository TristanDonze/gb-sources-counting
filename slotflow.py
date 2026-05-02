import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# -------------------------------
# Positional Encoding
# -------------------------------
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# -------------------------------
# Pooling: mean + max + attention
# -------------------------------
class PoolingConcat(nn.Module):
    def __init__(self, d_model, attn_dim=128):
        super().__init__()
        self.attn = nn.Linear(d_model, attn_dim)
        self.context_vec = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x):
        mean_pool = x.mean(dim=1)  # (B, C)
        max_pool, _ = x.max(dim=1)  # (B, C)

        a = torch.tanh(self.attn(x))  # (B, L, attn_dim)
        scores = self.context_vec(a).squeeze(-1)  # (B, L)
        attn_weights = F.softmax(scores, dim=-1)
        attn_pool = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

        return torch.cat([mean_pool, max_pool, attn_pool], dim=-1)


# -------------------------------
# SlotFlow
# -------------------------------
class SlotFlow(nn.Module):
    def __init__(
        self, hidden_dim=128, max_slots=3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_slots = max_slots

        def make_conv_encoder(in_channels=1):
            return nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=8, stride=2, padding=3),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
                nn.GELU(),
                nn.Conv1d(
                    64, 128, kernel_size=5, stride=1, padding=3
                ),  # ← was stride=2
                nn.GELU(),
            ) 

        # Classifier keeps long-only attention
        self.global_attn_long = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )

        self.pos = PosEnc(128)

        # --- Encoders ---
        self.encoder_conv_long = make_conv_encoder(
            in_channels=2
        )  # long: freq-domain (Re/Im) 

        # Classifier projection
        self.pool_long = PoolingConcat(128)
        self.encoder_fc_long = nn.Linear(3 * 128, 2 * hidden_dim)
        self.k_classifier = nn.Linear(2 * hidden_dim, max_slots)

    def forward(self, x_long, x_short, use_gt_k=None, k_prior=None):
        B, L = x_long.shape

        # --- FFT input (long) as Re/Im channels ---
        fft_long = torch.fft.rfft(x_long, dim=-1, norm="ortho")
        fft_long = torch.view_as_real(fft_long).permute(0, 2, 1)  # (B, 2, Lr)

        # --- Encode long (freq-domain) ---
        h_long = self.encoder_conv_long(fft_long)  # (B, C, L') for slotflow sample the shape is (batch_size, 128, 377)
        h_long = h_long.permute(0, 2, 1)  # (B, L', C)
        h_long = self.pos(h_long)

        # =========================
        # CLASSIFIER EMBEDDING (long-only attention)
        # =========================
        h_long_attn_cls, w_long_cls = self.global_attn_long(h_long, h_long, h_long)
        h_long_pool = self.pool_long(h_long_attn_cls)  # (B, 3*128)
        h_embed_long = self.encoder_fc_long(h_long_pool)  # (B, 2*hidden_dim)
        k_logits = self.k_classifier(h_embed_long)

        # if k_prior is not None:
        #     # allow tensor or np array; normalize and move to device if needed
        #     if not torch.is_tensor(k_prior):
        #         k_prior = torch.tensor(
        #             k_prior, dtype=k_logits.dtype, device=k_logits.device
        #         )
        #     else:
        #         k_prior = k_prior.to(k_logits.device, dtype=k_logits.dtype)
        #     k_prior = k_prior / (k_prior.sum() + 1e-8)
        #     log_prior = torch.log(k_prior + 1e-8)
        #     k_logits = k_logits + log_prior

        k_pred_int = (
            torch.argmax(F.softmax(k_logits, dim=-1), dim=-1) + 1
        )  # in {1..max_slots}

        
        return {
            "K_logits": k_logits,
            "h_embed_long": h_embed_long,
        }