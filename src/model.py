import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, hidden_dim: int, image_size: int):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c")
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(
            (image_size // patch_size) ** 2 + 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.proj(x)
        cls_token = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding
        return x

class MLPBLock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int, dropout: float):
        super(MLPBLock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, num_head: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float):
        super(EncoderBlock, self).__init__()

        self.num_head = num_head
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_head, dropout=attention_dropout, batch_first=True)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBLock(hidden_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.layernorm1(x)
        x = x + self.attention(qkv, qkv, qkv)[0]
        x = x + self.mlp(self.layernorm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, seq_len: int, num_layers: int, num_head: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        layers = [ EncoderBlock(num_head, hidden_dim, mlp_dim, dropout, attention_dropout) for _ in range(num_layers)]
        self.layers = nn.Sequential(*layers)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(self.layers(self.dropout(x)))
        return x

class ViT(nn.Module):
    def __init__(self, num_head: int,
                 num_layers: int,
                 num_classes: int,
                 image_size: int,
                 patch_size: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 dropout: float,
                 attention_dropout: float,
                 ):
        super(ViT, self).__init__()

        assert image_size % patch_size == 0, "image size must be divisible by patch size"

        self.num_head = num_head
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        self.patch_embedding = PatchEmbedding(
            3, patch_size, hidden_dim, image_size)

        seq_len = (image_size // patch_size) ** 2 + 1

        self.encoder = Encoder(
            seq_len,
            num_layers,
            num_head,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
        )
        self.seq_len = seq_len

        self.mlp_head = nn.Linear(hidden_dim, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        
        x = self.encoder(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = ViT(num_head=12, num_layers=12, num_classes=10, image_size=224,
             patch_size=16, hidden_dim=768, mlp_dim=3072, dropout=0.1, attention_dropout=0.1)
    print(model(x).shape)
