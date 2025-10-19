import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=3, attn_drop=0.0, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # x: (B, N, C)
        B, N, C = x.shape

        # TODO 1: compute qkv = self.qkv(x) and reshape to (B, N, 3, H, D)
        # H = self.num_heads, D = self.head_dim, with C = H*D
        # qkv = ...
        # q, k, v = qkv.unbind(dim=2)  # each (B, N, H, D)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # TODO 2: move heads before tokens -> (B, H, N, D)
        # q = ...
        # k = ...
        # v = ...
        q = q.transpose(1, 2)  
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)

        # TODO 3: attention scores = (q @ k^T) * self.scale -> (B, H, N, N)
        # attn = ...
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # TODO 4: softmax over last dim and dropout
        # attn = ...
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # TODO 5: attention-weighted sum with v -> (B, H, N, D)
        # out = ...
        out = attn @ v

        # TODO 6: merge heads back -> (B, N, C)
        # out = out.transpose(1, 2).reshape(B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)

        # TODO 7: final projection and dropout
        # out = ...
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: (B, C, H, W)
        x = self.proj(x)                # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):              # (B, N, C)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ---------- ViT for CIFAR-10 ----------

class ViTForCIFAR10(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        drop=0.1,
        attn_drop=0.0,
        use_cls_token=True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_tokens = self.num_patches + 1
        else:
            self.cls_token = None
            pos_tokens = self.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, pos_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, 3, 32, 32)
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N, C)

        if self.use_cls_token:
            cls_tok = self.cls_token.expand(B, -1, -1)       # (B, 1, C)
            x = torch.cat([cls_tok, x], dim=1)               # (B, 1+N, C)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.use_cls_token:
            x = x[:, 0]              # CLS token
        else:
            x = x.mean(dim=1)        # mean pooling over patches

        logits = self.head(x)        # (B, 10)
        return logits