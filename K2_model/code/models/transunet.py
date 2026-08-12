import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SimpleViTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class TransUNet(nn.Module):
    """
    维度完美对齐的精简版 TransUNet
    4次下采样 -> ViT瓶颈层 (16x16) -> 4次上采样
    """
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super().__init__()
        # ================= 1. CNN Encoder =================
        self.enc1 = ConvBlock(n_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        # ================= 2. ViT Bottleneck =================
        # 经过 4 次池化 (2^4 = 16), 256 / 16 = 16
        self.feat_size = img_size // 16 
        self.seq_len = self.feat_size ** 2   # 16 * 16 = 256
        self.embed_dim = 512
        
        # 位置编码与 Transformer
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.vit = nn.Sequential(*[SimpleViTBlock(self.embed_dim) for _ in range(4)])
        
        # ================= 3. CNN Decoder =================
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        # 拼接: 512(来自up4) + 512(来自enc4) = 1024 -> 映射回256
        self.dec4 = ConvBlock(1024, 256) 
        
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        # 拼接: 256(来自up3) + 256(来自enc3) = 512 -> 映射回128
        self.dec3 = ConvBlock(512, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        # 拼接: 128(来自up2) + 128(来自enc2) = 256 -> 映射回64
        self.dec2 = ConvBlock(256, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # 拼接: 64(来自up1) + 64(来自enc1) = 128 -> 映射回64
        self.dec1 = ConvBlock(128, 64) 
        
        # ================= 4. 输出层 =================
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        c1 = self.enc1(x)                                 # [B, 64, 256, 256]
        c2 = self.enc2(F.max_pool2d(c1, 2))               # [B, 128, 128, 128]
        c3 = self.enc3(F.max_pool2d(c2, 2))               # [B, 256, 64, 64]
        c4 = self.enc4(F.max_pool2d(c3, 2))               # [B, 512, 32, 32]
        
        # 额外增加一次下采样送入 ViT
        bottleneck_feat = F.max_pool2d(c4, 2)             # [B, 512, 16, 16]
        
        # --- ViT Bottleneck ---
        B, C, H, W = bottleneck_feat.shape
        x_flat = bottleneck_feat.flatten(2).transpose(1, 2) # [B, 256, 512]
        x_vit = self.vit(x_flat + self.pos_embed)           # 融合位置编码并Self-Attention
        x_bottleneck = x_vit.transpose(1, 2).view(B, C, H, W) # 还原为 [B, 512, 16, 16]
        
        # --- Decoder ---
        d4 = self.dec4(torch.cat([self.up4(x_bottleneck), c4], dim=1)) # -> [B, 256, 32, 32]
        d3 = self.dec3(torch.cat([self.up3(d4), c3], dim=1))           # -> [B, 128, 64, 64]
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))           # -> [B, 64, 128, 128]
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))           # -> [B, 64, 256, 256]
        
        return self.out(d1)