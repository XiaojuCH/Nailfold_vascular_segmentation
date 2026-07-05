import torch
import torch.nn as nn

class Enhancer(nn.Module):
    """
    图像增强网络(Student) - 带残差连接
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(Enhancer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        # 带残差连接：输入加上变化量，并限制在 [0, 1] 范围内
        out = torch.clamp(x + residual, min=0.0, max=1.0)
        return out


class MultiScaleEnhancer(nn.Module):
    """
    Lightweight student enhancer with parallel receptive fields for vessels
    at different widths and local shapes.
    """
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=16):
        super(MultiScaleEnhancer, self).__init__()
        self.stem = self._conv_block(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.branch_3x3 = self._conv_block(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.branch_5x5 = self._conv_block(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.branch_dilated = self._conv_block(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size, padding, dilation=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        feat = self.stem(x)
        multi_scale_feat = torch.cat(
            [self.branch_3x3(feat), self.branch_5x5(feat), self.branch_dilated(feat)],
            dim=1
        )
        residual = self.out_conv(self.fuse(multi_scale_feat))
        return torch.clamp(x + residual, min=0.0, max=1.0)


class JointModel(nn.Module):
    """
    绔埌绔仈鍚堣缁冩鏋讹細Enhancer + Segmentor
    """
    def __init__(self, enhancer, segmentor):
        super(JointModel, self).__init__()
        self.enhancer = enhancer
        self.segmentor = segmentor

    def forward(self, x):
        # 1. 鍘熷鍥惧儚閫氳繃澧炲己鍣紝鐢熸垚楂樺姣斿害鐗瑰緛鍥?
        enhanced_img = self.enhancer(x)
        
        # 2. 澧炲己鍚庣殑鍥捐緭鍏ョ粰涓嬫父鍒嗗壊鍣?
        seg_out = self.segmentor(enhanced_img)
        
        # 璁粌鏃舵垜浠渶瑕?enhanced_img 鏉ュ拰 Teacher 绠?Loss锛屾帹鐞嗘椂鍏跺疄鍙敤 seg_out
        return seg_out, enhanced_img


class JointModel_V2(nn.Module):
    """
    鍗囩骇鐗堣仈鍚堟鏋讹細铻嶅叆缁块€氶亾鍏堥獙鐨勭┖闂存敞鎰忓姏鏈哄埗 (Spatial Attention Gate)
    """
    def __init__(self, enhancer, segmentor, attention_mode="normal"):
        super(JointModel_V2, self).__init__()
        self.enhancer = enhancer
        self.segmentor = segmentor
        self.attention_mode = attention_mode

    def forward(self, x):
        # 1. 鐗╃悊鍏堥獙澧炲己鍒嗘敮 (鎻愬彇绫讳技缁块€氶亾鐨勮绠￠珮浜壒寰?
        enhanced_feat = self.enhancer(x)
        
        # 2. 鐢熸垚绌洪棿娉ㄦ剰鍔涙帺鑶?(Spatial Attention Mask)
        # 鍥犱负 enhanced_feat 鏄?3 閫氶亾锛屾垜浠寜閫氶亾姹傚钩鍧囧彉鎴愬崟閫氶亾鐨勬鐜囧浘锛?
        # 鐒跺悗缁忚繃 sigmoid 纭繚鍊煎煙鍦?(0, 1) 涔嬮棿銆?
        # 瓒婃帴杩?1 浠ｈ〃瓒婂彲鑳芥槸琛€绠★紝瓒婃帴杩?0 浠ｈ〃鏄儗鏅毊鑲ゃ€?
        attention_mask = torch.sigmoid(enhanced_feat.mean(dim=1, keepdim=True))
        if self.attention_mode == "inverse":
            attention_mask = 1.0 - attention_mask
        elif self.attention_mode != "normal":
            raise ValueError(f"Unknown attention_mode: {self.attention_mode}")
        
        # 3. 鐗╃悊鍏堥獙铻嶅悎 (娈嬪樊娉ㄦ剰鍔涘彔鍔?
        # X_fused = X + X * Mask
        # 杩欐牱鏃㈣繃婊や簡鑳屾櫙鍣０锛圶 * Mask锛夛紝鍙堜繚鐣欎簡鍘熷浘鐨勫熀纭€淇℃伅缁撴瀯锛? X锛?
        x_fused = x + x * attention_mask
        
        # 4. 閫佸叆涓嬫父鐨?TransUNet 杩涜鐗瑰緛鎻愬彇鍜屽垎鍓?
        seg_out = self.segmentor(x_fused)
        
        # 杩斿洖 seg_out 鐢ㄤ簬璁＄畻 Dice Loss
        # 杩斿洖 enhanced_feat 鐢ㄤ簬璁＄畻 MSE 鍜?Gradient Loss (鍚戠豢閫氶亾钂搁)
        # 棰濆杩斿洖 attention_mask 鏂逛究浠ュ悗鎴戜滑鍋氬彲瑙嗗寲璇佹槑
        return seg_out, enhanced_feat, attention_mask


class JointModel_Gated(nn.Module):
    """
    鑷€傚簲闂ㄦ帶铻嶅悎妗嗘灦锛氬湪鍘熷浘涓庡寮哄浘涔嬮棿瀛︿範铻嶅悎鏉冮噸銆?    """
    def __init__(self, enhancer, segmentor):
        super(JointModel_Gated, self).__init__()
        self.enhancer = enhancer
        self.segmentor = segmentor
        self.gate = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enhanced_img = self.enhancer(x)
        fusion_gate = self.gate(torch.cat([x, enhanced_img], dim=1))
        x_fused = fusion_gate * enhanced_img + (1.0 - fusion_gate) * x
        seg_out = self.segmentor(x_fused)
        return seg_out, enhanced_img, fusion_gate

class JointModel_BoundaryRefine(nn.Module):
    """Enhancer + segmentor with a lightweight boundary refinement head.

    The segmentor produces coarse vessel logits. A small refinement head predicts
    boundary logits from the enhanced image and coarse probability map, then feeds
    boundary evidence back into the final segmentation logits.
    """

    def __init__(self, enhancer, segmentor, hidden_channels=16):
        super(JointModel_BoundaryRefine, self).__init__()
        self.enhancer = enhancer
        self.segmentor = segmentor
        self.boundary_head = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(5, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        enhanced_img = self.enhancer(x)
        coarse_logits = self.segmentor(enhanced_img)
        coarse_prob = torch.sigmoid(coarse_logits)
        boundary_logits = self.boundary_head(torch.cat([enhanced_img, coarse_prob], dim=1))
        residual_logits = self.refine_head(torch.cat([enhanced_img, coarse_prob, torch.sigmoid(boundary_logits)], dim=1))
        seg_out = coarse_logits + residual_logits
        return seg_out, enhanced_img, boundary_logits
