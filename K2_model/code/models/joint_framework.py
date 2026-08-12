import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


def _conv_block(in_channels, out_channels, kernel_size=3, padding=1, norm_type="bn"):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        _norm_layer(out_channels, norm_type),
        nn.LeakyReLU(0.2, inplace=True),
    )

class Enhancer(nn.Module):
    """
    图像增强网络(Student) - 带残差连接
    """
    def __init__(self, in_channels=3, out_channels=3, norm_type="bn"):
        super(Enhancer, self).__init__()
        self.conv1 = _conv_block(in_channels, 16, kernel_size=3, padding=1, norm_type=norm_type)
        self.conv2 = _conv_block(16, 16, kernel_size=3, padding=1, norm_type=norm_type)
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
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=16, norm_type="bn"):
        super(MultiScaleEnhancer, self).__init__()
        self.norm_type = norm_type
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
            _norm_layer(hidden_channels, norm_type),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _norm_layer(hidden_channels, norm_type),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def _conv_block(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            ),
            _norm_layer(out_channels, self.norm_type),
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


class AnisotropicEnhancer(nn.Module):
    """Residual enhancer with strip kernels for elongated capillary structures."""

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=16, norm_type="bn"):
        super(AnisotropicEnhancer, self).__init__()
        self.stem = _conv_block(in_channels, hidden_channels, kernel_size=3, padding=1, norm_type=norm_type)
        self.local_branch = _conv_block(hidden_channels, hidden_channels, kernel_size=3, padding=1, norm_type=norm_type)
        self.strip7 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 7), padding=(0, 3), bias=False),
            _norm_layer(hidden_channels, norm_type),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(7, 1), padding=(3, 0), bias=False),
            _norm_layer(hidden_channels, norm_type),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.strip21 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 21), padding=(0, 10), bias=False),
            _norm_layer(hidden_channels, norm_type),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(21, 1), padding=(10, 0), bias=False),
            _norm_layer(hidden_channels, norm_type),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1),
            _norm_layer(hidden_channels, norm_type),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _norm_layer(hidden_channels, norm_type),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.stem(x)
        fused = self.fuse(torch.cat([self.local_branch(feat), self.strip7(feat), self.strip21(feat)], dim=1))
        residual = self.out_conv(fused)
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


class JointModel_DecoderDistill(nn.Module):
    """Joint model that exposes student/teacher decoder features for distillation."""

    def __init__(self, enhancer, segmentor):
        super(JointModel_DecoderDistill, self).__init__()
        self.enhancer = enhancer
        self.segmentor = segmentor

    def _teacher_features(self, teacher_img):
        was_training = self.segmentor.training
        try:
            self.segmentor.eval()
            with torch.no_grad():
                _, teacher_features = self.segmentor(teacher_img, return_decoder_features=True)
                teacher_features = [feat.detach() for feat in teacher_features]
        finally:
            if was_training:
                self.segmentor.train()
        return teacher_features

    def forward(self, x, teacher_img=None):
        enhanced_img = self.enhancer(x)
        if teacher_img is None:
            seg_out = self.segmentor(enhanced_img)
            return seg_out, enhanced_img, None
        seg_out, student_features = self.segmentor(enhanced_img, return_decoder_features=True)
        feature_pair = None
        feature_pair = (student_features, self._teacher_features(teacher_img))
        return seg_out, enhanced_img, feature_pair


class JointModel_DecoderDistillV2(nn.Module):
    """Frozen green-prior teacher with projected decoder feature distillation."""

    def __init__(self, enhancer, segmentor, teacher_segmentor):
        super(JointModel_DecoderDistillV2, self).__init__()
        self.enhancer = enhancer
        self.segmentor = segmentor
        self.teacher_segmentor = teacher_segmentor

        for param in self.teacher_segmentor.parameters():
            param.requires_grad = False
        self.teacher_segmentor.eval()

        decoder_channels = self._decoder_channels(segmentor)
        self.feature_projs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=1) for channels in decoder_channels]
        )

    @staticmethod
    def _decoder_channels(segmentor):
        config = getattr(getattr(segmentor, "model", None), "config", None)
        channels = getattr(config, "decoder_channels", None)
        if channels is None and isinstance(config, dict):
            channels = config.get("decoder_channels")
        return list(channels) if channels is not None else [256, 128, 64, 16]

    def _teacher_outputs(self, teacher_img):
        self.teacher_segmentor.eval()
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher_segmentor(
                teacher_img,
                return_decoder_features=True,
            )
        return teacher_logits.detach(), [feat.detach() for feat in teacher_features]

    def forward(self, x, teacher_img=None):
        enhanced_img = self.enhancer(x)
        if teacher_img is None:
            seg_out = self.segmentor(enhanced_img)
            return seg_out, enhanced_img, None

        seg_out, student_features = self.segmentor(enhanced_img, return_decoder_features=True)
        projected_features = [
            proj(feature) for proj, feature in zip(self.feature_projs, student_features)
        ]
        teacher_logits, teacher_features = self._teacher_outputs(teacher_img)
        feature_payload = {
            "student_features": projected_features,
            "teacher_features": teacher_features,
            "student_logits": seg_out,
            "teacher_logits": teacher_logits,
        }
        return seg_out, enhanced_img, feature_payload


class JointModel_DualFusion(nn.Module):
    """Light CNN + TransUNet fusion probe inspired by parallel CNN/Transformer designs."""

    def __init__(self, enhancer, segmentor, local_channels=32, norm_type="bn"):
        super(JointModel_DualFusion, self).__init__()
        self.enhancer = enhancer
        self.segmentor = segmentor
        decoder_channels = getattr(segmentor, "decoder_out_channels", 16)
        self.local_branch = nn.Sequential(
            _conv_block(3, 16, kernel_size=3, padding=1, norm_type=norm_type),
            _conv_block(16, local_channels, kernel_size=3, padding=1, norm_type=norm_type),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels + local_channels, decoder_channels, kernel_size=3, padding=1),
            _norm_layer(decoder_channels, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            _norm_layer(decoder_channels, norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enhanced_img = self.enhancer(x)
        _, decoder_feat = self.segmentor(enhanced_img, return_decoder_output=True)
        local_feat = self.local_branch(x)
        if local_feat.shape[-2:] != decoder_feat.shape[-2:]:
            local_feat = F.interpolate(local_feat, size=decoder_feat.shape[-2:], mode="bilinear", align_corners=False)
        refined_feat = decoder_feat + self.fuse(torch.cat([decoder_feat, local_feat], dim=1))
        seg_out = self.segmentor.segment_from_decoder_output(refined_feat)
        return seg_out, enhanced_img, refined_feat


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
