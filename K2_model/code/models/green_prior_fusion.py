import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PRIOR_FUSION_VARIANTS = (
    "plain_single",
    "directional_single",
    "directional_multiscale",
)


class _ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.SiLU(inplace=True),
        )


class PlainGreenStem(nn.Module):
    """A local green-channel feature extractor without directional kernels."""

    def __init__(self, channels=16):
        super().__init__()
        self.layers = nn.Sequential(
            _ConvAct(1, channels),
            _ConvAct(channels, channels),
        )

    def forward(self, green):
        return self.layers(green)

    def diagnostics(self):
        return {}


class _DirectionalBranch(nn.Sequential):
    def __init__(self, channels, kernel_size, padding):
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(inplace=True),
        )


class DirectionalGreenStem(nn.Module):
    """Parallel horizontal/vertical strip filters with per-pixel branch selection."""

    def __init__(self, channels=16):
        super().__init__()
        self.stem = _ConvAct(1, channels)
        self.branches = nn.ModuleList(
            [
                _DirectionalBranch(channels, kernel_size=(3, 3), padding=(1, 1)),
                _DirectionalBranch(channels, kernel_size=(1, 7), padding=(0, 3)),
                _DirectionalBranch(channels, kernel_size=(7, 1), padding=(3, 0)),
                _DirectionalBranch(channels, kernel_size=(1, 21), padding=(0, 10)),
                _DirectionalBranch(channels, kernel_size=(21, 1), padding=(10, 0)),
            ]
        )
        self.branch_gate = nn.Conv2d(channels * len(self.branches), len(self.branches), kernel_size=1)
        self.out = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.SiLU(inplace=True))
        nn.init.zeros_(self.branch_gate.weight)
        nn.init.zeros_(self.branch_gate.bias)
        self._last_branch_weights = None

    def forward(self, green):
        base = self.stem(green)
        branch_features = [branch(base) for branch in self.branches]
        branch_weights = torch.softmax(self.branch_gate(torch.cat(branch_features, dim=1)), dim=1)
        fused = sum(
            branch_weights[:, index : index + 1] * feature
            for index, feature in enumerate(branch_features)
        )
        self._last_branch_weights = branch_weights.detach().mean(dim=(0, 2, 3)).cpu()
        return base + self.out(fused)

    def diagnostics(self):
        if self._last_branch_weights is None:
            return {}
        names = ("local3", "horizontal7", "vertical7", "horizontal21", "vertical21")
        return {
            f"direction_weight_{name}": float(value)
            for name, value in zip(names, self._last_branch_weights.tolist())
        }


class PriorFusionBlock(nn.Module):
    """Identity-initialized gated residual fusion for one decoder scale."""

    def __init__(self, decoder_channels, prior_channels):
        super().__init__()
        self.prior_projection = nn.Conv2d(prior_channels, decoder_channels, kernel_size=1)
        merged_channels = decoder_channels * 2
        self.gate = nn.Conv2d(merged_channels, 1, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(merged_channels, decoder_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.zeros(()))
        self._last_gate_mean = None

    def forward(self, decoder_feature, prior_feature):
        if prior_feature.shape[-2:] != decoder_feature.shape[-2:]:
            prior_feature = F.interpolate(
                prior_feature,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        prior_feature = self.prior_projection(prior_feature)
        merged = torch.cat([decoder_feature, prior_feature], dim=1)
        gate = torch.sigmoid(self.gate(merged))
        self._last_gate_mean = gate.detach().mean().cpu()
        return decoder_feature + torch.tanh(self.alpha) * gate * self.refine(merged)

    def diagnostics(self):
        result = {"alpha": float(torch.tanh(self.alpha.detach()).cpu())}
        if self._last_gate_mean is not None:
            result["gate_mean"] = float(self._last_gate_mean)
        return result


class GreenPriorFusionModel(nn.Module):
    """RGB TransUNet with a lightweight green-channel decoder prior branch."""

    def __init__(self, segmentor, variant="plain_single", prior_channels=16):
        super().__init__()
        if variant not in PRIOR_FUSION_VARIANTS:
            raise ValueError(f"Unknown prior fusion variant: {variant}")

        self.segmentor = segmentor
        self.variant = variant
        directional = variant != "plain_single"
        self.green_stem = (
            DirectionalGreenStem(prior_channels)
            if directional
            else PlainGreenStem(prior_channels)
        )

        self.downsample_half = None
        self.downsample_quarter = None
        if variant == "directional_multiscale":
            self.downsample_half = _ConvAct(prior_channels, 24, stride=2)
            self.downsample_quarter = _ConvAct(24, 32, stride=2)
            fusion_specs = {
                "1": (128, 32, "quarter"),
                "2": (64, 24, "half"),
                "3": (16, prior_channels, "full"),
            }
        else:
            fusion_specs = {"3": (16, prior_channels, "full")}

        self.fusions = nn.ModuleDict(
            {
                decoder_index: PriorFusionBlock(decoder_channels, branch_channels)
                for decoder_index, (decoder_channels, branch_channels, _) in fusion_specs.items()
            }
        )
        self._prior_scale_by_decoder = {
            int(decoder_index): scale
            for decoder_index, (_, _, scale) in fusion_specs.items()
        }

    def _green_prior_features(self, rgb):
        full = self.green_stem(rgb[:, 1:2])
        features = {"full": full}
        if self.downsample_half is not None:
            half = self.downsample_half(full)
            features["half"] = half
            features["quarter"] = self.downsample_quarter(half)
        return features

    def _decode_with_prior(self, rgb, prior_features):
        vit = self.segmentor.model
        hidden_states, _, skip_features = vit.transformer(rgb)
        batch_size, patch_count, hidden_channels = hidden_states.shape
        spatial_size = math.isqrt(patch_count)
        if spatial_size * spatial_size != patch_count:
            raise ValueError(f"Expected a square token grid, got {patch_count} tokens")

        decoded = hidden_states.permute(0, 2, 1).contiguous().view(
            batch_size,
            hidden_channels,
            spatial_size,
            spatial_size,
        )
        decoded = vit.decoder.conv_more(decoded)

        for decoder_index, decoder_block in enumerate(vit.decoder.blocks):
            if skip_features is not None and decoder_index < vit.decoder.config.n_skip:
                skip = skip_features[decoder_index]
            else:
                skip = None
            decoded = decoder_block(decoded, skip=skip)
            key = str(decoder_index)
            if key in self.fusions:
                scale = self._prior_scale_by_decoder[decoder_index]
                decoded = self.fusions[key](decoded, prior_features[scale])

        return vit.segmentation_head(decoded)

    def forward(self, rgb):
        prior_features = self._green_prior_features(rgb)
        return self._decode_with_prior(rgb, prior_features)

    def fusion_diagnostics(self):
        result = self.green_stem.diagnostics()
        for decoder_index, fusion in self.fusions.items():
            for name, value in fusion.diagnostics().items():
                result[f"fusion_{decoder_index}_{name}"] = value
        return result
