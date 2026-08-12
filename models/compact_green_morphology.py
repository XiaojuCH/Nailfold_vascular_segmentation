"""Compact green-contrast prior adapter for final-decoder vessel refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactGreenMorphologyAdapter(nn.Module):
    """Inject a fixed local green-contrast prior once at the final decoder scale."""

    def __init__(self, segmentor, use_prior=True, use_auxiliary_heads=True, prior_channels=16):
        super().__init__()
        self.segmentor = segmentor
        self.use_prior = use_prior
        self.use_auxiliary_heads = use_auxiliary_heads
        decoder_channels = getattr(segmentor, "decoder_out_channels", 16)

        if use_prior:
            self.prior_adapter = nn.Sequential(
                nn.Conv2d(1, prior_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(prior_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(prior_channels, decoder_channels, kernel_size=1),
            )
            self.gate = nn.Conv2d(decoder_channels * 2, 1, kernel_size=1)
            # Start close to the baseline while keeping an active gradient path.
            self.prior_scale_logit = nn.Parameter(torch.tensor(-3.0))
        else:
            self.prior_adapter = None
            self.gate = None
            self.register_parameter("prior_scale_logit", None)

        if use_auxiliary_heads:
            self.boundary_head = self._auxiliary_head(decoder_channels)
            self.centerline_head = self._auxiliary_head(decoder_channels)
        else:
            self.boundary_head = None
            self.centerline_head = None

        self._last_gate_mean = None
        self._last_prior_scale = None
        sigma = 9.0
        radius = int(round(sigma * 3.0))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (coords / sigma).square())
        self.register_buffer("green_blur_kernel", kernel / kernel.sum())

    @staticmethod
    def _auxiliary_head(channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def green_local_contrast(self, rgb):
        """Return positive dark-vessel contrast relative to local green background."""
        green = rgb[:, 1:2]
        kernel_size = self.green_blur_kernel.numel()
        radius = kernel_size // 2
        kernel_1d = self.green_blur_kernel.to(dtype=green.dtype)
        kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
        kernel_y = kernel_1d.view(1, 1, kernel_size, 1)
        blurred = F.conv2d(F.pad(green, (radius, radius, 0, 0), mode="reflect"), kernel_x)
        blurred = F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode="reflect"), kernel_y)
        return F.relu(blurred - green)

    def forward(self, rgb):
        _, decoder_feature = self.segmentor(rgb, return_decoder_output=True)
        prior = None
        if self.use_prior:
            local_contrast = self.green_local_contrast(rgb)
            prior = self.prior_adapter(local_contrast)
            if prior.shape[-2:] != decoder_feature.shape[-2:]:
                prior = F.interpolate(
                    prior,
                    size=decoder_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            gate = torch.sigmoid(self.gate(torch.cat([decoder_feature, prior], dim=1)))
            prior_scale = torch.sigmoid(self.prior_scale_logit)
            decoder_feature = decoder_feature + prior_scale * gate * prior
            self._last_gate_mean = gate.detach().mean().cpu()
            self._last_prior_scale = prior_scale.detach().cpu()

        segmentation_logits = self.segmentor.segment_from_decoder_output(decoder_feature)
        boundary_logits = self.boundary_head(decoder_feature) if self.boundary_head is not None else None
        centerline_logits = self.centerline_head(decoder_feature) if self.centerline_head is not None else None
        return segmentation_logits, boundary_logits, centerline_logits

    def diagnostics(self):
        result = {}
        if self._last_gate_mean is not None:
            result["prior_gate_mean"] = float(self._last_gate_mean)
        if self._last_prior_scale is not None:
            result["prior_scale"] = float(self._last_prior_scale)
        return result
