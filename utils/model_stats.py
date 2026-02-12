import torch
from thop import profile


def get_model_stats(model, input_size=(1, 3, 256, 256)):
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    return flops / 1e9, params / 1e6  # GFLOPs, M
