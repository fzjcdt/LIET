import torch
import torch.nn as nn


def count_flops_custom(model, input_size, device):
    flops = 0

    def hook_conv(module, input, output):
        nonlocal flops
        # calculate FLOPs for convolutional layer
        batch_size, in_channels, in_h, in_w = input[0].size()
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        bias_ops = 1 if module.bias is not None else 0

        flops += batch_size * output.size(1) * output.size(2) * output.size(3) * (kernel_ops + bias_ops)

    def hook_linear(module, input, output):
        nonlocal flops
        # calculate FLOPs for linear layer
        flops += input[0].size(0) * input[0].size(1) * output.size(1)

    # register hooks
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.register_forward_hook(hook_conv)
        elif isinstance(module, nn.Linear):
            module.register_forward_hook(hook_linear)
    # forward pass
    with torch.no_grad():
        input_tensor = torch.randn(input_size).to(device)
        model(input_tensor)
    return flops


def estimate_total_flops(model, input_size, forward_passes=1, backward_passes=1, device='cuda:0'):
    # calculate FLOPs for forward pass
    forward_flops = count_flops_custom(model, input_size, device=device)

    # the number of FLOPs for backward pass is approximately equal to the number of FLOPs for forward pass
    backward_flops = forward_flops

    # total FLOPs
    total_flops = (forward_passes * forward_flops) + (backward_passes * backward_flops)

    return total_flops
