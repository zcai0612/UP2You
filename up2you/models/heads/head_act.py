import torch
import torch.nn.functional as F

def base_shape_act(shape_enc, act_type="linear"):
    """
    Apply basic activation function to shape parameters.

    Args:
        shape_enc: Tensor containing encoded shape parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated shape parameters
    """
    if act_type == "linear":
        return shape_enc
    elif act_type == "inv_log":
        return inverse_log_transform(shape_enc)
    elif act_type == "exp":
        return torch.exp(shape_enc)
    elif act_type == "relu":
        return F.relu(shape_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")
    

def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))

