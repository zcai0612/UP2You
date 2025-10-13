import torch
from safetensors.torch import load_file

def load_unstrict_safetensors(model, checkpoint_path):
    checkpoint = load_file(checkpoint_path)
    current_state_dict = model.state_dict()

    mismatched_keys = []
    for key, param in checkpoint.items():
        if key in current_state_dict:
            if param.shape != current_state_dict[key].shape:
                mismatched_keys.append(key)
                print(f"skip {key}, checkpoint shape: {param.shape}, model shape: {current_state_dict[key].shape}")

    for key in mismatched_keys:
        checkpoint.pop(key)

    model.load_state_dict(checkpoint, strict=False)

    return model