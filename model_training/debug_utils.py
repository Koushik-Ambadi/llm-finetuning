# model_training/debug_utils.py

import os
import json
import torch

def save_checkpoint(data, name, step, output_dir):
    """Save debug data to debug_checkpoints/ as JSON, text or PT."""
    debug_dir = os.path.join(output_dir, "debug_checkpoints")
    os.makedirs(debug_dir, exist_ok=True)

    filename = f"step{step:02d}_{name}"
    path = os.path.join(debug_dir, filename)

    if isinstance(data, dict):
        json_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                json_data[k] = v.tolist()
            else:
                json_data[k] = v
        with open(path + ".json", "w") as f:
            json.dump(json_data, f, indent=2)
    elif isinstance(data, torch.Tensor):
        torch.save(data, path + ".pt")
    else:
        with open(path + ".txt", "w") as f:
            f.write(str(data))
