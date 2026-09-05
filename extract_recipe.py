#!/usr/bin/env python3
"""Extract per-tensor bitrate recipe from quantization_config.json."""
import json
import yaml
from pathlib import Path

def extract_recipe(quant_config_path, output_path, model_config_path = "config.json"):
    with open(quant_config_path) as f:
        config = json.load(f)

    # The mirror's standalone quantization_config.json omits vision_bits even
    # though the vision tower really is quantized (V3). config.json carries the
    # authoritative quantization_config block, so prefer it when present.
    try:
        with open(model_config_path) as f:
            nested = json.load(f).get("quantization_config", {})
        for k in ("vision_bits", "mtp_bits", "head_bits", "codebook", "out_scales"):
            if k in nested:
                config[k] = nested[k]
    except FileNotFoundError:
        pass
    
    recipe = {
        "target_bpw": config["bits"],
        "achieved_bpw": config["bits"],
        "head_bits": config["head_bits"],
        "mtp_bits": config.get("mtp_bits", 4),
        "vision_bits": config.get("vision_bits", 16),
        "codebook": config.get("codebook", "mul1"),
        "out_scales": config.get("out_scales", "always"),
        "calibration": config.get("calibration", {"rows": 250, "cols": 2048}),
        "tensors": {}
    }
    
    # Extract per-tensor bits from tensor_storage
    tensor_storage = config.get("tensor_storage", {})
    for tensor_name, tensor_info in tensor_storage.items():
        # Skip non-quantized tensors (norms, embeddings, etc.)
        if "quant_format" not in tensor_info:
            continue
        
        # Get the bits_per_weight
        bpw = tensor_info.get("bits_per_weight")
        if bpw is not None:
            # Keys must match exllamav3's module keys exactly: convert.py aborts
            # on any unknown/missing key. For qwen3_5 the arch declares
            # key_prefix = "model.language_model" (exllamav3/architecture/qwen3_5.py),
            # so the checkpoint name IS the module key -- do not rewrite it.
            recipe["tensors"][tensor_name] = int(bpw)
    
    with open(output_path, "w") as f:
        yaml.dump(recipe, f, default_flow_style=False, sort_keys=False)
    
    print(f"Recipe extracted to {output_path}")
    print(f"Total tensors: {len(recipe['tensors'])}")
    
    # Print distribution
    from collections import Counter
    dist = Counter(recipe["tensors"].values())
    print("Bit distribution:")
    for bits, count in sorted(dist.items()):
        print(f"  {bits} bpw: {count} tensors")

if __name__ == "__main__":
    extract_recipe(
        "models/Qwen3.8-27B-EXL3-2.0bpw/quantization_config.json",
        "recipe_2.00bpw.yaml",
        model_config_path="models/Qwen3.8-27B-EXL3-2.0bpw/config.json",
    )
