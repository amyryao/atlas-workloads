import json
import torch
from pathlib import Path
from model import TinyMLP

torch.manual_seed(42)

out_dir = Path("mlp/artifacts")
out_dir.mkdir(parents=True, exist_ok=True)

model = TinyMLP().bfloat16()
model.eval()

x = torch.randn(4, 64).bfloat16()

with torch.no_grad():
    fc1_out = model.fc1(x)
    gelu_out = model.gelu(fc1_out)
    y_ref = model.fc2(gelu_out)

torch.save(model.state_dict(), out_dir / "weights.pt")
torch.save(x, out_dir / "inputs.pt")
torch.save(y_ref, out_dir / "reference_outputs.pt")

torch.save(
    {
        "fc1_out": fc1_out,
        "gelu_out": gelu_out,
        "fc2_out": y_ref,
    },
    out_dir / "intermediates.pt",
)

metadata = {
    "name": "tiny_mlp",
    "input_shape": [4, 64],
    "hidden_size": 128,
    "output_shape": [4, 64],
    "dtype": "bfloat16",
    "activation": "GELU",
}

with open(out_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("saved to", out_dir)