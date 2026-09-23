"""
Convert celeb_model.pth (PyTorch SimpleCNN weights) into celeb_model.onnx,
the format the website runs in the visitor's browser.

Run it from the repo root:
    pip install torch onnx onnxruntime
    python export_onnx.py

If you retrain and get a new celeb_model.pth, run this again and replace
docs/model/celeb_model.onnx with the new file.
"""

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

WEIGHTS = "celeb_model.pth"
OUT = "docs/model/celeb_model.onnx"
INPUT_SIZE = 112  # matches input_size in MLProjConfigs.yaml


class SimpleCNN(nn.Module):
    """Same architecture as the notebook, so the saved weights load unchanged."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def main():
    model = SimpleCNN()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()

    example = torch.rand(2, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model,
        (example,),
        OUT,
        input_names=["frames"],
        output_names=["logits"],
        dynamic_axes={"frames": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Check the ONNX file gives the same answers as PyTorch.
    session = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    batch = torch.rand(8, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        expected = model(batch).numpy()
    got = session.run(None, {"frames": batch.numpy()})[0]
    diff = float(np.max(np.abs(expected - got)))
    print(f"Wrote {OUT}. Largest difference from PyTorch: {diff:.2e}")
    assert diff < 1e-4, "ONNX output does not match PyTorch"


if __name__ == "__main__":
    main()
