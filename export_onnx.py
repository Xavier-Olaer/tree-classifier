import torch
from torchvision import models

# Load model
model = models.efficientnet_b0()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load("tree_classifier_efficientnet.pth", map_location="cpu"))
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("✅ ONNX model exported!")