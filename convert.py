import torch
from torchvision import models

# Load your model
model = models.efficientnet_b0()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("tree_classifier_efficientnet.pth", map_location="cpu"))
model.eval()

# Convert to TorchScript
example = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example)

# Save
traced_model.save("tree_classifier.pt")