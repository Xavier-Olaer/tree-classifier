import os
import torch
from torchvision import transforms, models
from PIL import Image

model = models.efficientnet_b0()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load("tree_classifier_efficientnet.pth", map_location="cpu"))
model.eval()

print("✅ Model loaded successfully\n")

classes = ["fruit_bearing", "non_fruit_bearing"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dir = "fruit_tree_dataset/test"

correct = 0
total = 0

print("📊 Testing Results with Full Breakdown:\n")


for class_name in classes:
    class_path = os.path.join(test_dir, class_name)

    for filename in os.listdir(class_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):

            img_path = os.path.join(class_path, filename)

            image = Image.open(img_path).convert("RGB")
            img = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(img)
                probs = torch.nn.functional.softmax(output[0], dim=0)

                confidence, pred = torch.max(probs, 0)
                predicted_class = classes[pred]

            actual_class = class_name
            is_correct = predicted_class == actual_class

            if is_correct:
                correct += 1
            total += 1

            print(f"{filename}")
            print(f"   Actual: {actual_class}")
            print(f"   Model: {predicted_class} ({confidence.item()*100:.2f}%)")

            print("   Breakdown:")
            for i, cname in enumerate(classes):
                print(f"      {cname}: {probs[i].item()*100:.2f}%")

            print(f"   Match: {'✅ Correct' if is_correct else '❌ Wrong'}\n")

if total > 0:
    accuracy = (correct / total) * 100
    print("===================================")
    print(f"📈 Overall Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("===================================")
else:
    print("⚠️ No images found in test folder")