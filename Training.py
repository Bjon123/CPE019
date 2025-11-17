# Training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# ----------------------------
# SETTINGS
# ----------------------------
DATASET_PATH = "dataset"   # folder containing subfolders for each car brand
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# ----------------------------
# IMAGE TRANSFORMS
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

# ----------------------------
# LOAD DATASET (automatic classes)
# ----------------------------
train_data = datasets.ImageFolder(DATASET_PATH, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)

class_names = train_data.classes
print("Classes found:", class_names)
print("Total images:", len(train_data))

# ----------------------------
# LOAD MODEL
# ----------------------------
num_classes = len(class_names)

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# TRAINING LOOP
# ----------------------------
print("Starting training...")

for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

print("Training complete!")

# ----------------------------
# SAVE MODEL AND CLASS NAMES
# ----------------------------
torch.save(model.state_dict(), "Model.h5")
# Save class names so App.py can read them
with open("classes.txt", "w") as f:
    for c in class_names:
        f.write(c + "\n")

print("Model saved as Model.h5")
print("Class names saved as classes.txt")
