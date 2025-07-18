#NOTE This script uses a custom convulutional neural network to do a binary classification on the pressure mapping images

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import shutil
import random
from torchvision.datasets import ImageFolder
import torch
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 16
NUM_EPOCHS = 25

# Image Folder for dataset 

IMAGE_FOLDER_SOURCE = r"C:\Users\11039638\Downloads\Pressure Mapping Image Binning" 
IMAGE_TEST_FOLDER = r"C:\Users\11039638\Downloads\Pressure Mapping Image Binning\HoldOutTestSetKFold"

if __name__ == "__main__":
    transform = transforms.Compose([  
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])

        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(15),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406],
        #                         [0.229, 0.224, 0.225])
        # ])


        #Get Dataset for all the images 


    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=2):
            super(SimpleCNN, self).__init__()
            
            self.conv_block = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input: 3x224x224 → 32x224x224
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),  # → 32x112x112

                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64x112x112
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # → 64x56x56

                nn.Conv2d(64, 128, kernel_size=3, padding=1),  # → 128x56x56
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),  # → 128x28x28
            )

            self.fc_block = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 28 * 28, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.conv_block(x)
            x = self.fc_block(x)
            return x

    def create_model():
        model = SimpleCNN(num_classes=2)
        return model
    
    def split_dataset(source_dir, dest_dir, split_ratios=(0.7, 0.15, 0.15), seed=42):
        random.seed(seed)
        classes = ['Good', 'Bad']
        subsets = ['BinaryTrain', 'BinaryVal', 'BinaryTest']

        for subset in subsets:
            for cls in classes:
                os.makedirs(os.path.join(dest_dir, subset, cls), exist_ok=True)

        # Create destination directories

        for cls in classes:
            class_dir = os.path.join(source_dir, cls)
            images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            random.shuffle(images)

            total = len(images)
            train_end = int(split_ratios[0] * total)
            val_end = train_end + int(split_ratios[1] * total)

            split_data = {
                'BinaryTrain': images[:train_end],
                'BinaryVal': images[train_end:val_end],
                'BinaryTest': images[val_end:]
            }

            for subset in subsets:
                for img in split_data[subset]:
                    src_path = os.path.join(class_dir, img)
                    dst_path = os.path.join(dest_dir, subset, cls, img)
                    shutil.copy2(src_path, dst_path)

    train_path = os.path.join(IMAGE_FOLDER_SOURCE, r"BinaryTrain")
    val_path = os.path.join(IMAGE_FOLDER_SOURCE, r"BinaryVal")
    test_path = os.path.join(IMAGE_FOLDER_SOURCE, r"BinaryTest")

    #Load in training and validation data 

    train_data = ImageFolder(train_path, transform=transform)
    val_data = ImageFolder(val_path, transform=transform)
    test_data = ImageFolder(test_path, transform = transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    def evaluate(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy

    def test_model(model, dataloader, device):
        model.eval()
        all_preds = []
        all_labels = []
        all_confidences = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)

                confidence, preds = torch.max(probabilities, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_confidences.extend(confidence.cpu().numpy())

        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['bad', 'good']))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

        print("\nPredictions with Confidence Scores:")
        for i, (pred, conf) in enumerate(zip(all_preds, all_confidences)):
            label = 'good' if pred == 1 else 'bad'
            print(f'Sample {i+1}: Predicted = {label}, Confidence = {conf:.4f}')
    
    def train_model(model, train_loader, val_loader, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00005)
        model.to(device)
        loss_arr = []
        eval_loss_arr = []
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        best_model_state = None
        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss, acc = evaluate(model, val_loader, criterion, device)
            eval_loss_arr.append(loss)
            loss_arr.append(epoch_loss/len(train_loader))
           
            avg_train_loss = epoch_loss / len(train_loader)
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

       
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    
        if best_model_state:
            model.load_state_dict(best_model_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model()
    train_model(model, train_loader=train_loader, val_loader=val_loader, device=device)
    torch.save(model.state_dict(), "best_binary_model.pth")

    test_model(model, test_loader, device)