import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os


st.write("working")
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["general"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True
    
# Define the model architecture (SimpleCNN)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
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

if check_password():

    
    st.title("Welcome to the Pressure Mapping App!")

    # Load model
    @st.cache_resource
    def load_model(path="best_binary_model.pth"):
        model = SimpleCNN(num_classes=2)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model
    model = load_model()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # Online learning function
    def fine_tune_model(model, image_tensor, label_tensor, epochs=3):
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(image_tensor)
            loss = criterion(outputs, label_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        return model

    # Streamlit UI
    st.title("Pressure Mapping Image Classifier")
    learning_mode = st.checkbox("Enable Learning Mode")
    uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)

            image_tensor = transform(image).unsqueeze(0)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            label = "Good" if predicted_class.item() == 1 else "Bad"
            st.write(f"Prediction: **{label}** with confidence {confidence.item():.4f}")

            if learning_mode:
                user_label = st.radio(f"Label for learning ({uploaded_file.name}):", ["Good", "Bad"], key=uploaded_file.name)
                label_tensor = torch.tensor([1 if user_label == "Good" else 0])
                model = fine_tune_model(model, image_tensor, label_tensor)
                st.success(f"Model updated with label: {user_label}")

        if learning_mode:
            save_model = st.button("Save Updated Model")
            if save_model:
                st.header("THIS WILL PERMANENTLY CHANGE THE MODEL TO LEARN FROM YOUR IMAGE(s)", )
                torch.save(model.state_dict(), "best_binary_model.pth", _use_new_zipfile_serialization=True)
                st.success("Model saved to best_binary_model.pth")

