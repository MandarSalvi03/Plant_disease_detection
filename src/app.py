import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from model import build_model

# Constants and configuration.
IMAGE_RESIZE = 224
CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model loading
@st.cache(allow_output_mutation=True)
def load_model():
    checkpoint = torch.load(r'C:\Users\salvi\Downloads\input\outputs\model.pth', map_location=torch.device('cpu'))
    model = build_model(pretrained=False, fine_tune=False, num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image).convert('RGB')
    image = test_transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image, model):
    with torch.no_grad():
        image = image.to(DEVICE)
        outputs = model(image)
        _, predicted_class = torch.max(outputs.data, 1)
        return CLASS_NAMES[predicted_class.item()]

# Streamlit UI
st.title("Plant Disease Detection")
st.write("Upload an image of the plant leaf to detect the disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image_tensor = preprocess_image(uploaded_file)
    
    # Load the model
    model = load_model()
    
    # Make prediction
    prediction = predict(image_tensor, model)
    
    # Display the prediction
    st.write(f"Prediction: **{prediction}**")
