import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import numpy as np

# Model architecture (same as your original)
class Hybrid_CNN_Vit(nn.Module):
    def __init__(self, image_size=64, num_classes=2, cnn_channels=32, num_heads=4, num_layers=1, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            nn.Conv2d(64, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
        )
        self.feature_h = image_size // 8
        self.feature_w = image_size // 8
        self.seq_len = self.feature_h * self.feature_w
        self.embedded_dim = cnn_channels
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.seq_len, self.embedded_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedded_dim,
            nhead=num_heads,
            dim_feedforward=self.embedded_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.clasifier = nn.Sequential(
            nn.LayerNorm(self.embedded_dim),
            nn.Dropout(0.6),
            nn.Linear(self.embedded_dim, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x += self.positional_embeddings
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.clasifier(x)
        return logits

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        downloaded_model_path = hf_hub_download(
            repo_id="Tomisin05/anime-ai-human-detector",
            filename="best_model.pth"
        )
        model = Hybrid_CNN_Vit(dropout=0.3).to(device)
        model.load_state_dict(torch.load(downloaded_model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def predict_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), probabilities.cpu().numpy()[0]

def main():
    st.title("🎨 AI vs Human Anime Face Detector")
    st.write("Upload an anime/artwork image to detect if it's AI-generated or human-drawn")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("Could not load the model. Please check your internet connection.")
        return
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        display_image = image.resize((64, 64))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(display_image, caption="Uploaded Image (64x64)", use_column_width=True)
        
        with col2:
            # Make prediction
            with st.spinner("Analyzing image..."):
                transformed_image = transform(image)
                predicted_class, probabilities = predict_image(model, transformed_image, device)
                
                class_names = {0: "Human-drawn", 1: "AI-generated"}
                prediction = class_names[predicted_class]
                confidence = probabilities[predicted_class] * 100
                
                # Display results
                st.subheader("Prediction Results")
                
                if predicted_class == 0:
                    st.success(f"🎨 **{prediction}**")
                else:
                    st.warning(f"🤖 **{prediction}**")
                
                st.write(f"**Confidence:** {confidence:.1f}%")
                
                # Progress bars for probabilities
                st.write("**Detailed Probabilities:**")
                st.write(f"Human-drawn: {probabilities[0]:.3f}")
                st.progress(float(probabilities[0]))
                st.write(f"AI-generated: {probabilities[1]:.3f}")
                st.progress(float(probabilities[1]))

if __name__ == "__main__":
    main()