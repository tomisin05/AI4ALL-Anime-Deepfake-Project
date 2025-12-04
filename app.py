import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
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
    # Page config
    st.set_page_config(page_title="Anime AI Detector", page_icon="🎭", layout="wide")
    
    # Header
    st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>🎭 Anime Face AI Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #666;'>Detect if anime character faces are AI-generated or human-created</p>", unsafe_allow_html=True)
    
    # Info section
    with st.expander("ℹ️ How it works"):
        st.write("""
        This AI model uses a hybrid CNN-Vision Transformer architecture to analyze anime character faces and determine if they were:
        - 🎨 **Human-created**: Traditional hand-drawn or digitally illustrated by artists
        - 🤖 **AI-generated**: Created using artificial intelligence tools
        """)
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("❌ Could not load the model. Please check your internet connection.")
        return
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # File uploader with better styling
    st.markdown("### 📤 Upload Anime Face Image")
    uploaded_file = st.file_uploader(
        "Choose an anime character face image...", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
        help="Upload clear images of anime character faces for best results"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        # # Enhance image for display
        # width, height = image.size
        # if width < 300 or height < 300:
        #     # Upscale small images
        #     scale_factor = max(300 / width, 300 / height)
        #     new_size = (int(width * scale_factor), int(height * scale_factor))
        #     display_image = image.resize(new_size, Image.LANCZOS)
        # else:
        #     display_image = image

        display_image = image.resize((64, 64), Image.LANCZOS)
        
        # Enhance sharpness and contrast
        enhancer = ImageEnhance.Sharpness(display_image)
        display_image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(display_image)
        display_image = enhancer.enhance(1.1)
        
        # Layout with better spacing
        col1, col2, col3 = st.columns([1, 0.1, 1])
        
        with col1:
            st.markdown("#### 🖼️ Uploaded Image")
            st.image(display_image, use_column_width=True)
        
        with col3:
            # Make prediction
            with st.spinner("🔍 Analyzing anime face..."):
                transformed_image = transform(image)
                predicted_class, probabilities = predict_image(model, transformed_image, device)
                
                class_names = {0: "Human-created", 1: "AI-generated"}
                prediction = class_names[predicted_class]
                confidence = probabilities[predicted_class] * 100
                
                # Display results with better styling
                st.markdown("#### 📊 Analysis Results")
                
                if predicted_class == 0:
                    st.success(f"🎨 **{prediction}** ({confidence:.1f}% confidence)")
                    result_color = "#28a745"
                else:
                    st.error(f"🤖 **{prediction}** ({confidence:.1f}% confidence)")
                    result_color = "#dc3545"
                
                # Probability bars with custom styling
                st.markdown("**Probability Breakdown:**")
                
                # Human-created probability
                human_prob = float(probabilities[0])
                st.markdown(f"🎨 Human-created: **{human_prob:.1%}**")
                st.progress(human_prob)
                
                # AI-generated probability  
                ai_prob = float(probabilities[1])
                st.markdown(f"🤖 AI-generated: **{ai_prob:.1%}**")
                st.progress(ai_prob)
                
        # Add some spacing
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #888; font-size: 14px;'>Powered by Hybrid CNN-Vision Transformer</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()