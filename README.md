# AI vs Human Art Detector

A Streamlit web application that uses a hybrid CNN-Vision Transformer model to detect whether anime/artwork images are AI-generated or human-drawn.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open the web interface in your browser
2. Upload an image (JPG, PNG, etc.)
3. View the prediction results with confidence scores

## Model

The app uses a pre-trained hybrid CNN-ViT model from Hugging Face Hub that combines:
- CNN feature extraction for spatial patterns
- Vision Transformer for attention-based analysis
- Binary classification (Human-drawn vs AI-generated)