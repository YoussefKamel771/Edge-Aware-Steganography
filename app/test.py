# app/main.py
"""
FastAPI deployment for Edge-Constrained Image Steganography model.
"""
import io
import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from typing import Optional
import base64
from pydantic import BaseModel

# Import model utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import load_checkpoint
from src.data_setup import get_transforms

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (256, 256)
EDGE_THRESHOLD = 0.12

# ============================================================================
# Initialize FastAPI App
# ============================================================================
app = FastAPI(
    title="Image Steganography API",
    description="Hide and reveal secret images using edge-constrained steganography",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global Model Loading
# ============================================================================
model = None
rgb_transform = None
edge_transform = None

@app.on_event("startup")
async def load_model():
    """Load model on startup to avoid repeated loading."""
    global model, rgb_transform, edge_transform
    try:
        model = load_checkpoint(MODEL_PATH, device=DEVICE)
        rgb_transform, edge_transform = get_transforms(img_size=IMG_SIZE, is_train=False)
        print(f"✓ Model loaded successfully on {DEVICE}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise

# ============================================================================
# Helper Functions
# ============================================================================
def process_image(file: UploadFile, is_edge: bool = False) -> torch.Tensor:
    """Process uploaded image file into tensor."""
    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if is_edge:
            image = image.convert('L')
            tensor = edge_transform(image).unsqueeze(0).to(DEVICE)
            tensor = (tensor > EDGE_THRESHOLD).float()
        else:
            image = image.convert('RGB')
            tensor = rgb_transform(image).unsqueeze(0).to(DEVICE)
        
        return tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Denormalize from [-1, 1] to [0, 1]
    img_array = tensor[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def pil_to_bytes(image: Image.Image, format: str = "PNG") -> io.BytesIO:
    """Convert PIL Image to bytes."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr

# ============================================================================
# Pydantic Models
# ============================================================================
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

class HideResponse(BaseModel):
    stego_image: str  # base64 encoded
    message: str

class RevealResponse(BaseModel):
    recovered_secret: str  # base64 encoded
    message: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Image Steganography API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "hide": "/hide",
            "reveal": "/reveal",
            "hide_return_image": "/hide/image",
            "reveal_return_image": "/reveal/image"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(DEVICE)
    )

@app.post("/hide", response_model=HideResponse)
async def hide_secret(
    cover: UploadFile = File(..., description="Cover image (JPG/PNG)"),
    secret: UploadFile = File(..., description="Secret image to hide (JPG/PNG)"),
    edge: UploadFile = File(..., description="Edge map (grayscale image)")
):
    """
    Hide a secret image inside a cover image using edge constraints.
    Returns base64-encoded stego image.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process inputs
        cover_tensor = process_image(cover, is_edge=False)
        secret_tensor = process_image(secret, is_edge=False)
        edge_tensor = process_image(edge, is_edge=True)
        
        # Run inference
        with torch.no_grad():
            stego, _, _ = model.hide_network(cover_tensor, secret_tensor, edge_tensor)
        
        # Convert to image
        stego_image = tensor_to_image(stego)
        stego_base64 = image_to_base64(stego_image)
        
        return HideResponse(
            stego_image=stego_base64,
            message="Secret successfully hidden in cover image"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hiding failed: {str(e)}")

@app.post("/hide/image")
async def hide_secret_return_image(
    cover: UploadFile = File(...),
    secret: UploadFile = File(...),
    edge: UploadFile = File(...)
):
    """
    Hide a secret image and return the stego image as a downloadable file.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process inputs
        cover_tensor = process_image(cover, is_edge=False)
        secret_tensor = process_image(secret, is_edge=False)
        edge_tensor = process_image(edge, is_edge=True)
        
        # Run inference
        with torch.no_grad():
            stego, _, _ = model.hide_network(cover_tensor, secret_tensor, edge_tensor)
        
        # Convert to image
        stego_image = tensor_to_image(stego)
        img_bytes = pil_to_bytes(stego_image, format="PNG")
        
        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=stego_image.png"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hiding failed: {str(e)}")

@app.post("/reveal", response_model=RevealResponse)
async def reveal_secret(
    stego: UploadFile = File(..., description="Stego image containing hidden secret"),
    edge: UploadFile = File(..., description="Edge map used during hiding")
):
    """
    Reveal the hidden secret image from a stego image.
    Returns base64-encoded recovered secret.
    
    Note: This endpoint requires the intermediate features from the hide network.
    For production use, consider storing these features or using the full pipeline.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    raise HTTPException(
        status_code=501,
        detail="Reveal endpoint requires hide network features. Use /hide_and_reveal instead."
    )

@app.post("/hide_and_reveal")
async def hide_and_reveal(
    cover: UploadFile = File(...),
    secret: UploadFile = File(...),
    edge: UploadFile = File(...)
):
    """
    Complete pipeline: hide secret, then immediately reveal it.
    Returns both stego and recovered images as base64.
    Useful for testing and demonstration.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process inputs
        cover_tensor = process_image(cover, is_edge=False)
        secret_tensor = process_image(secret, is_edge=False)
        edge_tensor = process_image(edge, is_edge=True)
        
        # Run full pipeline
        with torch.no_grad():
            stego, secret_feat, fused = model.hide_network(cover_tensor, secret_tensor, edge_tensor)
            recovered = model.reveal_network(stego, edge_tensor, secret_feat, fused)
        
        # Convert to images
        stego_image = tensor_to_image(stego)
        recovered_image = tensor_to_image(recovered)
        
        return JSONResponse({
            "stego_image": image_to_base64(stego_image),
            "recovered_secret": image_to_base64(recovered_image),
            "message": "Pipeline completed successfully"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@app.post("/reveal/image")
async def reveal_secret_return_image(
    stego: UploadFile = File(...),
    edge: UploadFile = File(...),
    # These would need to be stored/transmitted separately in production
    secret_feat: Optional[UploadFile] = File(None),
    fused: Optional[UploadFile] = File(None)
):
    """
    Reveal secret and return as downloadable image.
    Note: Requires intermediate features from hide network.
    """
    raise HTTPException(
        status_code=501,
        detail="This endpoint requires implementation of feature serialization"
    )

# ============================================================================
# Error Handlers
# ============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# ============================================================================
# Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# ============================================================================