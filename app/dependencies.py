import io, os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.utils import load_checkpoint
from dotenv import load_dotenv
from src.data_setup import get_transforms

load_dotenv()

# Load model globally (Singleton pattern)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH")
model = load_checkpoint(MODEL_PATH, device=DEVICE)

# Define the same transforms used during training
rgb_transform, edge_transform = get_transforms()

def preprocess_image(image_bytes, is_edge=False):
    """Converts raw bytes to a normalized torch tensor."""
    img = Image.open(io.BytesIO(image_bytes))
    if is_edge:
        img = img.convert('L')
        tensor = edge_transform(img)
        return (tensor > 0.12).float().unsqueeze(0).to(DEVICE)
    
    img = img.convert('RGB')
    return rgb_transform(img).unsqueeze(0).to(DEVICE)

def postprocess_tensor(tensor):
    """Converts a torch tensor back to a bytes-ready PIL Image."""
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor * 0.5 + 0.5  # Denormalize
    tensor = torch.clamp(tensor, 0, 1)
    img = transforms.ToPILImage()(tensor)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()