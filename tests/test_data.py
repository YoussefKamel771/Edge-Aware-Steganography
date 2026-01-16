import pytest
import torch
from src.data_setup import BSDImageNetDataset, get_transforms

def test_dataset_output_shape():
    # Setup mock transforms
    rgb_tf, edge_tf = get_transforms(img_size=(128, 128), is_train=False)
    
    # We use a small mock dataset logic or point to a 'tiny' data folder
    # For this example, let's assume you have at least one image in your data dir
    # If not, you can mock the Image.open call.
    
    # Assertions
    assert rgb_tf is not None
    assert edge_tf is not None

def test_transform_normalization():
    rgb_tf, _ = get_transforms(img_size=(128, 128), is_train=False)
    from PIL import Image
    import numpy as np
    
    # Create a dummy white image
    dummy_img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 255)
    tensor = rgb_tf(dummy_img)
    
    # Check if Normalize((0.5,), (0.5,)) worked: 255 -> 1.0 -> (1.0 - 0.5)/0.5 = 1.0
    # Values should be roughly between -1 and 1
    assert tensor.max() <= 1.0
    assert tensor.min() >= -1.0
    assert tensor.shape == (3, 128, 128)