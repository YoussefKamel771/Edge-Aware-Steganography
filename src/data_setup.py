"""
Contains functionality for creating PyTorch DataLoaders for 
BSDS500 dataset.
"""
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BSDImageNetDataset(Dataset):
    def __init__(self, cover_dir, secret_dir, edge_dir, limit=None, rgb_transform=None, edge_transform=None, is_val=False):
        self.cover_dir = cover_dir
        self.secret_dir = secret_dir
        self.edge_dir = edge_dir
        self.rgb_transform = rgb_transform
        self.edge_transform = edge_transform

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.cover_images = sorted([f for f in os.listdir(cover_dir) if f.lower().endswith(valid_extensions)])
        self.secret_images = sorted([f for f in os.listdir(secret_dir) if f.lower().endswith(valid_extensions)])
        self.edge_images = sorted([f for f in os.listdir(edge_dir) if f.lower().endswith(valid_extensions)])

        # Find common image filenames between cover and edge (based on filename without extension)
        cover_names = set([os.path.splitext(f)[0] for f in self.cover_images])
        edge_names = set([os.path.splitext(f)[0] for f in self.edge_images])

        # Get common names between cover and edge directories only
        common_names = list(cover_names.intersection(edge_names))
        common_names.sort()  # Sort for consistency

        # Limit the number of images if specified
        if limit and limit < len(common_names):
            common_names = common_names[:limit]

        # Create filtered lists for cover and edge images
        self.cover_images = [f for f in self.cover_images if os.path.splitext(f)[0] in common_names]
        self.edge_images = [f for f in self.edge_images if os.path.splitext(f)[0] in common_names]

        # For secret images, just take the first 'limit' or use all if no limit
        if limit is not None and limit < len(self.secret_images):
            self.secret_images = self.secret_images[:-limit-1:-1]
        elif limit is not None and limit < len(self.secret_images):
            self.secret_images = self.secret_images[:limit]
        else:
            self.secret_images = self.secret_images[:len(self.cover_images)]

        # Make sure we have exactly the same number of images now
        assert len(self.cover_images) == len(self.secret_images) == len(self.edge_images), \
            f"Failed to match images across directories: cover={len(self.cover_images)}, secret={len(self.secret_images)}, edge={len(self.edge_images)}"

        print(f"Dataset created with {len(self.cover_images)} images (limit={limit})")

    def __len__(self):
        return len(self.cover_images)

    def __getitem__(self, idx):
        cover_path = os.path.join(self.cover_dir, self.cover_images[idx])
        secret_path = os.path.join(self.secret_dir, self.secret_images[idx])
        edge_path = os.path.join(self.edge_dir, self.edge_images[idx])
        cover_name = os.path.basename(cover_path).replace(".jpg", "")

        try:
            cover = Image.open(cover_path).convert('RGB')
            secret = Image.open(secret_path).convert('RGB')
            edge = Image.open(edge_path).convert('L')
        except Exception as e:
            raise RuntimeError(f"Failed to load image at index {idx}: {e}\n"
                              f"Cover: {cover_path}\nSecret: {secret_path}\nEdge: {edge_path}")

        if self.rgb_transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            cover = self.rgb_transform(cover)
            torch.manual_seed(seed)
            secret = self.rgb_transform(secret)
        if self.edge_transform:
            torch.manual_seed(seed)
            edge = self.edge_transform(edge)

        edge = (edge > 0.12).float()  # Binary threshold

        return cover, secret, edge, cover_name
    
def get_transforms(img_size=(256, 256), is_train=True):
    """Returns the transformation pipeline for RGB and Edge images."""
    if is_train:
        rgb_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        rgb_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    edge_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    return rgb_tf, edge_tf

def create_dataloader(config, mode='train'):
    """
    Creates a PyTorch DataLoader for the specified mode.

    Args:
        config (dict): Dictionary containing paths, image sizes, and batch settings.
        mode (str): Execution mode, either 'train' or 'val'.

    Returns:
        DataLoader: A configured PyTorch DataLoader object.
    """
    is_train = (mode == 'train')
    
    # Ensure img_size is a tuple (YAML loads it as a list)
    img_size = tuple(config['img_size'])
    
    rgb_tf, edge_tf = get_transforms(img_size=img_size, is_train=is_train)
    
    dataset = BSDImageNetDataset(
        cover_dir=config[f'{mode}_cover_dir'],
        secret_dir=config[f'{mode}_secret_dir'],
        edge_dir=config[f'{mode}_edge_dir'],
        limit=config.get(f'{mode}_limit'),
        rgb_transform=rgb_tf,
        edge_transform=edge_tf,
        is_val=not is_train
    )
    
    return DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=is_train, 
        num_workers=config['num_workers'],
        pin_memory=True
    )