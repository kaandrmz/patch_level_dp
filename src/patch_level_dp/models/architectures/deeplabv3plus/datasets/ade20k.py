import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm

class ADE20KDataset(Dataset):
    """
    ADE20K Dataset Loader
    http://groups.csail.mit.edu/vision/datasets/ADE20K/
    """
    def __init__(self, root, split='training', transform=None, min_dim=500):
        self.root = root
        self.split = split
        self.transform = transform
        self.files = []

        self.image_dir = os.path.join(self.root, 'images', self.split)
        self.label_dir = os.path.join(self.root, 'annotations', self.split)

        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            raise RuntimeError(f'Dataset not found or incomplete. Please make sure directory exists at {self.root}')
        
        # Scan for all files and filter them by size
        print(f"Scanning and filtering images in {self.image_dir} (min dimension: {min_dim}x{min_dim})...")
        all_files = [os.path.basename(path) for path in glob(os.path.join(self.image_dir, '*.jpg'))]
        
        for filename in tqdm(all_files):
            image_path = os.path.join(self.image_dir, filename)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width >= min_dim and height >= min_dim:
                        self.files.append(filename)
            except Exception as e:
                print(f"Warning: Could not read image {image_path}, skipping. Error: {e}")

        if not self.files:
            raise RuntimeError(f"No valid images found in {self.image_dir} for split {self.split} with min dimensions {min_dim}x{min_dim}.")
            
        print(f"Found {len(self.files)} valid images for the {self.split} split.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_filename = self.files[index]
        image_path = os.path.join(self.image_dir, image_filename)
        label_path = os.path.join(self.label_dir, image_filename.replace('.jpg', '.png'))

        image = Image.open(image_path).convert('RGB')
        target = Image.open(label_path)

        if self.transform:
            image, target = self.transform(image, target)
        
        # Remap labels: 0->255 (ignore), 1-150 -> 0-149
        target = np.array(target, dtype=np.int64)
        target = target - 1  # 0 (bg) becomes -1, 1-150 become 0-149
        target[target == -1] = 255 # Map ignored background to 255
        
        return image, torch.from_numpy(target) 