import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Explicit session lists for reproducible dataset splits, adapted from supervisor's script.
# Using the unlabeled sessions for training as we are not in an active learning scenario.
TRAIN_SESSIONS = [
    "20181107_132730", "20181108_091945", "20181107_133258",
    "20181108_084007", "20180807_145028", "20180810_142822",
    "20180925_135056", "20181008_095521", "20181107_132300",
    "20181204_154421", "20181204_170238"
]
VAL_SESSIONS = [
    "20180925_101535", "20181016_125231", "20181204_135952"
]
TEST_SESSIONS = [
    "20180925_124435", "20181108_123750", "20181108_103155"
]

class A2D2Dataset(Dataset):
    """
    A2D2 Dataset Loader
    https://www.a2d2.audi/a2d2/en/dataset.html
    Refactored to include explicit data splits and optional class grouping.
    """
    def __init__(self, root, split='training', transform=None, use_group_labels=True):
        self.root = root
        self.split = split
        self.transform = transform
        self.use_group_labels = use_group_labels
        self.files = []
        
        # --- Manifest Caching Logic ---
        manifest_filename = f'a2d2_{split}_{"grouped" if use_group_labels else "full"}_manifest.json'
        manifest_path = os.path.join(os.path.dirname(self.root), manifest_filename)
        
        if os.path.exists(manifest_path):
            print(f"Attempting to load file list from manifest: {manifest_path}")
            with open(manifest_path, 'r') as f:
                self.files = json.load(f)
            print(f"Loaded {len(self.files)} files from manifest.")

        if not self.files:
            if os.path.exists(manifest_path):
                print("Manifest was empty or failed to load. Re-scanning...")
            else:
                print(f"Manifest not found. Scanning for {split} images in {self.root}...")

            if self.split == 'training':
                sequences_to_scan = TRAIN_SESSIONS
            elif self.split == 'validation':
                sequences_to_scan = VAL_SESSIONS
            elif self.split == 'test':
                sequences_to_scan = TEST_SESSIONS
            else:
                raise ValueError(f"Invalid split '{self.split}'. Choose from 'training', 'validation', 'test'.")

            available_sequences = {s for s in os.listdir(root) if s.startswith('2018') and os.path.isdir(os.path.join(root, s))}
            
            sequences_to_scan_found = []
            for seq in sequences_to_scan:
                if seq in available_sequences:
                    sequences_to_scan_found.append(seq)
                else:
                    print(f"Warning: Specified sequence '{seq}' for split '{self.split}' not found in dataset at {self.root}")

            for seq in tqdm(sequences_to_scan_found, desc=f"Scanning {self.split} sequences"):
                seq_path = os.path.join(self.root, seq)
                camera_folder = os.path.join(seq_path, 'camera')
                label_folder = os.path.join(seq_path, 'label')
                
                if not os.path.isdir(camera_folder) or not os.path.isdir(label_folder):
                    continue

                cameras = os.listdir(camera_folder)
                
                for cam in cameras:
                    img_dir = os.path.join(camera_folder, cam)
                    label_dir = os.path.join(label_folder, cam)
                    
                    if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
                        continue

                    for img_filename in os.listdir(img_dir):
                        if not img_filename.endswith('.png'):
                            continue
                        
                        label_filename = img_filename.replace('camera', 'label')
                        image_path = os.path.join(img_dir, img_filename)
                        label_path = os.path.join(label_dir.replace('label', 'categorical_label'), label_filename)

                        if os.path.exists(label_path):
                            self.files.append({
                                'image': image_path,
                                'label': label_path
                            })
            
            if self.files:
                print(f"Saving file list to manifest: {manifest_path}")
                with open(manifest_path, 'w') as f:
                    json.dump(self.files, f)

        # The label conversion is now done in the pre-processing script.
        # We only need to know the number of classes.
        if self.use_group_labels:
            self.num_classes = 18  # Based on class_group_mapping2.json
        else:
            # If not using grouped labels, you would need a different pre-processing script
            # that saves the 55-class IDs.
            self.num_classes = 55 

        if not self.files:
            raise RuntimeError(f"No valid images found for split {self.split}.")
            
        print(f"Found {len(self.files)} valid images for the {self.split} split.")
        print(f"Number of classes: {self.num_classes}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafile = self.files[index]
        image_path = datafile['image']
        label_path = datafile['label']

        image = Image.open(image_path).convert('RGB')
        # The new label is a single-channel image with class IDs, so we load it in grayscale ('L') mode.
        target = Image.open(label_path).convert('L')

        if self.transform:
            image, target = self.transform(image, target)
        
        target = np.array(target, dtype=np.int64)
        return image, torch.from_numpy(target) 