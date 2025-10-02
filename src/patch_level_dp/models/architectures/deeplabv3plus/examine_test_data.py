import numpy as np
from PIL import Image
import os
import glob
from datasets.cityscapes import Cityscapes

def examine_dataset():
    print("Examining Cityscapes dataset structure...")
    
    # Check if the test directory exists
    test_img_dir = '/nfs/shared/cityscapes/leftImg8bit/test'
    test_gt_dir = '/nfs/shared/cityscapes/gtFine/test'
    
    print(f"Image directory exists: {os.path.exists(test_img_dir)}")
    print(f"GT directory exists: {os.path.exists(test_gt_dir)}")
    
    # Count files
    test_images = glob.glob(os.path.join(test_img_dir, '**/*.png'), recursive=True)
    test_labels = glob.glob(os.path.join(test_gt_dir, '**/*_labelIds.png'), recursive=True)
    
    print(f"Number of test images: {len(test_images)}")
    print(f"Number of test labels: {len(test_labels)}")
    
    # Check a few sample images and labels
    if len(test_images) > 0 and len(test_labels) > 0:
        # Sample a few images and their corresponding labels
        for i in range(min(3, len(test_labels))):
            label_path = test_labels[i]
            base_name = os.path.basename(label_path).replace('_gtFine_labelIds.png', '')
            city = os.path.basename(os.path.dirname(label_path))
            
            img_path = os.path.join(test_img_dir, city, f"{base_name}_leftImg8bit.png")
            
            if os.path.exists(img_path):
                print(f"\nExamining pair {i+1}:")
                print(f"Image: {img_path}")
                print(f"Label: {label_path}")
                
                # Load and analyze the image
                img = np.array(Image.open(img_path))
                print(f"Image shape: {img.shape}")
                print(f"Image data type: {img.dtype}")
                print(f"Image value range: [{np.min(img)}, {np.max(img)}]")
                
                # Load and analyze the label
                label = np.array(Image.open(label_path))
                print(f"Label shape: {label.shape}")
                print(f"Label data type: {label.dtype}")
                print(f"Label unique values: {np.unique(label)}")
                print(f"Number of pixels labeled as 255 (ignored): {np.sum(label == 255)}")
                print(f"Percentage of ignored pixels: {np.sum(label == 255) / (label.shape[0] * label.shape[1]) * 100:.2f}%")
            else:
                print(f"Couldn't find matching image for {label_path}")
    
    # Now examine using the dataset class
    print("\n===== Using Cityscapes dataset class =====")
    try:
        test_dataset = Cityscapes(root='/nfs/shared/cityscapes', split='test')
        print(f"Successfully created test dataset with {len(test_dataset)} samples")
        
        for i in range(min(3, len(test_dataset))):
            img, label = test_dataset[i]
            print(f"\nSample {i+1} from dataset class:")
            print(f"Image type: {type(img)}")
            print(f"Image size: {img.size}")
            
            label_np = np.array(label)
            print(f"Label shape: {label_np.shape}")
            print(f"Label unique values: {np.unique(label_np)}")
            print(f"Number of pixels labeled as 255 (ignored): {np.sum(label_np == 255)}")
            print(f"Percentage of ignored pixels: {np.sum(label_np == 255) / (label_np.shape[0] * label_np.shape[1]) * 100:.2f}%")
            
            # Check if any label values are actual valid classes (not 255)
            valid_pixels = np.sum(label_np != 255)
            if valid_pixels == 0:
                print("WARNING: This label has NO valid pixels (all 255/ignored)!")
            else:
                print(f"Label has {valid_pixels} valid pixels")
    
    except Exception as e:
        print(f"Error while using dataset class: {e}")

if __name__ == "__main__":
    examine_dataset() 